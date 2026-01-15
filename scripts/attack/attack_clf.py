import json
import os
import time

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from gcg_clf import GCGConfig, run_gcg
from sklearn.model_selection import train_test_split

from clfextract.classifiers import Classifier
from clfextract.configs import set_config
from clfextract.datasets import ParquetManager
from clfextract.lenses.embeddings import EmbeddingLens
from clfextract.utils import VRAMMonitor, get_model_tag


class LinearClassifierNoSigmoid(Classifier):
    def __init__(self, input_dim: int, **kwargs):
        model = nn.Linear(input_dim, 1)
        super().__init__(model, criterion=nn.BCEWithLogitsLoss(), **kwargs)


class SurrogateClassifier:
    def __init__(self, model, head, lens):
        self.model = model
        self.head = head.model
        self.lens = lens
        self.device = model.device
        self.dtype = model.dtype

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    # This gets the logits without Sigmoid, as we use BCEWithLogitsLoss in GCG and during training
    def __call__(self, **kwargs):
        embeds = self.lens(**kwargs)[:, -1, -1, :].float()
        return self.head(embeds)


def load_data(
    mn: ParquetManager,
    model="llama2",
    lens_type="embedding",
    aggregation="last",
    attacks=["benign"],
    layer=None,
    device=None,
    source="advbench",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_DIR = os.getenv("DATA_DIR")
    if device is None:
        device = torch.device("cpu")
    x_list, y_list, y_pred_list = [], [], []

    layer = layer if layer is not None else end_layer

    print("Loading all data from Parquet file...")
    # Load all relevant files
    files = [
        f
        for f in os.listdir(DATA_DIR)
        if model in f and source in f and lens_type in f and not f.startswith("gcg")
    ]

    if len(files) == 0:
        raise FileNotFoundError("No files found matching the criteria.")

    print(f"Found {len(files)} files to load.")

    # Load and concatenate data from all files

    filters = {"layer": {"equals": layer}}
    columns = [
        "x",
        "num_positions",
        "layer",
        "y_true",
        "y_pred_protectai",
        "attack",
        "base",
    ]
    embedding_df = pl.concat(
        [
            mn.load_dataset(os.path.join(DATA_DIR, f), filters=filters, columns=columns)
            for f in files
        ]
    )

    embedding_df = embedding_df.rename({"y_pred_protectai": "y_pred"})
    x, y, y_pred = mn.load_torch_from_df(
        embedding_df, aggregation="last", device=device
    )
    bases = embedding_df["base"].to_numpy()

    return x, y, y_pred, bases


if __name__ == "__main__":
    config = set_config()
    print(config)

    tokenizer, model = config.exp.tokenizer, config.exp.model

    num_layers = model.config.num_hidden_layers
    layer = int(config.exp.num_layers)

    end_layer = num_layers - 1

    model_tag = get_model_tag(config.exp.model_path)

    mn = ParquetManager()

    x, y, y_pred, bases = load_data(
        mn,
        model=model_tag,
        lens_type="embedding",
        device=None,
        layer=layer,
        source="advbench",
    )

    classifier = LinearClassifierNoSigmoid(
        x.shape[-1], learning_rate=1e-3, device=model.device
    )
    (
        x_train,
        x_test,
        y_train,
        y_test,
        y_pred_train,
        y_pred_test,
        bases_train,
        bases_test,
    ) = train_test_split(
        x, y, y_pred, bases, test_size=0.2
    )  # Can use random state

    print("Training classifier...")
    num_epochs = 500
    patience = 15
    n_splits = 5
    clf_params = {
        "num_epochs": num_epochs,
        "patience": patience,
        "n_splits": n_splits,
    }
    classifier.train(x_train, y_pred_train, **clf_params)

    train_acc, train_f1 = classifier.evaluate(x_train, y_pred_train)
    test_acc, test_f1 = classifier.evaluate(x_test, y_pred_test)

    print("Pred Accuracy Train :", train_acc, "F1 Score Train :", train_f1)
    print("Pred Accuracy Test :", test_acc, "F1 Score Test :", test_f1)

    precision, recall, thresholds = classifier.precision_recall_curve(
        x_train, y_pred_train
    )

    f_score = 2 * (precision * recall) / (precision + recall)

    best_threshold = thresholds[np.argmax(f_score)]
    best_threshold = torch.sigmoid(torch.tensor(best_threshold)).item()
    print("Best Threshold: ", best_threshold)

    lens = EmbeddingLens(model, tokenizer, requires_grad_=True)

    surrogate_classifier = SurrogateClassifier(model.model, classifier, lens)

    dataset = config.exp.dataset
    results = []
    gcg_config = GCGConfig(
        num_steps=int(config.misc.num_steps),
        search_width=int(config.misc.search_width),
        topk=int(config.misc.topk),
        n_replace=1,
        verbosity="WARNING",
        use_prefix_cache=False,
    )

    metadata = {
        "attack": "gcg",
        "dataset": os.path.basename(config.exp.dataset_path),
        "model": model_tag,
        "model_path": config.exp.model_path,
        "attack_config": {
            "num_steps": gcg_config.num_steps,
            "search_width": gcg_config.search_width,
            "topk": gcg_config.topk,
        },
        "clf": {
            "type": classifier.__class__.__name__,
            "layer": layer,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "train_f1": train_f1,
            "test_f1": test_f1,
            "best_threshold": float(best_threshold),
        },
    }

    # Check if label is present in the dataset
    if "label" in config.exp.dataset.columns:
        target = 1 - int(config.exp.dataset["label"][0])
    else:
        target = 0 if "harmful" in config.exp.dataset_path else 1

    records = []

    # Filter the dataset so that it matches the test bases
    dataset = dataset[dataset["base"].isin(bases_test)].reset_index(drop=True)

    device_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    device_id = int(device_id) if device_id.isdigit() else 0
    # Initialize VRAM monitor
    vram_monitor = VRAMMonitor(device_id=device_id, interval=0.1)

    for i in range(len(dataset)):
        if i == 1:
            break
        base = dataset["base"][i]
        print(f"Running {i}/{len(dataset['base'])} : {base}")
        messages = [{"role": "user", "content": base}]
        gcg_config.target_class = target

        # Start measurements
        vram_monitor.start()
        start_time = time.time()
        result = run_gcg(surrogate_classifier, tokenizer, messages, gcg_config)
        # Stop measurements
        elapsed_time = time.time() - start_time
        vram_monitor.stop()

        vram_stats = vram_monitor.get_vram_stats()
        print(f"Time taken for run_gcg: {elapsed_time:.2f} seconds")
        print(
            f"VRAM Usage during run_gcg (MB): Low={vram_stats['low']:.2f}, High={vram_stats['high']:.2f}"
        )

        best_string, best_loss = result.best_string, result.best_loss
        strings, losses = result.strings, result.losses

        # Print loss of every 100th string
        for j in range(0, len(strings), 100):
            print(f"String: {strings[j]} Loss: {losses[j]}")

        print(f"Best String: {best_string} Loss: {best_loss}")
        config.prompt_manager.base = base
        prompt = config.prompt_manager.get_prompt("")

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        confidence_pre = surrogate_classifier(input_ids=input_ids)
        y_clf_base = torch.sigmoid(confidence_pre).item()

        # Every 50 steps until the num_steps
        eval_steps = list(range(50, gcg_config.num_steps + 1, 50))
        step_results = []
        for step in eval_steps:
            best_string_step = strings[losses.index(min(losses[:step]))]
            best_loss_step = min(losses[:step])

            prompt = config.prompt_manager.get_prompt(best_string_step)
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(
                model.device
            )
            confidence = surrogate_classifier(input_ids=input_ids)

            y_clf = float(torch.sigmoid(confidence.detach()[0][0]))

            step_result = {
                "step": step,
                "attack": best_string_step,
                "loss": best_loss_step,
            }
            step_results.append(step_result)

        prompt = config.prompt_manager.get_prompt(best_string)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        confidence = surrogate_classifier(input_ids=input_ids)
        y_clf = torch.sigmoid(confidence).item()

        record = {
            "base": base,
            "target": target,
            "attack": best_string,
            "loss": best_loss,
            "append_strat": "suffix",
            "y_clf_base": y_clf_base,
            "y_clf": y_clf,
            "step_results": step_results,
            "attacks": strings,
            "losses": losses,
            "time": elapsed_time,
            "vram_low": vram_stats["low"],
            "vram_high": vram_stats["high"],
        }

        records.append(record)

    results = {"metadata": metadata, "records": records}

    if not os.path.exists(config.exp.output_dir):
        os.makedirs(config.exp.output_dir)

    with open(
        os.path.join(
            config.exp.output_dir,
            f'gcg_surrogate_{model_tag}_{end_layer:02d}_layer{layer:02d}_gcg_{gcg_config.num_steps}steps_{gcg_config.topk}topk_{config.exp.dataset_path.split("/")[-1].replace(".json","")}_{config.exp.start:03d}_{config.exp.end:03d}.json',
        ),
        "w",
    ) as f:
        json.dump(results, f, indent=4)
