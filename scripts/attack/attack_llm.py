import json
import os
import time

import torch
from gcg_llm import GCGConfig, run_gcg
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          pipeline)

from clfextract.configs import set_config
from clfextract.evaluators import PipelineEvaluator
from clfextract.utils import VRAMMonitor, get_model_tag


def map_y_pred(y_pred):
    if not isinstance(y_pred, str):
        return None
    if y_pred.lower() == "normal":
        return False
    elif y_pred.lower() == "rejection":
        return True
    else:
        return None


if __name__ == "__main__":
    config = set_config()

    print(config)

    tokenizer, model = config.exp.tokenizer, config.exp.model
    start, end = config.exp.start, config.exp.end

    model_tag = get_model_tag(config.exp.model_path)

    dataset = config.exp.dataset
    records = []
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
        "model": model_tag,
        "dataset": os.path.basename(config.exp.dataset_path),
        "model_path": config.exp.model_path,
        "attack_config": {
            "num_steps": gcg_config.num_steps,
            "search_width": gcg_config.search_width,
            "topk": gcg_config.topk,
        },
    }

    clf_tokenizer = AutoTokenizer.from_pretrained(
        "ProtectAI/distilroberta-base-rejection-v1"
    )
    clf_model = AutoModelForSequenceClassification.from_pretrained(
        "ProtectAI/distilroberta-base-rejection-v1"
    )

    clf_pipeline = pipeline(
        "text-classification",
        model=clf_model,
        tokenizer=clf_tokenizer,
        truncation=True,
        max_length=512,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    evaluator = PipelineEvaluator(
        config.exp.model,
        config.exp.tokenizer,
        config.prompt_manager,
        clf_pipeline,
    )

    # Get device id from CUDA_VISIBLE_DEVICES

    device_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    device_id = int(device_id) if device_id.isdigit() else 0
    # Initialize VRAM monitor
    vram_monitor = VRAMMonitor(device_id=device_id, interval=0.1)

    for i in range(config.exp.start, config.exp.end):
        base = dataset["base"][i]
        target = dataset["target"][i]
        messages = [{"role": "user", "content": base}]

        # Start measurements
        start_time = time.time()
        vram_monitor.start()
        result = run_gcg(model, tokenizer, messages, target, gcg_config)
        # End measurements
        vram_monitor.stop()
        elapsed_time = time.time() - start_time

        vram_stats = vram_monitor.get_vram_stats()
        print(f"Time taken for run_gcg: {elapsed_time:.2f} seconds")
        print(
            f"VRAM Usage during run_gcg (MB): Low={vram_stats['low']:.2f}, High={vram_stats['high']:.2f}"
        )

        best_string, best_loss = result.best_string, result.best_loss
        strings, losses = result.strings, result.losses

        y_llm_base = map_y_pred(evaluator([{"base": base, "attack": ""}])[0])

        # Every 100 steps until the num_steps
        eval_steps = list(range(100, gcg_config.num_steps + 1, 100))

        step_results = []
        for step in eval_steps:
            # argmin of losses[:step]
            best_string_step = strings[losses.index(min(losses[:step]))]
            best_loss_step = min(losses[:step])

            y_llm = map_y_pred(
                evaluator([{"base": base, "attack": best_string_step}])[0]
            )
            success = abs(y_llm - y_llm_base)
            output = evaluator.logger[-1].get("output", None)
            step_result = {
                "step": step,
                "attack": best_string_step,
                "loss": best_loss_step,
                "y_llm_base": y_llm_base,
                "y_llm": y_llm,
                "success": success,
                "output": output,
            }
            step_results.append(step_result)

        # Final evaluation
        y_llm = map_y_pred(evaluator([{"base": base, "attack": best_string}])[0])
        success = abs(y_llm - y_llm_base)
        output = evaluator.logger[-1].get("output", None)
        records.append(
            {
                "base": base,
                "target": target,
                "attack": best_string,
                "loss": best_loss,
                "y_llm_base": y_llm_base,
                "y_llm": y_llm,
                "step_results": step_results,
                "success": success,
                "output": output,
                "append_strat": "suffix",
                "attacks": strings,
                "losses": losses,
                "time": elapsed_time,
                "vram_low": vram_stats["low"],
                "vram_high": vram_stats["high"],
            }
        )

    results = {"metadata": metadata, "records": records}

    if not os.path.exists(config.exp.output_dir):
        os.makedirs(config.exp.output_dir)

    with open(
        os.path.join(
            config.exp.output_dir,
            f'gcg_llm_{model_tag}_{gcg_config.num_steps}steps_{gcg_config.search_width}search_width_{gcg_config.topk}topk_{config.exp.dataset_path.split("/")[-1].replace(".json","")}_{start:03d}_{end:03d}.json',
        ),
        "w",
    ) as f:
        json.dump(results, f, indent=4)
