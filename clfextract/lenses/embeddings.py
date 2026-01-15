from typing import List, Optional, Union

import torch

from clfextract.lenses.lenses import Lens
from clfextract.utils import find_executable_batch_size, type_check


class EmbeddingLens(Lens):
    """
    Embedding class represents a lens for embeddings in general.

    Args:
        model (Model): The model used for generating embeddings.
        tokenizer (Tokenizer): The tokenizer used for tokenizing input.
        layers (Union[int, List[int]]): The layers of the model to extract embeddings from.
        style (str): The style of embeddings to extract.

    Attributes:
        model (Model): The model used for generating embeddings.
        tokenizer (Tokenizer): The tokenizer used for tokenizing input.
        layers (Union[int, List[int]]): The layers of the model to extract embeddings from.
        style (str): The style of embeddings to extract.

    Methods:
        __call__(self, input): Computes the last embeddings for the given input.
        distance(self, input_vec, target, p): Computes the distance between input and target embeddings.

    """

    def __init__(
        self,
        model,
        tokenizer,
        layers: Optional[Union[int, List[int]]] = None,
        positions: Optional[Union[int, List[int]]] = None,
        requires_grad_: bool = False,
    ):
        super().__init__(model, tokenizer)
        self.layers = [layers] if isinstance(layers, int) else layers
        self.positions = [positions] if isinstance(positions, int) else positions

        if self.layers is None:
            print("Warning: No layers specified. Defaulting to all layers.")
            self.layers = [l for l in range(model.config.num_hidden_layers + 1)]

        if self.positions is None:
            print("Warning: No positions specified. Defaulting to all positions.")

        self.target = None
        self.requires_grad_ = requires_grad_

        assert (
            min(self.layers) >= -model.config.num_hidden_layers
            and max(self.layers) <= model.config.num_hidden_layers
        ), "Invalid layer(s) specified"

    @find_executable_batch_size
    def __call__(
        batch_size,
        self,
        input: Optional[Union[str, List[str]]] = None,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_numpy: bool = False,
        unpad: bool = False,
    ) -> torch.Tensor:
        """
        Computes the context for the given input.
        """
        assert (
            input is not None or inputs_embeds is not None or input_ids is not None
        ), "Input tensor or embeddings must be provided"

        if input is not None:
            inputs = self.tokenizer(input, return_tensors="pt", padding=True).to(
                self.model.device
            )
            lengths = torch.sum(inputs.attention_mask, dim=1).cpu().numpy()
            num_inputs = inputs.input_ids.shape[0]
            max_length = inputs.input_ids.shape[1]

        elif input_ids is not None or inputs_embeds is not None:
            inputs = input_ids if input_ids is not None else inputs_embeds
            lengths = None
            num_inputs = inputs.shape[0]
            max_length = inputs.shape[1]
            # Ensure the tensor for inputs is on the correct device and requires grad if it's the one we're optimizing
            # This is implicitly handled when input_ids/inputs_embeds come from `optim_ids_onehot` which requires grad.
            if input_ids is not None:
                inputs = inputs.to(self.model.device)
            elif inputs_embeds is not None:
                inputs = inputs.to(self.model.device)

        positions = (
            [i for i in range(max_length)] if self.positions is None else self.positions
        )

        emb = torch.zeros(
            (
                num_inputs,
                len(self.layers),
                len(positions),
                self.model.config.hidden_size,
            ),
            device=self.model.device,
            dtype=self.model.dtype,  # Ensure consistent dtype
        )

        # Iterate in batches to handle large inputs
        for i in range(0, num_inputs, batch_size):
            if input is not None:
                batch_inputs = {
                    k: v[i : min(i + batch_size, num_inputs)] for k, v in inputs.items()
                }
                # Ensure output_hidden_states=True to get all layer outputs
                output = self.model(**batch_inputs, output_hidden_states=True)

            elif input_ids is not None:
                batch_input_ids = inputs[i : min(i + batch_size, num_inputs), :]
                # Ensure output_hidden_states=True
                output = self.model(
                    input_ids=batch_input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )

            elif inputs_embeds is not None:
                batch_inputs_embeds = inputs[i : min(i + batch_size, num_inputs), :, :]
                # Ensure output_hidden_states=True
                output = self.model(
                    inputs_embeds=batch_inputs_embeds,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )

            hidden_states = output.hidden_states

            for j, layer in enumerate(self.layers):
                # Ensure that `emb` is assigned a slice that retains its grad_fn.
                # Direct slicing is usually fine, but confirm `hidden_states[layer]` still has grad_fn.
                # If `hidden_states[layer]` is a tuple/list of tensors and you want the last, ensure it's selected.
                emb[i : min(i + batch_size, num_inputs), j, :] = (
                    hidden_states[layer][:, :]
                    if self.positions is None
                    else hidden_states[layer][:, positions, :]
                )

        if return_numpy:
            # Only detach if you truly need a numpy array.
            # For GCG, you need gradients, so this path should ideally not be taken.
            emb = emb.cpu().detach().numpy()

        if self.positions is None and lengths is not None and return_numpy and unpad:
            emb = [emb[i, :, -lengths[i] :, :] for i in range(emb.shape[0])]

        return emb

    def distance(
        self, input_vec, target=None, p: Optional[Union[int, str]] = 2
    ) -> float:
        """
        Computes the distance between input and target embeddings.

        Args:
            input_vec (torch.Tensor): The input embeddings.
            target (torch.Tensor): The target embeddings.
            p (Union[int, str], optional): The type of distance lens to use. Defaults to 2.

        Returns:
            float: The computed distance between input and target embeddings.

        Raises:
            ValueError: If the dimensions of input and target embeddings do not match.
            ValueError: If an invalid type of distance lens is provided.

        """
        # Remove dimensions of size 1
        input_vec = input_vec.squeeze()
        target = target if target is not None else self.target
        target = target.squeeze()
        dim_axis = (
            1 if len(input_vec.shape) == 2 else 0
        )  # We assume that the batch size, if not 1, corresponds to the first axis

        # Check that dimensions match
        if input_vec.shape != target.shape:
            raise ValueError(
                f"input and target dimensions do not match. Input has shape {input_vec.shape} and target has shape {target.shape}"
            )

        # Compute the distance between the input and target
        if p == 1:
            result = torch.norm(input_vec - target, p=1, dim=dim_axis)
        elif p == 2:
            result = torch.norm(input_vec - target, p=2, dim=dim_axis)
        elif p == "inf":
            result = torch.norm(input_vec - target, p=float("inf"), dim=dim_axis)
        elif p == "cosine":
            result = torch.dot(input_vec, target) / (
                torch.norm(input_vec) * torch.norm(target) + 1e-8
            )
        else:
            raise ValueError(f"Invalid type of distance. (provided p={p})")

        if not self.requires_grad_:
            result = result.item()

        return result


class KVLens(Lens):
    @type_check
    def __init__(
        self,
        model,
        tokenizer,
        type: str,  # "key", "value"
        layers: Optional[Union[int, List[int]]] = None,
        heads: Optional[Union[int, List[int]]] = None,
        positions: Optional[Union[int, List[int]]] = None,
        requires_grad_: bool = False,
    ):
        super().__init__(model, tokenizer)
        assert type in [
            "key",
            "value",
        ], "Invalid type specified. Must be either 'key' or 'value'"
        self.layers = [layers] if isinstance(layers, int) else layers
        self.positions = [positions] if isinstance(positions, int) else positions
        self.heads = [heads] if isinstance(heads, int) else heads

        self.embed_size = (
            self.model.config.hidden_size // self.model.config.num_attention_heads
        )

        if self.layers is None:
            print("Warning: No layers specified. Default to all layers.")
            self.layers = [
                l for l in range(model.config.num_hidden_layers)
            ]  # num_hidden_layers also considers first embedding conversion

        if self.heads is None:
            print("Warning: No heads specified. Default to all heads.")
            self.heads = [h for h in range(model.config.num_attention_heads)]

        if self.positions is None:
            print("Warning: No positions specified. Default to all positions.")

        self.target = None
        self.requires_grad_ = requires_grad_
        self.type = 0 if type == "key" else 1

        assert (
            min(self.layers) >= -model.config.num_hidden_layers
            and max(self.layers) <= model.config.num_hidden_layers
        ), "Invalid layer(s) specified"

    @find_executable_batch_size
    def __call__(
        batch_size,
        self,
        input: Union[str, List[str], torch.Tensor],
        return_numpy: bool = False,
        unpad: bool = False,
    ) -> torch.Tensor:
        """
        Computes the context for the given input.

        Args:
            batch_size (int): The batch size for processing the input.
            input (Union[str, List[str], torch.Tensor]): The input text or batch of texts.

        Returns:
            torch.Tensor: The computed contextual embeddings.

        """
        assert (
            isinstance(input, str)
            or isinstance(input, list)
            or isinstance(input, torch.Tensor)
        ), "Input must be a string, a list of strings, or a tensor"

        if isinstance(input, torch.Tensor):
            inputs = input
            lengths = None
        else:
            input = [input] if isinstance(input, str) else input
            inputs = self.tokenizer(input, return_tensors="pt", padding=True).to(
                self.model.device
            )
            # Use attention mask to get the prompt length
            lengths = torch.sum(inputs.attention_mask, dim=1).cpu().numpy()

        positions = (
            [i for i in range(inputs.input_ids.shape[1])]
            if self.positions is None
            else self.positions
        )

        emb = torch.zeros(
            (
                inputs.input_ids.shape[0],
                len(self.layers),
                len(self.heads),
                len(positions),
                self.embed_size,
            ),
            device=self.model.device,
        )

        for i in range(0, inputs.input_ids.shape[0], batch_size):
            batch_inputs = {
                k: v[i : min(i + batch_size, inputs.input_ids.shape[0])]
                for k, v in inputs.items()
            }
            output = self.model(**batch_inputs)
            past_key_values = output.past_key_values

            for j, layer in enumerate(self.layers):
                emb[i : min(i + batch_size, inputs.input_ids.shape[0]), j] = (
                    past_key_values[layer][self.type][:, self.heads, positions, :]
                    if self.positions is not None
                    else past_key_values[layer][self.type][:, self.heads, :, :]
                )

        if self.requires_grad_:
            emb.requires_grad_()

        if return_numpy:
            emb = emb.cpu().detach().numpy()

        if self.positions is None and lengths is not None and return_numpy and unpad:
            emb = [emb[i, :, -lengths[i] :, :] for i in range(emb.shape[0])]

        return emb

    def distance(self) -> float:
        raise NotImplementedError("Distance not implemented for KVLens")
