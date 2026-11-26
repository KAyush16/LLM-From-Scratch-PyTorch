"""Instruction Supervised Fine-Tuning (SFT) prompt formatting and batching."""

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


def format_input(entry: Dict[str, str]) -> str:
    """Format an instruction entry into Stanford Alpaca prompt structure."""
    instruction_text = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry.get("input") else ""
    return instruction_text + input_text


class InstructionDataset(Dataset):
    """Pre-tokenized Alpaca instruction-response dataset."""

    def __init__(self, data: List[Dict[str, str]], tokenizer: Any) -> None:
        self.data = data
        self.encoded_texts: List[List[int]] = []
        for entry in data:
            prompt = format_input(entry)
            response = f"\n\n### Response:\n{entry['output']}"
            self.encoded_texts.append(tokenizer.encode(prompt + response))

    def __getitem__(self, index: int) -> List[int]:
        return self.encoded_texts[index]

    def __len__(self) -> int:
        return len(self.data)


def custom_collate_fn(
    batch: List[List[int]],
    pad_token_id: int = 50256,
    ignore_index: int = -100,
    allowed_max_length: Optional[int] = None,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length instruction sequences and mask trailing pads with `-100`.

    The first `<|endoftext|>` token (`50256`) is retained in `targets` so the model
    learns when to stop generating; subsequent padding tokens are replaced with
    `ignore_index=-100` so PyTorch `cross_entropy` ignores them.
    """
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy() + [pad_token_id]
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1], dtype=torch.long)
        targets = torch.tensor(padded[1:], dtype=torch.long)

        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    return torch.stack(inputs_lst).to(device), torch.stack(targets_lst).to(device)
