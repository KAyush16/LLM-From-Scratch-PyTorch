"""Tokenization, sliding-window dataset creation, and DataLoader utilities.

Covers the progression from a simple regex-based tokenizer (`SimpleTokenize`)
to GPT-2 Byte Pair Encoding (`tiktoken`) with shifted `(input, target)` chunks
for autoregressive next-token prediction.
"""

import re
from typing import Dict, List

import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset


class SimpleTokenize:
    """Minimal word/punctuation regex tokenizer with `<UNK>` fallback."""

    def __init__(self, vocab: Dict[str, int]) -> None:
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> List[int]:
        """Split text on punctuation/whitespace and map tokens to vocabulary IDs."""
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int else "<UNK>"
            for item in preprocessed
        ]
        return [self.str_to_int[s] for s in preprocessed]

    def decode(self, ids: List[int]) -> str:
        """Convert token IDs back into readable text without extra space before punctuation."""
        text = " ".join(self.int_to_str[i] for i in ids)
        return re.sub(r'\s+([,.:;?!"()\'])', r"\1", text)


class GPTDatasetV1(Dataset):
    """Sliding-window next-token prediction dataset using GPT-2 BPE."""

    def __init__(self, text: str, max_len: int, stride: int) -> None:
        self.input_ids: List[torch.Tensor] = []
        self.target_ids: List[torch.Tensor] = []

        tokenizer = tiktoken.get_encoding("gpt2")
        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

        # Each target sequence is shifted right by 1 token so position t predicts t+1.
        for i in range(0, len(token_ids) - max_len, stride):
            input_chunk = token_ids[i : i + max_len]
            target_chunk = token_ids[i + 1 : i + max_len + 1]
            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt: str,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create a PyTorch DataLoader over sliding-window BPE token chunks."""
    dataset = GPTDatasetV1(txt, max_len=max_length, stride=stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
