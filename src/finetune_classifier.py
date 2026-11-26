"""Classification fine-tuning pipeline for GPT-2 on the SMS Spam dataset.

Because causal self-attention ensures the final sequence token can attend to
every preceding token, classification logits are read directly from `logits[:, -1, :]`.
"""

from typing import Any, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def create_balanced_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Downsample `ham` messages to match `spam` count for balanced fine-tuning."""
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    return pd.concat([ham_subset, df[df["Label"] == "spam"]], ignore_index=True)


def random_split(
    df: pd.DataFrame, train_frac: float, validation_frac: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Shuffle and split dataframe into train, validation, and test partitions."""
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    train_end = int(len(df) * train_frac)
    val_end = train_end + int(len(df) * validation_frac)
    return df[:train_end], df[train_end:val_end], df[val_end:]


class SpamDataset(Dataset):
    """Tokenized, fixed-length padded dataset for binary spam classification."""

    def __init__(
        self,
        csv_file: str,
        tokenizer: Any,
        max_length: Optional[int] = None,
        pad_token_id: int = 50256,
    ) -> None:
        self.data = pd.read_csv(csv_file)
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]

        if max_length is None:
            self.max_length = max(len(seq) for seq in self.encoded_texts)
        else:
            self.max_length = max_length
            self.encoded_texts = [seq[: self.max_length] for seq in self.encoded_texts]

        self.encoded_texts = [
            seq + [pad_token_id] * (self.max_length - len(seq))
            for seq in self.encoded_texts
        ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.encoded_texts[index], dtype=torch.long),
            torch.tensor(self.data.iloc[index]["Label"], dtype=torch.long),
        )


def calc_loss_batch_cls(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """Compute cross-entropy loss using only the final causal token's logits."""
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]
    return torch.nn.functional.cross_entropy(logits, target_batch)


def calc_accuracy_loader(
    data_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    num_batches: Optional[int] = None,
) -> float:
    """Evaluate classification accuracy across a DataLoader."""
    model.eval()
    correct_predictions, num_examples = 0, 0
    limit = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= limit:
            break
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        with torch.no_grad():
            logits = model(input_batch)[:, -1, :]
        predicted_labels = torch.argmax(logits, dim=-1)
        num_examples += predicted_labels.shape[0]
        correct_predictions += (predicted_labels == target_batch).sum().item()

    return correct_predictions / num_examples


def train_classifier_simple(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
) -> Tuple[List[float], List[float], List[float], List[float], int]:
    """Fine-tune GPT-2 classification head and track epoch train/val accuracy."""
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch_cls(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            examples_seen += input_batch.shape[0]
            global_step += 1

            if global_step % eval_freq == 0:
                model.eval()
                with torch.no_grad():
                    tr_loss = sum(
                        calc_loss_batch_cls(xb, yb, model, device).item()
                        for i, (xb, yb) in enumerate(train_loader)
                        if i < eval_iter
                    ) / min(eval_iter, len(train_loader))
                    va_loss = sum(
                        calc_loss_batch_cls(xb, yb, model, device).item()
                        for i, (xb, yb) in enumerate(val_loader)
                        if i < eval_iter
                    ) / min(eval_iter, len(val_loader))
                model.train()
                train_losses.append(tr_loss)
                val_losses.append(va_loss)
                print(
                    f"Ep {epoch + 1} (Step {global_step:06d}): "
                    f"Train loss {tr_loss:.3f}, Val loss {va_loss:.3f}"
                )

        train_acc = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_acc = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        print(f"Training accuracy: {train_acc * 100:.2f}% | Validation accuracy: {val_acc * 100:.2f}%")

    return train_losses, val_losses, train_accs, val_accs, examples_seen


def classify_review(
    text: str,
    model: nn.Module,
    tokenizer: Any,
    device: torch.device,
    max_length: int,
    pad_token_id: int = 50256,
) -> str:
    """Run inference on a single SMS/review string and return `'spam'` or `'not spam'`."""
    model.eval()
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]
    limit = min(max_length, supported_context_length)
    input_ids = input_ids[:limit] + [pad_token_id] * max(0, limit - len(input_ids))

    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]
    return "spam" if torch.argmax(logits, dim=-1).item() == 1 else "not spam"
