# Building a GPT-2 LLM From Scratch in PyTorch

End-to-end implementation of a GPT-2–scale decoder-only Transformer built from scratch in PyTorch on Apple Silicon (`mps`)—progressing from regex & BPE tokenization and causal multi-head attention to full `GPTModel` pretraining, OpenAI checkpoint weight mapping (`124M` & `355M`), binary classification fine-tuning, and Alpaca-style instruction supervised fine-tuning (SFT).

---

## Architecture & Progression

| Stage | Module / Notebook | Key Components Implemented |
| :--- | :--- | :--- |
| **1. Tokenization & Embeddings** | [`src/tokenizer_dataset.py`](src/tokenizer_dataset.py)<br>[`GPT_Tokenizer/tokenization_embeddings_preprocessing.ipynb`](GPT_Tokenizer/tokenization_embeddings_preprocessing.ipynb) | `SimpleTokenize` (regex + `<UNK>`), GPT-2 BPE (`tiktoken`, `50,257` vocab), sliding-window `GPTDatasetV1`, `create_dataloader_v1()`, token + positional embeddings (`768`-dim, `1024` context) |
| **2. Attention Mechanisms** | [`src/attention.py`](src/attention.py)<br>[`Attention_Mechanism/attention_basics.ipynb`](Attention_Mechanism/attention_basics.ipynb) | `SelfAttention_v1` -> `CausalAttention` (upper-triangular `-inf` mask + dropout) -> `MultiHeadAttention` (parallel `12` heads &times; `64` head dim, `1/sqrt(d_k)` scaling) |
| **3. GPT-2 Model & Decoding** | [`src/gpt_model.py`](src/gpt_model.py)<br>[`llm_architecture.ipynb`](llm_architecture.ipynb) | `LayerNormalization`, `GELU`, `FeedForward` (`768 -> 3072 -> 768`), pre-norm residual `TransformerBlock` (`12` layers), `GPTModel` (`162.4M` params before weight tying), `softmax_with_temperature()`, `generate()` (top-k + temperature sampling) |
| **4. Pretraining Loop** | [`src/pretraining.py`](src/pretraining.py)<br>[`llm_architecture.ipynb`](llm_architecture.ipynb) | `calc_loss_batch()`, `calc_loss_loader()`, `evaluate_model()`, `train_model_simple()` trained on *The Verdict* (`4,608` train / `512` val tokens) |
| **5. OpenAI Weight Mapping** | [`src/weight_loader.py`](src/weight_loader.py), [`gpt_download3.py`](gpt_download3.py)<br>[`pre_trained_gpt.ipynb`](pre_trained_gpt.ipynb) | TensorFlow-to-PyTorch checkpoint loader (`124M` & `355M`), `assign()` shape verification, `load_weights_into_gpt()` splitting fused `c_attn` QKV projections and transposing weights (`.T`) |
| **6. Classification Fine-Tuning** | [`src/finetune_classifier.py`](src/finetune_classifier.py)<br>[`pre_trained_gpt.ipynb`](pre_trained_gpt.ipynb) | `create_balanced_dataset()`, `random_split()`, `SpamDataset` (`130` train / `19` val / `38` test batches), `2`-class head on final causal token `logits[:, -1, :]`, `train_classifier_simple()`, `classify_review()` |
| **7. Instruction SFT Pipeline** | [`src/finetune_instruction.py`](src/finetune_instruction.py)<br>[`pre_trained_gpt.ipynb`](pre_trained_gpt.ipynb) | `format_input()` (Stanford Alpaca template), `InstructionDataset` (`935` train / `55` val / `110` test on `instruction-data.json`), `custom_collate_fn()` keeping one EOS token (`50256`) and masking trailing pads with `-100` |

---

## Verified Results & Observations

- **SMS Spam Classification Fine-Tuning (`124M` GPT-2 on Apple Silicon MPS, 5 Epochs):**
  - Untrained classification head baseline: **46.25%** train / **46.25%** val / **50.00%** test accuracy (`2.779` train loss).
  - After 5 epochs of fine-tuning: **87.69%** train / **89.26%** val / **86.67%** test accuracy (see [`accuracy-plot.pdf`](accuracy-plot.pdf)).
- **Temperature Scaling Decoding Analysis:**
  - Verified probability sharpening (`T = 0.1`) vs. flat exploration (`T = 5.0`) vs. baseline (`T = 1.0`) in [`temperature-plot.pdf`](temperature-plot.pdf).
- **Notebook Fixes Applied in Modular `src/`:**
  - Fixed `MultiHeadAttention.forward` to reassign `attn_scores = attn_scores.masked_fill(...)` (instead of discarding the non-in-place return value).
  - Standardized `TransformerBlock` / `GPTModel` attribute names (`trf_blocks`, `att`) so `load_weights_into_gpt()` maps cleanly without attribute mismatches.
  - Added explicit device/dtype handling in `GELU` and `generate()` top-k masking.

---

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── gpt_download3.py
├── llm_architecture.ipynb
├── pre_trained_gpt.ipynb
├── GPT_Tokenizer/
│   ├── tokenization_embeddings_preprocessing.ipynb
│   └── the-verdict.txt
├── Attention_Mechanism/
│   └── attention_basics.ipynb
├── sms_spam_collection/
│   ├── SMSSpamCollection.tsv
│   └── readme
├── instruction-data.json
├── train.csv / validation.csv / test.csv
├── accuracy-plot.pdf / temperature-plot.pdf
└── src/
    ├── tokenizer_dataset.py
    ├── attention.py
    ├── gpt_model.py
    ├── pretraining.py
    ├── weight_loader.py
    ├── finetune_classifier.py
    └── finetune_instruction.py
```

*(Note: Large binary model weights `*.pth` and downloaded OpenAI `gpt2/` TensorFlow checkpoints are excluded via `.gitignore` to stay within GitHub file limits. Run `gpt_download3.py` to download official GPT-2 checkpoints locally.)*

---

## Quick Start

```bash
pip install -r requirements.txt
python3 -c "from src.gpt_model import GPTModel, GPT_CONFIG_124M; m = GPTModel(GPT_CONFIG_124M); print(sum(p.numel() for p in m.parameters()))"
```
