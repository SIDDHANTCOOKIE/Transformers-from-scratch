# Transformer From Scratch

A clean, modular, from-scratch PyTorch implementation of the original Transformer architecture from the landmark paper:

> **"Attention Is All You Need"**
> Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin
> *NeurIPS 2017*
>
> 📄 [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)


---

##  Features

- **Full encoder-decoder Transformer** — not a wrapper around `nn.Transformer`, every component is written from scratch.
- **Multi-Head Scaled Dot-Product Attention** with proper $\frac{QK^T}{\sqrt{d_k}}$ scaling.
- **Sinusoidal Positional Encoding** as described in Section 3.5 of the paper.
- **Causal + padding masking** for autoregressive decoding.
- **Paper-faithful warmup LR schedule** (Section 5.3): $lr = d_{model}^{-0.5} \cdot \min(step^{-0.5},\; step \cdot warmup\_steps^{-1.5})$
- **Synthetic copy-task** to verify the model can learn sequence-to-sequence mappings.

---

##  Architecture Overview

```
Input Tokens ──► Embedding ──► + Positional Encoding ──► Encoder Stack (×N) ──►
                                                                                 ├──► Decoder Stack (×N) ──► Linear ──► Output Probabilities
Target Tokens ──► Embedding ──► + Positional Encoding ──────────────────────────►
```

Each **Encoder Layer** contains:
1. Multi-Head Self-Attention + Residual Connection + LayerNorm
2. Position-wise Feed-Forward Network + Residual Connection + LayerNorm

Each **Decoder Layer** contains:
1. Masked Multi-Head Self-Attention + Residual Connection + LayerNorm
2. Multi-Head Cross-Attention (over encoder output) + Residual Connection + LayerNorm
3. Position-wise Feed-Forward Network + Residual Connection + LayerNorm

---

##  Project Structure

```
transformer-from-scratch/
├── model.py          # Transformer architecture (attention, encoder, decoder, PE)
├── dataset.py        # Synthetic copy-task dataset and mask utilities
├── train.py          # Training loop with warmup LR schedule
├── config.py         # Hyperparameters and model configuration
├── requirements.txt  # Python dependencies
└── README.md
```

---

##  Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 2.0+

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train the Model

```bash
python train.py
```

You should see the loss decrease steadily and the copy-task accuracy approach ~100%:

```
Using device: cuda
Epoch  1/20 | Loss: 3.8942 | LR: 0.000069
Epoch  2/20 | Loss: 3.1205 | LR: 0.000044
...
Epoch 20/20 | Loss: 0.0123 | LR: 0.000015

Copy-task accuracy on one batch: 99.43%
```

---

##  Configuration

All hyperparameters are centralized in [`config.py`](config.py):

| Parameter | Default | Description |
|---|---|---|
| `d_model` | 512 | Embedding / hidden dimension |
| `num_heads` | 8 | Number of attention heads |
| `num_layers` | 6 | Number of encoder & decoder layers |
| `d_ff` | 2048 | Feed-forward inner dimension |
| `dropout` | 0.1 | Dropout rate |
| `max_seq_len` | 100 | Maximum sequence length |
| `batch_size` | 64 | Training batch size |
| `epochs` | 20 | Number of training epochs |

---

##  How the Copy Task Works

The model is trained on a **sequence-copying task**: given an input sequence of random tokens, the model must reproduce the exact same sequence as output. This is a standard sanity check that verifies:

1. The attention mechanism can learn to "look at" the right source positions.
2. The encoder-decoder information flow works correctly.
3. The causal masking allows proper autoregressive generation.

If the model reaches near-perfect accuracy on this task, the architecture is implemented correctly.

---

##  References

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). *Attention Is All You Need*. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Jay Alammar
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) — Harvard NLP

---

##  License

This implementation was done to learn more about transformers while reading the great "Attention is all you need" paper.
