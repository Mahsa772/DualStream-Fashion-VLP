# DualStream Fashion VLP

A vision-language pre-training framework for fine-grained fashion understanding, centered on a **Decoupled Dual-Stream Text Encoder** that learns to separate category-level semantics from attribute-level semantics — without any hard-coded symbol tokens or label supervision.

---

## Core Idea: Decoupled Text Attention

Standard fashion VLP models encode an entire text description into a single vector, mixing category words (*tops*, *dresses*) with fine-grained attribute words (*long sleeve*, *floral print*). This makes it hard for the model to distinguish what kind of item it is from how it looks.

**DualStream Fashion VLP solves this by splitting the text encoder into two independently learned streams:**

| Stream | Learns to focus on | Used for |
|--------|--------------------|----------|
| **S — Symbol stream** | Category / class words | MoCo-style self-supervised contrastive loss |
| **D — Description stream** | Attribute / detail words | Image-text alignment (ITA) |

Both streams share the same BERT token outputs but use separate Q, K, V projections — the model discovers the decomposition purely from data.

```
BERT encoder
     │
     ▼
text_embeds  (B, L, 768)
     │
     └──► DecoupledTextAttention
               ├── S head  (B, 768)  →  MoCo loss   (self-supervised)
               └── D head  (B, 768)  →  ITA loss    (image-text alignment)
```

### Why it works for Category & Subcategory Recognition

By forcing S to specialise in category words and D to specialise in attributes, the final classifier receives **both signals concatenated** as a 1536-dim vector `[S ; D]`. This gives the classification heads a richer, non-redundant input compared to a single `[CLS]` token:

- **CR (48 categories): Accuracy 100% — Macro F1 1.000**
- **SCR (121 subcategories): Accuracy 97.3% — Macro F1 0.890**

---

## Pre-Training Losses

| Loss | What it does |
|------|-------------|
| `loss_moco` | Contrastive loss — S stream vs. momentum image queue (no label needed) |
| `loss_ita` | Image-text alignment — D stream vs. image features |
| `loss_itm` | Image-text matching — cross-modal fusion |
| `loss_mlm` | Masked language modeling |

**D-stream warmup:** For the first N epochs the ITA loss falls back to the raw `[CLS]` token, preventing degraded alignment while D is still untrained. After warmup, ITA switches to D permanently.

---

## Requirements

```bash
pip install -r requirements.txt
```

Core: PyTorch ≥ 1.9, `transformers`, `ruamel.yaml`, `h5py`, `timm`.

---

## Dataset Setup

### FashionGen

1. Download the HDF5 files from [FashionGen](https://arxiv.org/abs/1806.08317) and place them under `data/data-fashion/`.
2. Edit `data_root` and `split` in `prepare_dataset.py`, then run:
   ```bash
   python prepare_dataset.py
   ```

---

## Pre-Trained Model

Our dual-stream pre-trained checkpoint is available for download at:

> **[Download checkpoint](https://drive.google.com/file/d/1cJsk0Ql66KMhcciyMull0othdxRljjEz/view?usp=drive_link)**

Place the checkpoint at `output/full_run/epoch029/checkpoint.pth` before running fine-tuning scripts.

---

## Training

### 1. Pre-Training

```bash
# Quick sanity check (5% data, 1 GPU)
bash run_pretrain_sym.sh smoke

# Medium run (20% data, 1 GPU)
bash run_pretrain_sym.sh tier2

# Full training (100% data, 2 GPUs)
bash run_pretrain_sym.sh full

# Resume from checkpoint
bash run_pretrain_sym.sh full output/full_run/epoch029/checkpoint.pth
```

### 2. Retrieval Fine-Tuning

```bash
bash run_retrieval_sym.sh
```

Text feature at inference: **SD = (S + D) / 2** — combines both streams for richer image–text matching.

### 3. Category Recognition Fine-Tuning

```bash
bash run_caterg_sym.sh
```

Classification input: **[S ; D]** concatenated (1536-dim) → separate heads for CR and SCR.

---

## Results

Pre-training: 30 epochs on the full FashionGen dataset.

### **Category & Subcategory Recognition**

| Task | Accuracy | Macro F1 |
|------|----------|----------|
| **Category Recognition** (48 classes) | **100.0%** | **1.000** |
| **Subcategory Recognition** (121 classes) | **97.3%** | **0.890** |

### Image–Text Retrieval (FashionGen)

| Split | I2T R@1 | T2I R@1 | Mean R@1 |
|-------|---------|---------|----------|
| Global test set | 53.85 | 60.41 | 57.13 |
| Tiny (avg × 5 random sets) | — | — | 70.88 ± 1.03 |

> **Note on SCR Macro-F1:** One subcategory (`CORSETS & BODYSUITS`) appears in the test set but not in the training split, so every item in that class is predicted wrong by construction. This fully explains the gap between SCR accuracy (97.3%) and Macro-F1 (0.890) — macro averaging weights every class equally, so one completely unseen class pulls the average down noticeably.

---

## Contact

**Mahsa Ahmadi** — mahsaahmadi772@gmail.com

---

## Acknowledgements

Inspired by [FashionSAP](https://github.com/hssip/FashionSAP) (Han et al., CVPR 2023). Utility code partially adapted from [ALBEF](https://github.com/salesforce/ALBEF).

```bibtex
@inproceedings{FashionSAP,
  title     = {FashionSAP: Symbols and Attributes Prompt for Fine-grained Fashion Vision-Language Pre-training},
  author    = {Han, Yunpeng and Zhang, Lisai and Chen, Qingcai and Chen, Zhijian and Li, Zhonghua and Yang, Jianxin and Cao, Zhao},
  booktitle = {CVPR},
  year      = {2023}
}
```
