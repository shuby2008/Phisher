# Multimodal Phishing Website Detection

A phishing website classifier that combines three views of a web page: its **URL**, its **HTML source**, and a **screenshot** of how it renders. Each view becomes a feature vector. The vectors are concatenated and passed to a classifier that labels the page as phishing or benign.

The best run so far gets about **98% accuracy** on a held-out test set of 2,997 pages.

---

## How it works

```
               ┌─────────────────────────────┐
  InputURL ───▶│ 7 hand-crafted URL features │──┐
               │ + URL TF-IDF (1,000 dims)   │  │
               └─────────────────────────────┘  │
               ┌─────────────────────────────┐  │    ┌──────────────┐    ┌────────────┐
  HTML ───────▶│ BERT pooler output (768)    │──┼───▶│ Concatenate  │───▶│ Classifier │──▶ phishing / benign
               └─────────────────────────────┘  │    │ + Standard-  │    │ (MLP or    │
               ┌─────────────────────────────┐  │    │   Scaler     │    │  XGBoost)  │
  Screenshot ─▶│ ResNet-50 embedding (2,048) │──┘    └──────────────┘    └────────────┘
               └─────────────────────────────┘
```

**Total input size: 3,823 features per page** (768 + 1,000 + 7 + 2,048).

### 1. Train/test split first
The data is split 70/30 (`random_state=42`) before any feature fitting, so the TF-IDF vectorizers and the scaler only ever see training data.

### 2. URL features
Seven lexical features from `InputURL`:

| Feature | Description |
|---|---|
| `url_length` | Number of characters |
| `num_digits` | Count of digits |
| `num_dots` | Count of `.` |
| `num_slashes` | Count of `/` |
| `num_hyphens` | Count of `-` |
| `has_ip` | 1 if the URL contains an IPv4-like pattern |
| `uses_https` | 1 if the URL starts with `https://` |

The URL is also vectorized with a TF-IDF model (top 1,000 terms, fit on the training set only).

### 3. HTML features
The raw HTML is tokenized with `bert-base-uncased` (truncated/padded to 512 tokens), and BERT's 768-dim **pooler output** is used as the page embedding. BERT is frozen and used only as a feature extractor.

### 4. Screenshot features
Each screenshot is resized to 224×224, normalized with ImageNet statistics, and passed through a **ResNet-50** pretrained on ImageNet with the final classification layer removed, giving a 2,048-dim embedding. Missing or unreadable screenshots are replaced with an all-zero image.

### 5. Classifier

**Neural network (`PhishingNet`)**, used in three of the four notebooks:

```
Linear(3823 → 512) → BatchNorm → ReLU → Dropout(0.4)
Linear(512 → 128)  → BatchNorm → ReLU → Dropout(0.3)
Linear(128 → 1)    → Sigmoid
```

- Loss: binary cross-entropy
- Optimizer: Adam, learning rate 1e-4
- Batch size: 32
- Epochs: 15
- Decision threshold: 0.5

**XGBoost** (`MM_phishingXGB`): an `XGBClassifier` with default hyperparameters (`eval_metric='logloss'`, `random_state=42`) trained on the same 3,823-dim feature matrix.

---

## Dataset

`drive/MyDrive/TR-OP/tr_op_summary_5000.csv`, with **9,990 pages** in total (6,993 train / 2,997 test) and a near-even class balance.

| Column | Description |
|---|---|
| `HTMLContent` | Raw HTML source of the page |
| `InputURL` | The URL that was visited |
| `Info` | Final URL after redirects (not currently used as a feature) |
| `ScreenshotPath` | Local path to a screenshot of the rendered page |
| `GoodorBad` | Label: **0 = phishing, 1 = benign** |

The dataset and screenshots are not included in this repository.

---

## Notebooks

| Notebook | Classifier | Status |
|---|---|---|
| `MM_phishing6.ipynb` | Neural network | Evaluated |
| `MM_phishingRFC.ipynb` | Neural network, with inference timing | Evaluated |
| `MM_phishingRFC_random_forest.ipynb` | Neural network, with inference timing | Evaluated (identical to `MM_phishingRFC`) |
| `MM_phishingXGB.ipynb` | XGBoost | Not yet run |

> **Note:** Despite their names, both "RFC" notebooks currently train the neural network, not a random forest. A random forest version has not been written yet.

---

## Results

Test set: 2,997 pages (1,508 phishing, 1,489 benign).

| Notebook | Accuracy | Phishing P / R / F1 | Benign P / R / F1 |
|---|---|---|---|
| `MM_phishing6` | **98.50%** | 0.98 / 0.99 / 0.99 | 0.99 / 0.98 / 0.98 |
| `MM_phishingRFC` | **98.13%** | 0.99 / 0.98 / 0.98 | 0.98 / 0.99 / 0.98 |
| `MM_phishingXGB` | n/a | n/a | n/a |

Training loss for the `MM_phishingRFC` run fell from 0.3264 in epoch 1 to 0.0151 in epoch 15.

**Inference time** (`MM_phishingRFC`, GPU): 0.0135 s for all 2,997 test pages, or about 4.5 µs per page. This measures only the classifier's forward pass on precomputed features. It does **not** include BERT or ResNet-50 feature extraction, which accounts for nearly all of the real per-page cost.

---

## Setup

Developed with Python 3.13.4 on a CUDA-enabled machine (the code falls back to CPU automatically, but BERT and ResNet-50 extraction will be slow).

```bash
pip install pandas numpy scikit-learn torch torchvision transformers pillow xgboost joblib jupyter
```

Then:

1. Place the dataset CSV at `drive/MyDrive/TR-OP/tr_op_summary_5000.csv` relative to the notebook.
2. Make sure the paths in the `ScreenshotPath` column point to real files on your machine (they are currently absolute Windows paths).
3. Run the notebook cells top to bottom.

The first run downloads `bert-base-uncased` from Hugging Face and the ResNet-50 ImageNet weights from torchvision.

---

## Known issues and next steps

- **Models are not saved.** The training cell prints "Model, vectorizer, and scaler saved." but nothing is written to disk. `joblib` is imported but unused. Add `torch.save(model.state_dict(), ...)` and `joblib.dump(...)` for the vectorizer and scaler.
- **HTML TF-IDF is computed and then discarded.** Step 3 builds an 8,000-feature HTML TF-IDF matrix, but the BERT cell overwrites `X_html_train` / `X_html_test`. Either remove the HTML TF-IDF step or keep both under different names and compare.
- **Truncated HTML.** BERT only sees the first 512 tokens, which for most pages is mostly `<head>` markup. Extracting visible text first, or chunking and pooling, may help.
- **Missing screenshots fail silently.** A bare `except` replaces any unreadable image with zeros. It's worth counting how many screenshots are missing, since an all-zero embedding could itself become a label signal.
- **Single split, no validation set.** Results come from one 70/30 split with no validation set or cross-validation, and PyTorch seeds aren't fixed, so numbers will vary slightly between runs.
- **Out-of-order execution.** In the RFC notebooks, the later cells show lower execution counts than the earlier ones, which means they were run in a separate kernel session. Re-running top to bottom is recommended before reporting results.
- **Possible URL leakage.** Benign URLs in the sample tend to be bare domains (`https://takealot.com`) while phishing URLs carry paths and subdomains, so the URL features may be learning dataset-collection artifacts. Testing on phishing and benign data from a different source would show how well the model generalizes.
- **Deprecation warnings.** Replace `models.resnet50(pretrained=True)` with `weights=ResNet50_Weights.DEFAULT`, and drop `use_label_encoder=False` from `XGBClassifier` (removed in recent XGBoost versions).
- **Finish the comparisons.** Run the XGBoost notebook, write a real random forest notebook, and run an ablation study (URL only / HTML only / screenshot only) to see what each modality contributes.
