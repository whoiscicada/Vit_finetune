# Chest X-Ray Pneumonia Detection with Vision Transformer (ViT)

Fine-tunes `google/vit-base-patch16-224-in21k` on chest X-ray images to classify **NORMAL** vs **PNEUMONIA** using LoRA (PEFT).

## Results

| Metric | Score |
|--------|-------|
| Accuracy | 87.98% |
| Precision | 84.16% |
| Recall | 99.49% |
| F1-Score | 91.19% |

Confusion matrix (test set, 624 images):

|  | Predicted NORMAL | Predicted PNEUMONIA |
|--|--|--|
| Actual NORMAL | 161 | 73 |
| Actual PNEUMONIA | 2 | 388 |

High recall (99.49%) — only 2 pneumonia cases missed — makes this suitable for screening.

## Dataset

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) from Kaggle.

```
chest_xray/
├── train/   (NORMAL/, PNEUMONIA/)
├── val/
└── test/
```

## Approach

- **Model:** ViT-Base (patch 16, 224×224), pretrained on ImageNet-21k
- **Fine-tuning:** LoRA via PEFT — only 589,824 / 86,390,018 params trainable (0.68%)
- **LoRA config:** r=16, alpha=32, targets `query` and `value` layers, dropout=0.1
- **Training:** 10 epochs, AdamW (lr=1e-4, weight_decay=0.01), CrossEntropyLoss

## Setup

```bash
pip install transformers==4.41.2 peft==0.11.1 accelerate==0.31.0
pip install torch torchvision scikit-learn matplotlib seaborn pillow
```

Download dataset:

```bash
kaggle datasets download paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip
```

## Usage

Open and run [vit_pneumonia2.ipynb](vit_pneumonia2.ipynb) top to bottom. Update `train_dir`, `val_dir`, `test_dir` paths to match local dataset location.

Saved model weights: `vit_chest_xray_model.pth`

```python
model.load_state_dict(torch.load("vit_chest_xray_model.pth"))
```

## Dependencies

| Package | Version |
|---------|---------|
| transformers | 4.41.2 |
| peft | 0.11.1 |
| accelerate | 0.31.0 |
| torch | — |
| torchvision | — |
| scikit-learn | — |
