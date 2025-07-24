Airstrip Segmentation using U-Net and EfficientNet Encoder
## 🧠 Model Architecture

- **Base Model**: [U-Net](https://arxiv.org/abs/1505.04597)
- **Encoder**: `EfficientNet-B0` pretrained on ImageNet (via `segmentation_models_pytorch`)
- **Input Channels**: 3 (RGB)
- **Output Channels**: 1 (Binary Mask)

---

## ⚙️ Training Configuration

| Parameter       | Value          |
|----------------|----------------|
| Image Size     | 512 × 512      |
| Batch Size     | 8              |
| Epochs         | 30             |
| Learning Rate  | 1e-4           |
| Optimizer      | Adam           |
| Device         | GPU / CPU      |
| Frameworks     | PyTorch, Albumentations |

---

## 📉 Loss Function & Metrics

- **Loss Function**: Tversky Loss (from `smp.losses`)
- **Evaluation Metric**: Dice Score  
  Dice Score is calculated using a sigmoid activation followed by a threshold of 0.2.

---

## 🎨 Data Augmentation

Data augmentation is applied using `Albumentations`:
- Random Brightness/Contrast
- Gaussian Blur
- Random Gamma
- CLAHE
- Horizontal Flip
- Normalization (ImageNet stats)

---

## 📊 Output

- Training and Validation Loss curves
- Dice Score curve
- Predicted masks vs Ground Truth for visual inspection
- Trained model saved as: `airstrip_seg_model.pth`
- Training curves saved as: `training_curves.png`
