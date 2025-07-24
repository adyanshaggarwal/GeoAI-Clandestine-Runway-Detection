import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
torch.backends.cudnn.benchmark = True
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# ---------- Config ----------
IMG_DIR = 'images'
MASK_DIR = 'masks'
IMAGE_SIZE = 512
BATCH_SIZE = 8
EPOCHS = 30
LR = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
VISUALIZE_SAMPLES = True
# ----------------------------

# ---------- Dataset ----------
class AirstripDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

        image_files = glob(os.path.join(img_dir, '*.png'))
        self.pairs = []
        for img_path in image_files:
            img_name = os.path.basename(img_path)
            id_part = img_name.split('_')[1]
            id_core = os.path.splitext(id_part)[0]
            mask_name = f"1_id_{id_core}.png"
            mask_path = os.path.join(mask_dir, mask_name)
            if os.path.exists(mask_path):
                self.pairs.append((img_path, mask_path))

        print(f"[INFO] Matched {len(self.pairs)} image-mask pairs.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype('float32')
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].unsqueeze(0)
        return image, mask

# ---------- Transforms ----------
def get_transforms():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.OneOf([
            A.RandomBrightnessContrast(p=0.5),
            A.GaussianBlur(p=0.5),
            A.RandomGamma(p=0.5),
            A.CLAHE(p=0.5)
        ], p=0.7),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

# ---------- Model ----------
def get_model():
    model = smp.Unet(encoder_name="efficientnet-b0", encoder_weights="imagenet", in_channels=3, classes=1)
    return model.to(DEVICE)

# ---------- Metrics ----------
def dice_score(preds, targets, threshold=0.2):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum()
    return (2. * intersection) / (preds.sum() + targets.sum() + 1e-7)

# ---------- Loss Function ----------
loss_fn = smp.losses.TverskyLoss(mode='binary', log_loss=False, from_logits=True)

# ---------- Training ----------
def train_fn(loader, model, optimizer):
    model.train()
    total_loss = 0
    for x, y in tqdm(loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        preds = model(x)
        loss = loss_fn(preds, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# ---------- Validation ----------
def evaluate_fn(loader, model):
    model.eval()
    total_loss = 0
    dice_total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x)
            loss = loss_fn(preds, y)
            total_loss += loss.item()
            dice_total += dice_score(preds, y).item()
    return total_loss / len(loader), dice_total / len(loader)

# ---------- Visual Debug ----------
def visualize_sample(dataset):
    print("🔍 Visualizing sample image + mask...")
    img, mask = dataset[0]
    img = img.permute(1, 2, 0).numpy()
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)
    mask = mask.squeeze().numpy()
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Image")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.title("Mask")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_predictions(model, dataset):
    model.eval()
    x, y = dataset[0]
    x_input = x.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = model(x_input)
        pred_mask = torch.sigmoid(pred).squeeze().cpu().numpy()
    x_np = x.permute(1, 2, 0).numpy()
    x_np = x_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    x_np = np.clip(x_np, 0, 1)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(x_np)
    plt.title("Image")
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(y.squeeze().numpy(), cmap='gray')
    plt.title("Ground Truth")
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask > 0.2, cmap='gray')
    plt.title("Predicted Mask")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# ---------- Main ----------
def main():
    dataset = AirstripDataset(IMG_DIR, MASK_DIR, transform=get_transforms())
    if VISUALIZE_SAMPLES:
        visualize_sample(dataset)

    val_split = int(0.8 * len(dataset))
    train_set, val_set = torch.utils.data.random_split(dataset, [val_split, len(dataset) - val_split])
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    model = get_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_losses = []
    val_losses = []
    val_dice_scores = []

    print(f"\n🚀 Starting training on {DEVICE}...\n")
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        train_loss = train_fn(train_loader, model, optimizer)
        val_loss, val_dice = evaluate_fn(val_loader, model)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_dice_scores.append(val_dice)

        print(f"📉 Train Loss: {train_loss:.4f} | 📈 Val Loss: {val_loss:.4f} | 🎯 Dice: {val_dice:.4f}")

    torch.save(model.state_dict(), 'airstrip_seg_model.pth')
    print("\n✅ Model saved as 'airstrip_seg_model.pth'")

    # Plot curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(val_dice_scores, label='Val Dice Score', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.title('Validation Dice Score')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

    visualize_predictions(model, dataset)

# ---------- Entry ----------
if __name__ == "__main__":
    main()
