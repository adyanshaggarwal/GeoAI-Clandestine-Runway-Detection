import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# -------- CONFIG --------
IMAGE_SIZE = 512
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
THRESHOLD = 0.2

TEST_IMG_DIR = 'test_images'
TEST_MASK_DIR = 'test_masks'
MODEL_PATH = 'airstrip_seg_model.pth'
SAVE_PRED_DIR = 'predictions'
os.makedirs(SAVE_PRED_DIR, exist_ok=True)

# -------- Transforms --------
transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# -------- Load Model --------
model = smp.Unet(
    encoder_name='efficientnet-b0',
    encoder_weights='imagenet',
    in_channels=3,
    classes=1
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# -------- Metrics --------
def dice_score(pred, target):
    pred = (pred > THRESHOLD).astype(np.float32)
    target = (target > 0.5).astype(np.float32)
    intersection = (pred * target).sum()
    return (2. * intersection) / (pred.sum() + target.sum() + 1e-7)

def iou_score(pred, target):
    pred = (pred > THRESHOLD).astype(np.float32)
    target = (target > 0.5).astype(np.float32)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return intersection / (union + 1e-7)

# -------- Loop Over Test Images --------
all_dice_scores = []
all_iou_scores = []

image_paths = sorted(glob(os.path.join(TEST_IMG_DIR, '*.png')))
print(f"[INFO] Found {len(image_paths)} test images.")

for img_path in tqdm(image_paths):
    fname = os.path.basename(img_path)

    # --- Load & Transform Image ---
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    augmented = transform(image=resized)
    input_tensor = augmented['image'].unsqueeze(0).to(DEVICE)

    # --- Inference ---
    with torch.no_grad():
        pred = model(input_tensor)
        pred_mask = torch.sigmoid(pred).squeeze().cpu().numpy()

    # --- Load Ground Truth Mask ---
    mask_path = os.path.join(TEST_MASK_DIR, fname)
    if os.path.exists(mask_path):
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.resize(gt_mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
        gt_mask = (gt_mask > 0).astype(np.float32)

        # --- Score ---
        dice = dice_score(pred_mask, gt_mask)
        iou = iou_score(pred_mask, gt_mask)
        all_dice_scores.append(dice)
        all_iou_scores.append(iou)

    # --- Save Predicted Mask ---
    pred_binary = (pred_mask > THRESHOLD).astype(np.uint8) * 255
    save_path = os.path.join(SAVE_PRED_DIR, fname)
    cv2.imwrite(save_path, pred_binary)

# -------- Plot Dice & IoU --------
if all_dice_scores:
    x = range(len(all_dice_scores))
    plt.figure(figsize=(12, 5))
    plt.plot(x, all_dice_scores, label='Dice Score', marker='o')
    plt.plot(x, all_iou_scores, label='IoU Score', marker='s')
    plt.xlabel('Image Index')
    plt.ylabel('Score')
    plt.title('Segmentation Evaluation: Dice vs IoU')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('dice_iou_scores_plot.png')
    plt.show()

    print(f"\n📊 Average Dice Score: {np.mean(all_dice_scores):.4f}")
    print(f"📊 Average IoU Score:  {np.mean(all_iou_scores):.4f}")
