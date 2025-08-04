import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

print("🚀 训练脚本启动")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("警告：未找到cv2模块，将使用PIL进行图像处理")

try:
    from dyunet import DyUNet
    print("成功导入dyunet模块")
except ImportError:
    print("错误：未找到dyunet模块")
    class DyUNet(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        def forward(self, x):
            return self.conv(x)

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))

        assert len(self.image_files) == len(self.mask_files), "图像和掩码文件数量不匹配"
        for img, msk in zip(self.image_files, self.mask_files):
            img_name = os.path.splitext(img)[0]
            msk_name = os.path.splitext(msk)[0].replace("_mask", "")
            assert img_name == msk_name, f"文件不匹配: {img} vs {msk}"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Ensure single channel

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask

class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum()
        dice = (2. * intersection + self.smooth) / (pred_sigmoid.sum() + target.sum() + self.smooth)
        bce = self.bce(pred, target)
        return bce + (1 - dice)

def dice_coef(pred, target, smooth=1e-5):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def iou_score(pred, target, smooth=1e-5):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    base_data_path = "/root/autodl-tmp/dyunet_data"
    print(f"数据路径: {base_data_path}")

    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > 0.5).float())
    ])

    train_dataset = SegmentationDataset(
        image_dir=os.path.join(base_data_path, 'imgs'),
        mask_dir=os.path.join(base_data_path, 'masks'),
        image_transform=image_transform,
        mask_transform=mask_transform
    )

    val_dataset = train_dataset

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    model = DyUNet(in_channels=3, out_channels=1).to(device)
    print(f"模型结构:\n{model}")

    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    os.makedirs("output", exist_ok=True)

    def predict(model, image_path, image_transform, device):
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        image_tensor = image_transform(image).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            output = model(image_tensor)
            probs = torch.sigmoid(output)
            predicted_mask = (probs > 0.5).float()

        predicted_mask = predicted_mask.squeeze().cpu().numpy()
        predicted_mask = (predicted_mask * 255).astype(np.uint8)

        if CV2_AVAILABLE:
            return cv2.resize(predicted_mask, original_size, interpolation=cv2.INTER_NEAREST)
        else:
            return Image.fromarray(predicted_mask).resize(original_size, Image.NEAREST)

    def visualize(image_path, mask_path, prediction, save_path):
        image = Image.open(image_path).convert('RGB')
        true_mask = Image.open(mask_path).convert('L')

        plt.figure(figsize=(15, 10))
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title('原始图像')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(true_mask, cmap='gray')
        plt.title('真实掩码')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(prediction, cmap='gray')
        plt.title('预测结果')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    for epoch in range(50):
        print(f"\nEpoch {epoch + 1}/50")
        start_time = time.time()

        # Training
        model.train()
        train_loss = 0.0
        for images, masks in tqdm(train_loader, desc="Training"):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation with metrics
        model.eval()
        val_loss = 0.0
        dice_total, iou_total = 0.0, 0.0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validation"):
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                preds = torch.sigmoid(outputs)
                dice_total += dice_coef(preds, masks).item()
                iou_total += iou_score(preds, masks).item()

        avg_val_loss = val_loss / len(val_loader)
        avg_dice = dice_total / len(val_loader)
        avg_iou = iou_total / len(val_loader)
        val_losses.append(avg_val_loss)

        scheduler.step()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'output/best_model.pth')
            print(f"保存最佳模型，验证损失: {avg_val_loss:.4f}, Dice: {avg_dice:.4f}, IoU: {avg_iou:.4f}")

        epoch_time = time.time() - start_time
        print(f"训练损失: {avg_train_loss:.4f}, 验证损失: {avg_val_loss:.4f}, Dice: {avg_dice:.4f}, IoU: {avg_iou:.4f}, 耗时: {epoch_time:.2f}s")

        # Save loss curve
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='训练损失')
        plt.plot(val_losses, label='验证损失')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.title('训练和验证损失')
        plt.legend()
        plt.savefig('output/loss_curve.png')
        plt.close()

        if (epoch + 1) % 5 == 0:
            sample_image_path = os.path.join(base_data_path, 'imgs', os.listdir(os.path.join(base_data_path, 'imgs'))[0])
            sample_mask_path = os.path.join(base_data_path, 'masks', os.listdir(os.path.join(base_data_path, 'masks'))[0])
            prediction = predict(model, sample_image_path, image_transform, device)
            if isinstance(prediction, Image.Image):
                prediction = np.array(prediction)
            visualize(sample_image_path, sample_mask_path, prediction,
                      f'output/prediction_epoch_{epoch + 1}.png')
            print(f"保存预测结果: output/prediction_epoch_{epoch + 1}.png")

    print("训练完成！")
    print(f"最佳模型已保存至: output/best_model.pth")

if __name__ == "__main__":
    main()
