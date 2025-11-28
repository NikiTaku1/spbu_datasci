"""
Training script for document orientation using MobileNetV3-Small with debug outputs.
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import OrientationDataset
from model import get_model
import argparse


def train(data_dir, epochs=6, batch_size=16, lr=1e-4, out="orientation_mobilenet.pth"):
    csv = os.path.join(data_dir, "annotations.csv")
    img_dir = os.path.join(data_dir, "images")

    if not os.path.exists(csv) or not os.path.exists(img_dir):
        raise FileNotFoundError(f"Dataset not found in {data_dir}")

    ds = OrientationDataset(csv, img_dir)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)

    print(f"Dataset loaded: {len(ds)} samples")
    print(f"Batch size: {batch_size}, Total batches per epoch: {len(dl)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = get_model(num_classes=4, pretrained=True).to(device)
    print("Model created: MobileNetV3-Small with 4-class head")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for ep in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        print(f"Starting epoch {ep+1}/{epochs}")

        for i, (x, y) in enumerate(dl):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            if (i+1) % 10 == 0 or (i+1) == len(dl):
                print(f"Batch {i+1}/{len(dl)} - loss: {loss.item():.4f}, batch accuracy: {(preds==y).float().mean():.4f}")

        epoch_loss = running_loss / len(dl)
        epoch_acc = correct / total
        print(f"Epoch {ep+1} finished - avg loss: {epoch_loss:.4f}, accuracy: {epoch_acc:.4f}")

    if not isinstance(out, str) or out.strip() == "":
        out = "orientation_mobilenet.pth"
        print(f"Output path not valid, using default: {out}")

    torch.save(model.state_dict(), out)
    print(f"Model saved at {out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train MobileNetV3-Small orientation model")
    parser.add_argument('--data', default='synthetic_orb_realistic', help='Path to dataset folder')
    parser.add_argument('--epochs', type=int, default=6, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--out', default='orientation_mobilenet.pth', help='Output model path')
    args = parser.parse_args()

    print("Starting training")
    train(data_dir=args.data, epochs=args.epochs, batch_size=args.batch, out=args.out)
    print("Training completed")