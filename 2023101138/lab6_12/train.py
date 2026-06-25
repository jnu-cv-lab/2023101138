import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataset import BadmintonSkeletonDataset
from model import SkeletonTransformer


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "./data/processed"
BATCH_SIZE = 16
LR = 1e-3
EPOCHS = 20
SAVE_MODEL_PATH = "./best_model.pth"

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for seq, label in tqdm(loader, desc="Train"):
        seq, label = seq.to(DEVICE), label.to(DEVICE)
        optimizer.zero_grad()
        logits = model(seq)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = torch.argmax(logits, dim=1)
        correct += (pred == label).sum().item()
        total += label.size(0)
    avg_loss = total_loss / len(loader)
    acc = correct / total
    return avg_loss, acc

def val_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for seq, label in tqdm(loader, desc="Val"):
            seq, label = seq.to(DEVICE), label.to(DEVICE)
            logits = model(seq)
            loss = criterion(logits, label)
            total_loss += loss.item()
            pred = torch.argmax(logits, dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)
    avg_loss = total_loss / len(loader)
    acc = correct / total
    return avg_loss, acc

if __name__ == "__main__":

    train_ds = BadmintonSkeletonDataset(
        data_npy=os.path.join(DATA_PATH, "X_train.npy"),
        label_npy=os.path.join(DATA_PATH, "y_train.npy")
    )
    test_ds = BadmintonSkeletonDataset(
        data_npy=os.path.join(DATA_PATH, "X_test.npy"),
        label_npy=os.path.join(DATA_PATH, "y_test.npy")
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)


    model = SkeletonTransformer().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_acc = 0.0
    train_loss_list, train_acc_list = [], []
    val_loss_list, val_acc_list = [], []

    for epoch in range(EPOCHS):
        print(f"\n==== Epoch {epoch+1}/{EPOCHS} ====")
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = val_epoch(model, test_loader, criterion)
        print(f"Train Loss:{tr_loss:.4f} Acc:{tr_acc:.4f} | Val Loss:{val_loss:.4f} Acc:{val_acc:.4f}")

        train_loss_list.append(tr_loss)
        train_acc_list.append(tr_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), SAVE_MODEL_PATH)
            print(f"保存最优模型，最佳准确率: {best_acc:.4f}")


    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(train_loss_list, label="Train Loss")
    plt.plot(val_loss_list, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(train_acc_list, label="Train Acc")
    plt.plot(val_acc_list, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("./train_curve.png")
    print("训练曲线已保存 train_curve.png")