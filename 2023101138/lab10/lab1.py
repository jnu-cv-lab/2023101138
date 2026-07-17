import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


# mission1
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

full_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_set = datasets.MNIST('./data', train=False, download=True, transform=transform)
train_set, val_set = random_split(full_train, [48000, 12000])

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(optimizer_type, lr=0.001, epochs=5):
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    if optimizer_type == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_type == "SGD+Momentum":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_type == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    for epoch in range(epochs):
        # 训练
        model.train()
        t_loss, t_correct, t_total = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            _, pred = torch.max(out, 1)
            t_correct += (pred == y).sum().item()
            t_total += y.size(0)
        # 验证
        model.eval()
        v_loss, v_correct, v_total = 0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                v_loss += loss.item()
                _, pred = torch.max(out, 1)
                v_correct += (pred == y).sum().item()
                v_total += y.size(0)
        # 记录
        history["train_loss"].append(t_loss/len(train_loader))
        history["val_loss"].append(v_loss/len(val_loader))
        history["train_acc"].append(t_correct/t_total)
        history["val_acc"].append(v_correct/v_total)
        print(f"[{optimizer_type}] Epoch {epoch+1} | TL:{history['train_loss'][-1]:.4f} VA:{history['val_acc'][-1]:.4f}")
    # 测试
    model.eval()
    test_correct, test_total = 0, 0
    all_preds, all_labels = [], []
    wrong_samples = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            _, pred = torch.max(out, 1)
            test_correct += (pred == y).sum().item()
            test_total += y.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            # 收集错误样本
            for i in range(len(y)):
                if pred[i] != y[i] and len(wrong_samples) < 8:
                    wrong_samples.append((x[i].cpu(), y[i].item(), pred[i].item()))
    test_acc = test_correct / test_total
    return model, history, test_acc, all_labels, all_preds, wrong_samples

#mission2
print("\n===== 任务2:优化器对比 =====")
opts = ["SGD", "SGD+Momentum", "Adam"]
opt_results = {}
for opt in opts:
    model, hist, acc, labels, preds, wrong = train_model(opt, lr=0.001)
    opt_results[opt] = (model, hist, acc, labels, preds, wrong)
    print(f"{opt} Test Acc: {acc:.4f}")

#mission3
print("\n===== 任务3:学习率对比 =====")
lrs = [0.1, 0.01, 0.001]
lr_results = {}
for lr in lrs:
    _, hist, acc, _, _, _ = train_model("Adam", lr=lr)
    lr_results[lr] = (hist, acc)
    print(f"Adam lr={lr} Test Acc: {acc:.4f}")

#mission4
print("\n===== 任务4:卷积核可视化 =====")
model = opt_results["Adam"][0]
conv1_weights = model.conv1.weight.data.cpu().numpy()
plt.figure(figsize=(10, 4))
for i in range(8):
    plt.subplot(1, 8, i+1)
    plt.imshow(conv1_weights[i, 0], cmap="gray")
    plt.axis("off")
#plt.savefig("conv1_kernels.png")
plt.close()

#mission5
print("\n===== 任务5:Feature map可视化 =====")
img, _ = test_set[0]
img_tensor = img.unsqueeze(0).to(device)
model.eval()
with torch.no_grad():
    fm1 = model.relu(model.conv1(img_tensor)).cpu().squeeze().numpy()
plt.figure(figsize=(10, 4))
for i in range(8):
    plt.subplot(1, 8, i+1)
    plt.imshow(fm1[i], cmap="gray")
    plt.axis("off")
#plt.savefig("feature_maps.png")
plt.close()

#mission6
print("\n===== 任务6:错误样本 =====")
wrong_samples = opt_results["Adam"][5]
plt.figure(figsize=(12, 4))
for i, (img, true_lbl, pred_lbl) in enumerate(wrong_samples):
    plt.subplot(1, 8, i+1)
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(f"T:{true_lbl}\nP:{pred_lbl}")
    plt.axis("off")
#plt.savefig("wrong_samples.png")
plt.close()

#mission7
print("\n===== 任务7:混淆矩阵 =====")
labels, preds = opt_results["Adam"][3], opt_results["Adam"][4]
cm = confusion_matrix(labels, preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Pred")
plt.ylabel("True")
plt.title("Confusion Matrix")
#plt.savefig("confusion_matrix.png")
plt.close()
