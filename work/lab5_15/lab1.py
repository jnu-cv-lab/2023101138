import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

#mission1
print("PyTorch 版本:", torch.__version__)
print("GPU 可用:", torch.cuda.is_available())

a = torch.tensor([1, 2, 3])
b = a + 1
print("张量测试：", a, b)

#mission2
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

full_train = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)
test_set = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

train_size = int(0.8 * len(full_train))
val_size = len(full_train) - train_size
train_set, val_set = random_split(full_train, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

print("===== 任务2:数据集信息 =====")
print(f"训练集：{len(train_set)} 张")
print(f"验证集：{len(val_set)} 张")
print(f"测试集：{len(test_set)} 张")

# 显示8张样本并保存
plt.figure(figsize=(10, 4))
for i in range(8):
    img, label = full_train[i]
    plt.subplot(1, 8, i + 1)
    plt.imshow(img.squeeze(), cmap='gray')
    plt.title(f"number: {label}")
    plt.axis('off')

plt.tight_layout()
#plt.savefig("mnist_samples.png")
plt.close()
print("已保存样本图:mnist_samples.png")

#mission3
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
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

model = SimpleCNN()
print("===== 任务3:CNN模型结构 =====")
print(model)

test_input = torch.randn(1, 1, 28, 28)
output = model(test_input)
print("\n输入图像尺寸:", test_input.shape)
print("输出预测结果尺寸:", output.shape)
print("模型定义完成,可正常接收28x28的手写数字图像并输出10类预测结果")

#mission4 + mission5
print("\n===== 任务4 + 任务5:训练并验证 =====")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5
history = {
    "train_loss": [], "train_acc": [],
    "val_loss": [], "val_acc": []
}

for epoch in range(epochs):
    # 训练
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    avg_train_loss = train_loss / len(train_loader)
    avg_train_acc = 100 * train_correct / train_total
    history["train_loss"].append(avg_train_loss)
    history["train_acc"].append(avg_train_acc)

    # 验证
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    avg_val_acc = 100 * val_correct / val_total
    history["val_loss"].append(avg_val_loss)
    history["val_acc"].append(avg_val_acc)

    print(f"第 {epoch+1} 轮")
    print(f"训练损失: {avg_train_loss:.4f} | 训练准确率: {avg_train_acc:.2f}%")
    print(f"验证损失: {avg_val_loss:.4f} | 验证准确率: {avg_val_acc:.2f}%")
    print("-" * 50)

print("\n任务4、任务5完成")

# mission6
print("\n===== 任务6:测试模型并展示预测结果 =====")

model.eval()
test_loss = 0.0
test_correct = 0
test_total = 0
samples = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

        if len(samples) < 8:
            for i in range(min(8 - len(samples), images.size(0))):
                samples.append((images[i].cpu(), labels[i].item(), predicted[i].item()))

avg_test_loss = test_loss / len(test_loader)
avg_test_acc = 100 * test_correct / test_total
print(f"测试损失: {avg_test_loss:.4f} | 测试准确率: {avg_test_acc:.2f}%")

plt.figure(figsize=(12, 4))
for i, (img, true_label, pred_label) in enumerate(samples):
    plt.subplot(1, 8, i + 1)
    plt.imshow(img.squeeze(), cmap='gray')
    plt.title(f"T:{true_label}\nP:{pred_label}")
    plt.axis('off')

plt.tight_layout()
plt.savefig("test_preds.png")
plt.close()
print(" 已保存测试预测图:test_preds.png")
print("\n 任务6完成")

# mission7
print("\n===== 任务7:绘制训练曲线 =====")

epochs_list = list(range(1, epochs + 1))

plt.figure(figsize=(12, 4))
# Loss曲线
plt.subplot(1, 2, 1)
plt.plot(epochs_list, history["train_loss"], label="Train Loss")
plt.plot(epochs_list, history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()
plt.grid(True)

# Acc曲线
plt.subplot(1, 2, 2)
plt.plot(epochs_list, history["train_acc"], label="Train Acc")
plt.plot(epochs_list, history["val_acc"], label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy Curve")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("train_curves.png")
plt.close()
print(" 已保存训练曲线:train_curves.png")
print("\n 任务7完成 ")