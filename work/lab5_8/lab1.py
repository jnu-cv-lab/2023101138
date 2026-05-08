from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

digits = load_digits()
X = digits.data
y = digits.target
images = digits.images

#mission1
print("===== 任务1 数据集信息 =====")
print("样本总数：", len(images))
print("图像尺寸：", images.shape[1], "x", images.shape[2])
print("标签：", np.unique(y))

plt.figure(figsize=(10, 3))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(images[i], cmap="gray")
    plt.title(str(y[i]))
    plt.axis("off")
plt.show()
#plt.savefig("lab1_sample.png")

#mission2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print("\n===== 任务2:数据划分 =====")
print("训练集数量：", len(X_train))
print("测试集数量：", len(X_test))

#mission3
print("\n===== 任务3:特征表示 =====")
print("原始图像形状:", digits.images.shape)
print("展平后特征向量形状:", digits.data.shape)
print("单张图像的特征向量维度:", digits.data[0].shape)

#mission4
models = {
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=10000),
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

results = {}
print("\n===== 任务4:模型准确率 =====")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name:20s} {acc:.4f}")

#mission6
print("\n===== 任务6:错误样本分析(SVM) =====")
svm = SVC()
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("混淆矩阵：")
print(cm)

err = np.sum(y_pred != y_test)
print("错误分类数量: ", err)

