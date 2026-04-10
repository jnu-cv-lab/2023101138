import numpy as np
import cv2
import matplotlib.pyplot as plt

# ===================== 生成清晰条纹图（绝对不报错！） =====================
size = 512
img = np.zeros((size, size), dtype=np.uint8)

# 生成竖条纹：越往右条纹越细，高频明显，一眼看出混叠
for x in range(size):
    freq = x / 800   # 频率越来越高
    val = 127 + 127 * np.sin(x * freq * 10)
    img[:, x] = val.astype(np.uint8)

# ===================== 下采样倍数固定 M=4 =====================
M = 4

# 1. 直接下采样（混叠最严重）
direct = img[::M, ::M]

# 2. 不同 sigma 高斯模糊 + 下采样
s05  = cv2.GaussianBlur(img, (9,9), 0.5)[::M, ::M]
s10  = cv2.GaussianBlur(img, (9,9), 1.0)[::M, ::M]
s18  = cv2.GaussianBlur(img, (9,9), 1.8)[::M, ::M]
s20  = cv2.GaussianBlur(img, (9,9), 2.0)[::M, ::M]
s40  = cv2.GaussianBlur(img, (9,9), 4.0)[::M, ::M]

# ===================== 保存所有图片 =====================
cv2.imwrite("direct.png", direct)
cv2.imwrite("sigma_0.5.png", s05)
cv2.imwrite("sigma_1.0.png", s10)
cv2.imwrite("sigma_1.8.png", s18)
cv2.imwrite("sigma_2.0.png", s20)
cv2.imwrite("sigma_4.0.png", s40)

# ===================== 一个窗口显示 =====================
plt.figure(figsize=(18,5))

plt.subplot(1,6,1)
plt.imshow(direct, cmap='gray')
plt.title("Direct\n(混叠严重)")

plt.subplot(1,6,2)
plt.imshow(s05, cmap='gray')
plt.title("σ=0.5\n(混叠)")

plt.subplot(1,6,3)
plt.imshow(s10, cmap='gray')
plt.title("σ=1.0\n(轻微混叠)")

plt.subplot(1,6,4)
plt.imshow(s18, cmap='gray')
plt.title("σ=1.8\n(最优)")

plt.subplot(1,6,5)
plt.imshow(s20, cmap='gray')
plt.title("σ=2.0\n(有点糊)")

plt.subplot(1,6,6)
plt.imshow(s40, cmap='gray')
plt.title("σ=4.0\n(太糊)")

plt.show()