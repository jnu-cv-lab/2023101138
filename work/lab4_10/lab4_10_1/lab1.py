import numpy as np
import cv2
import matplotlib.pyplot as plt

#1-1
def create_checkerboard(size=512, block_size=16):
    img = np.zeros((size, size), dtype=np.uint8)
    for i in range(0, size, block_size * 2):
        for j in range(0, size, block_size * 2):
            img[i:i+block_size, j:j+block_size] = 255
            img[i+block_size:i+2*block_size, j+block_size:j+2*block_size] = 255
    return img

# 1-2
def create_chirp(size=512):
    x = np.linspace(-np.pi, np.pi, size)
    y = np.linspace(-np.pi, np.pi, size)
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx**2 + yy**2)
    img = np.sin(r * size / 4) * 127 + 128
    return img.astype(np.uint8)

# 1-3
def direct_downsample(img, M):
    return img[::M, ::M]

# 1-4
def gaussian_downsample(img, M, sigma):
    ksize = int(2 * np.ceil(3 * sigma) + 1)
    blurred = cv2.GaussianBlur(img, (ksize, ksize), sigma)
    return blurred[::M, ::M]

# ====================== 主程序开始 ======================
M = 4
sigma = 0.45 * M

# 生成所有图像
checker = create_checkerboard()
chirp = create_chirp()

checker_direct = direct_downsample(checker, M)
chirp_direct = direct_downsample(chirp, M)

checker_gauss = gaussian_downsample(checker, M, sigma)
chirp_gauss = gaussian_downsample(chirp, M, sigma)

# ====================== 保存每张图片（单独保存） ======================
cv2.imwrite("checker_original.png", checker)
cv2.imwrite("checker_direct.png", checker_direct)
cv2.imwrite("checker_gaussian.png", checker_gauss)

cv2.imwrite("chirp_original.png", chirp)
cv2.imwrite("chirp_direct.png", chirp_direct)
cv2.imwrite("chirp_gaussian.png", chirp_gauss)

# ====================== 所有图放在一个窗口显示 ======================
plt.figure(figsize=(12, 8))

# 第一行：棋盘格
plt.subplot(2, 3, 1)
plt.imshow(checker, cmap='gray')
plt.title("Checker Original")

plt.subplot(2, 3, 2)
plt.imshow(checker_direct, cmap='gray')
plt.title("Checker Direct Down")

plt.subplot(2, 3, 3)
plt.imshow(checker_gauss, cmap='gray')
plt.title("Checker Gaussian Down")

# 第二行：Chirp
plt.subplot(2, 3, 4)
plt.imshow(chirp, cmap='gray')
plt.title("Chirp Original")

plt.subplot(2, 3, 5)
plt.imshow(chirp_direct, cmap='gray')
plt.title("Chirp Direct Down")

plt.subplot(2, 3, 6)
plt.imshow(chirp_gauss, cmap='gray')
plt.title("Chirp Gaussian Down")

plt.show()