import cv2
import numpy as np
import matplotlib.pyplot as plt

size = 512
img = np.zeros((size, size), dtype=np.uint8)
# 左半部分：细条纹（高梯度/边缘区），右半部分：粗条纹（低梯度/平坦区）
for x in range(size):
    if x < size//2:
        # 左半：细条纹（高频，高梯度）
        img[:, x] = 127 + 127 * np.sin(x * 0.3)
    else:
        # 右半：粗条纹（低频，低梯度/平坦）
        img[:, x] = 127 + 127 * np.sin(x * 0.05)

# ===================== 2. 固定参数（和前两部分统一） =====================
M_base = 4
sigma_fixed = 0.45 * M_base  # 全图统一σ=1.8（理论值）

# ===================== 3. 全图统一下采样（基准对比） =====================
# 高斯模糊 + 下采样
img_uniform_blur = cv2.GaussianBlur(img, (9, 9), sigma_fixed)
img_uniform_down = img_uniform_blur[::M_base, ::M_base]
# 上采样回原尺寸，用于计算误差
img_uniform_up = cv2.resize(img_uniform_down, (size, size), interpolation=cv2.INTER_NEAREST)

# ===================== 4. 自适应下采样（严格按题目要求） =====================
# --- 步骤1：梯度分析（找边缘/平坦区，估计局部M值）---
# 用Sobel算子算梯度（OpenCV入门级边缘检测）
grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
grad_mag = np.sqrt(grad_x**2 + grad_y**2)
# 归一化到0-255，方便阈值分割
grad_mag = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# --- 步骤2：根据梯度分区域，设置不同局部M和σ ---
# 阈值划分：高梯度（边缘区）、中梯度、低梯度（平坦区）
M_local = np.ones_like(img, dtype=np.float32) * M_base  # 基础M=4
sigma_local = np.ones_like(img, dtype=np.float32) * sigma_fixed  # 基础σ=1.8

# 高梯度（边缘区）：M更小（少下采样，保细节），σ更小（少模糊）
M_local[grad_mag > 150] = M_base / 2  # 局部M=2
sigma_local[grad_mag > 150] = 0.45 * (M_base / 2)  # σ=0.45*M=0.45

# 低梯度（平坦区）：M更大（多下采样，省空间），σ更大（多模糊，防混叠）
M_local[grad_mag < 50] = M_base * 2  # 局部M=8
sigma_local[grad_mag < 50] = 0.45 * (M_base * 2)  # σ=0.45*M=3.6

# --- 步骤3：分区域高斯滤波 + 下采样（极简入门版）---
# 对高/低梯度区域分别用不同σ模糊，再融合
# 高梯度区（边缘）用小σ
img_edge = cv2.GaussianBlur(img, (9, 9), sigma_local[grad_mag > 150].mean())
# 低梯度区（平坦）用大σ
img_flat = cv2.GaussianBlur(img, (9, 9), sigma_local[grad_mag < 50].mean())
# 融合：边缘区用edge图，平坦区用flat图
img_adaptive_blur = np.where(grad_mag > 150, img_edge, img_flat)
img_adaptive_blur = np.where(grad_mag < 50, img_flat, img_adaptive_blur)

# 自适应下采样（按局部M，这里用平均M简化，入门友好）
M_adaptive = M_local.mean()
img_adaptive_down = img_adaptive_blur[::int(M_adaptive), ::int(M_adaptive)]
# 上采样回原尺寸，用于计算误差
img_adaptive_up = cv2.resize(img_adaptive_down, (size, size), interpolation=cv2.INTER_NEAREST)

# ===================== 5. 计算误差图（题目要求对比） =====================
# 绝对误差图：原图 - 下采样后上采样的图
error_uniform = cv2.absdiff(img, img_uniform_up)
error_adaptive = cv2.absdiff(img, img_adaptive_up)
# 计算均方误差MSE（量化对比）
mse_uniform = np.mean((img - img_uniform_up) ** 2)
mse_adaptive = np.mean((img - img_adaptive_up) ** 2)

# ===================== 6. 保存所有图片 =====================
cv2.imwrite("part3_original.png", img)
cv2.imwrite("part3_uniform_down.png", img_uniform_down)
cv2.imwrite("part3_adaptive_down.png", img_adaptive_down)
cv2.imwrite("part3_error_uniform.png", error_uniform)
cv2.imwrite("part3_error_adaptive.png", error_adaptive)
cv2.imwrite("part3_gradient.png", grad_mag)

# ===================== 7. 一个窗口显示全部（按题目要求） =====================
plt.figure(figsize=(18, 10))

# 第一行：原图、梯度图、统一下采样、自适应下采样
plt.subplot(2, 3, 1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(grad_mag, cmap="gray")
plt.title("Gradient Map (局部M估计)")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(img_uniform_down, cmap="gray")
plt.title(f"Uniform Downsample (σ={sigma_fixed:.1f}, M={M_base})")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(img_adaptive_down, cmap="gray")
plt.title("Adaptive Downsample (不同区域不同σ)")
plt.axis("off")

# 第二行：统一下采样误差图、自适应下采样误差图
plt.subplot(2, 3, 5)
plt.imshow(error_uniform, cmap="hot")
plt.title(f"Uniform Error (MSE={mse_uniform:.1f})")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(error_adaptive, cmap="hot")
plt.title(f"Adaptive Error (MSE={mse_adaptive:.1f})")
plt.axis("off")

plt.tight_layout()
plt.show()
