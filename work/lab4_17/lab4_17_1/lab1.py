import cv2
import numpy as np
import matplotlib.pyplot as plt

#1
size = 500
img = np.ones((size, size, 3), dtype=np.uint8) * 255  # 白色背景

cv2.rectangle(img, (100, 100), (400, 400), (0,0,0), 2)
cv2.circle(img, (250, 250), 50, (0,0,0), 2)
for y in range(150, 351, 50):
    cv2.line(img, (100, y), (400, y), (0,0,0), 1)
for x in range(150, 351, 50):
    cv2.line(img, (x, 100), (x, 400), (0,0,0), 1)

cv2.imwrite("test_image.png", img)

# 2
scale = 0.6
angle = 30
center = (size//2, size//2)
M_similar = cv2.getRotationMatrix2D(center, angle, scale)
img_similar = cv2.warpAffine(img, M_similar, (size, size))
cv2.imwrite("similar_transform.png", img_similar)

#3
pts1 = np.float32([[100,100], [400,100], [100,400]])
pts2 = np.float32([[150,150], [350,100], [50,400]])
M_affine = cv2.getAffineTransform(pts1, pts2)
img_affine = cv2.warpAffine(img, M_affine, (size, size))
cv2.imwrite("affine_transform.png", img_affine)

#4
pts1 = np.float32([[100,100], [400,100], [100,400], [400,400]])
pts2 = np.float32([[50,200], [450,150], [80,450], [420,380]])
M_perspective = cv2.getPerspectiveTransform(pts1, pts2)
img_perspective = cv2.warpPerspective(img, M_perspective, (size, size))
cv2.imwrite("perspective_transform.png", img_perspective)

#5
plt.figure(figsize=(16, 8))

plt.subplot(1, 4, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Test Image")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(cv2.cvtColor(img_similar, cv2.COLOR_BGR2RGB))
plt.title("Similarity Transform")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(cv2.cvtColor(img_affine, cv2.COLOR_BGR2RGB))
plt.title("Affine Transform")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(cv2.cvtColor(img_perspective, cv2.COLOR_BGR2RGB))
plt.title("Perspective Transform")
plt.axis("off")

plt.tight_layout()
plt.show()
