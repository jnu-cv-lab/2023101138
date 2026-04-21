import cv2
import numpy as np
import matplotlib.pyplot as plt
import os 

img_path="/mnt/e/amnesiac/Pictures/Saved Pictures/微信图片_20260421201140_289_6.jpg"
img = cv2.imread(img_path)

if img is None:
    exit()

points = []

def mouseClick(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append([x, y])
        cv2.circle(img, (x, y), 8, (0, 0, 255), -1)
        cv2.imshow("click 4 points", img)

cv2.imshow("click 4 points", img)
cv2.setMouseCallback("click 4 points", mouseClick)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 确保点了4个点
if len(points) != 4:
    exit()

pts1 = np.float32(points)

width, height = 500, 700
pts2 = np.float32([[0,0], [width,0], [width,height], [0,height]])

M = cv2.getPerspectiveTransform(pts1, pts2)
img_corrected = cv2.warpPerspective(img, M, (width, height))

current_dir = os.path.dirname(os.path.abspath(__file__))
cv2.imwrite(os.path.join(current_dir, "original_a4.jpg"), img)
cv2.imwrite(os.path.join(current_dir, "corrected_a4.jpg"), img_corrected)

plt.figure(figsize=(12,6))
plt.subplot(121)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis('off')

plt.subplot(122)
plt.imshow(cv2.cvtColor(img_corrected, cv2.COLOR_BGR2RGB))
plt.title("Corrected")
plt.axis('off')

plt.show()