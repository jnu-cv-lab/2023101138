import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plot

# mission1——读取图片
img_path="/mnt/e/amnesiac/Pictures/lkl/IMG_20241128_154344.jpg"
img=cv2.imread(img_path)

if img is None:
    raise FileNotFoundError("无法读取图片")
else:
    print("图片读取成功")

# mission2——输出图片基本信息
H,W,Channels=img.shape
photo_type=img.dtype

print(f"图像宽度 = {W}，图像高度 = {H}")
print(f"图像通道数 = {Channels}")
print(f"图像类型 = {photo_type}")

# mission3——显示原图（因为显示不了所以改成保存）
img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

plot.figure("原图",figsize=(8,6))
plot.imshow(img_rgb)
plot.axis("off")
plot.title("beautiful photo")
plot.show(block=False)
#plot.savefig("original_photo.png", dpi=300, bbox_inches="tight")

# mission4——转为灰度图
gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

plot.figure("灰图",figsize=(8,6))
plot.imshow(gray_img,cmap="gray")
plot.axis("off")
plot.title("gray photo")
plot.show()

# mission5——保存灰度图
cv2.imwrite("gray_result.png",gray_img)
print("灰图已保存")

# mission6——输出某个像素值
photo_val = img_rgb[0, 0]
print(f"左上角像素(0,0)的RGB值：{photo_val}")
