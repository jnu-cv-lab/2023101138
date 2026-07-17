import os
import cv2
import numpy as np

# ===================== 路径与标定参数配置 =====================
# 存放标定原图的文件夹
IMG_INPUT_PATH = "/home/amnesiac/cv-course2/mygithub_1/2023101138/lab6_26/photo"

# 棋盘格内角点规格 9列×6行
CHESS_W = 9
CHESS_H = 6
# 棋盘方格实际边长(mm)，按需修改 25 / 30
SQUARE_LENGTH = 25
# 亚像素角点细化迭代标准
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# ===================== 1. 构建棋盘格世界三维坐标 =====================
world_points = np.zeros((CHESS_W * CHESS_H, 3), np.float32)
world_points[:, :2] = np.mgrid[0:CHESS_W, 0:CHESS_H].T.reshape(-1, 2)
world_points = world_points * SQUARE_LENGTH

# 存储全部图片的3D世界点、2D图像点
all_world_pts = []
all_img_pts = []
valid_image_paths = []

# ===================== 2. 遍历photo文件夹，检测棋盘角点 =====================
image_names = [file for file in os.listdir(IMG_INPUT_PATH)
               if file.lower().endswith((".jpg", ".png", ".jpeg", ".bmp"))]

if len(image_names) == 0:
    raise Exception(f"文件夹 {IMG_INPUT_PATH} 内没有找到图片，请放入棋盘格标定照片！")

for img_name in image_names:
    full_img_path = os.path.join(IMG_INPUT_PATH, img_name)
    frame = cv2.imread(full_img_path)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测棋盘内角点
    detect_ret, corner_coords = cv2.findChessboardCorners(gray_frame, (CHESS_W, CHESS_H))
    if detect_ret:
        # 亚像素精细化角点
        fine_corners = cv2.cornerSubPix(gray_frame, corner_coords, (11, 11), (-1, -1), SUBPIX_CRITERIA)
        all_world_pts.append(world_points)
        all_img_pts.append(fine_corners)
        valid_image_paths.append(full_img_path)

        # 绘制角点，直接保存到当前目录
        draw_frame = frame.copy()
        cv2.drawChessboardCorners(draw_frame, (CHESS_W, CHESS_H), fine_corners, detect_ret)
        cv2.imwrite(f"corner_{img_name}", draw_frame)
        print(f"【有效图片】{img_name} 角点检测完成，已保存 corner_{img_name}")
    else:
        print(f"【跳过】{img_name} 未检测到完整棋盘格")

print(f"\n总计有效标定图片数量：{len(valid_image_paths)} 张")
if len(valid_image_paths) < 15:
    print("警告：有效图片不足15张，标定结果误差较大，建议补充更多角度照片")

# ===================== 3. 执行相机标定，求解内参、畸变系数、重投影误差 =====================
sample_img = cv2.imread(valid_image_paths[0])
img_size = (sample_img.shape[1], sample_img.shape[0])

reproj_error, camera_matrix, dist_coeff, rotate_vec, trans_vec = cv2.calibrateCamera(
    all_world_pts, all_img_pts, img_size, None, None
)

# 打印标定结果，直接复制到实验报告
print("\n==================== 相机标定输出结果 ====================")
print(f"整体平均重投影误差（像素）：{reproj_error:.4f}")
print("\n相机内参矩阵 K：")
print(camera_matrix)
print("\n径向+切向畸变系数 [k1,k2,p1,p2,k3]：")
print(dist_coeff.ravel())

# ===================== 4. 生成原图&去畸变对比图，直接保存在当前目录 =====================
demo_img_path = valid_image_paths[0]
demo_img = cv2.imread(demo_img_path)
h, w = demo_img.shape[:2]

opt_cam_matrix, crop_roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeff, (w, h), 1, (w, h))
undist_image = cv2.undistort(demo_img, camera_matrix, dist_coeff, None, opt_cam_matrix)

# 裁剪黑边
x_start, y_start, crop_w, crop_h = crop_roi
undist_cropped = undist_image[y_start:y_start+crop_h, x_start:x_start+crop_w]

# 直接输出到代码同级目录
cv2.imwrite("original_demo.jpg", demo_img)
cv2.imwrite("undistort_cropped.jpg", undist_cropped)
print(f"\n去畸变对比图已保存至当前目录：original_demo.jpg、undistort_cropped.jpg")

# ===================== 5. 逐图校验重投影误差 =====================
total_reproj_err = 0
for idx in range(len(all_world_pts)):
    project_points, _ = cv2.projectPoints(all_world_pts[idx], rotate_vec[idx], trans_vec[idx], camera_matrix, dist_coeff)
    single_err = cv2.norm(all_img_pts[idx], project_points, cv2.NORM_L2) / len(project_points)
    total_reproj_err += single_err
avg_single_err = total_reproj_err / len(all_world_pts)
print(f"逐图平均重投影误差校验值：{avg_single_err:.4f}")