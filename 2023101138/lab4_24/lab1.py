import cv2
import numpy as np

#misson1
box_img = cv2.imread("/mnt/e/图片/Screenshots/屏幕截图 2026-04-24 124244.png")         #E:\图片\Screenshots\屏幕截图 2026-04-24 124229.png
scene_img = cv2.imread("/mnt/e/图片/Screenshots/屏幕截图 2026-04-24 124229.png")   #E:\图片\Screenshots\屏幕截图 2026-04-24 124244.png

orb = cv2.ORB_create(nfeatures=1000)

kp_box, des_box = orb.detectAndCompute(box_img, None)
kp_scene, des_scene = orb.detectAndCompute(scene_img, None)

box_kp_img = cv2.drawKeypoints(box_img, kp_box, None, color=(0, 255, 0), flags=0)
scene_kp_img = cv2.drawKeypoints(scene_img, kp_scene, None, color=(0, 255, 0), flags=0)

print("=== box.png 特征信息 ===")
print(f"关键点数量：{len(kp_box)}")
print(f"描述子维度：{des_box.shape[1] if des_box is not None else 0}")
print("\n=== box_in_scene.png 特征信息 ===")
print(f"关键点数量：{len(kp_scene)}")
print(f"描述子维度：{des_scene.shape[1] if des_scene is not None else 0}")

# cv2.imwrite("mission1_1.png", box_kp_img)
# cv2.imwrite("mission1_2.png", scene_kp_img)

# cv2.imshow("box.png ORB Keypoints", box_kp_img)
# cv2.imshow("box_in_scene.png ORB Keypoints", scene_kp_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#mission2
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des_box, des_scene)

matches = sorted(matches, key=lambda x: x.distance)

print("\n===== ORB 特征匹配结果 =====")
print("总匹配数量：", len(matches))

img_matches = cv2.drawMatches(box_img, kp_box, scene_img, kp_scene, matches[:50], None, flags=2)

# cv2.imwrite("mission2.png", img_matches)

#mission3

pts_box = []
pts_scene = []
for m in matches:
    pts_box.append(kp_box[m.queryIdx].pt)
    pts_scene.append(kp_scene[m.trainIdx].pt)

pts_box = np.float32(pts_box)
pts_scene = np.float32(pts_scene)

H, mask = cv2.findHomography(pts_box, pts_scene, cv2.RANSAC, 5.0)

matches_mask = mask.ravel().tolist()
num_inliers = sum(matches_mask)
num_matches = len(matches)
inlier_ratio = num_inliers / num_matches

print("\n===== RANSAC 剔除结果 =====")
print(f"总匹配数量：{num_matches}")
print(f"RANSAC内点数量：{num_inliers}")
print(f"内点比例：{inlier_ratio:.2f}")
print("\nHomography 矩阵：")
print(np.round(H, 3))

img_ransac = cv2.drawMatches(box_img, kp_box,scene_img, kp_scene,matches, None,matchesMask=matches_mask,flags=2)

# cv2.imwrite("mission3.png", img_ransac)
# cv2.imshow("RANSAC 正确匹配", img_ransac)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#mission4

h1, w1 = box_img.shape[:2]
pts_box_corner = np.float32([[0, 0],[w1, 0],[w1, h1],[0, h1]]).reshape(-1, 1, 2)


H_inv = np.linalg.inv(H)
pts_scene_corner = cv2.perspectiveTransform(pts_box_corner, H)

img_scene_copy = scene_img.copy()
cv2.polylines(img_scene_copy, [np.int32(pts_scene_corner)], True, (255, 0, 0), 2)

# cv2.imwrite("mission4.png", img_scene_copy)
# cv2.imshow("目标定位结果", img_scene_copy)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

print("\n===== 任务4完成 =====")

#mission5
nfeatures_list = [500, 1000, 2000]

for nfeat in nfeatures_list:
    print("\n==================================")
    print(f"nfeatures = {nfeat}")

    orb = cv2.ORB_create(nfeatures=nfeat)
    kp1, des1 = orb.detectAndCompute(box_img, None)
    kp2, des2 = orb.detectAndCompute(scene_img, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    inliers = sum(mask.ravel())
    ratio = inliers / len(matches)

    try:
        h, w = box_img.shape[:2]
        corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
        cv2.perspectiveTransform(corners, H)
        locate = "成功"
    except:
        locate = "失败"

    print(f"模板关键点数量：{len(kp1)}")
    print(f"场景关键点数量：{len(kp2)}")
    print(f"总匹配数量：{len(matches)}")
    print(f"RANSAC内点数量:{inliers}")
    print(f"内点比例：{ratio:.2f}")
    print(f"目标定位：{locate}")

cv2.destroyAllWindows()