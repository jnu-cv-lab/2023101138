import os
import json
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import requests

# ===================== 配置参数 =====================
RAW_DATA_ROOT = "/home/amnesiac/cv-course2/mygithub_1/2023101138/data/raw"
PROCESS_SAVE_ROOT = "/home/amnesiac/cv-course2/mygithub_1/2023101138/data/processed"
TARGET_FRAMES = 30
TEST_SPLIT_RATIO = 0.2
KEYPOINT_NUM = 33
FEAT_PER_KPT = 4
FRAME_DIM = KEYPOINT_NUM * FEAT_PER_KPT
MODEL_PATH = "pose_landmarker_lite.task"


if not os.path.exists(MODEL_PATH):
    print("正在下载姿态检测模型...")
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    res = requests.get(url)
    with open(MODEL_PATH, "wb") as f:
        f.write(res.content)


from mediapipe.tasks import python
from mediapipe.tasks.python import vision
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    min_pose_detection_confidence=0.1,
    min_tracking_confidence=0.1
)

CLASS_MAP = {
    "forehand_drive": 0,
    "forehand_lift": 1,
    "forehand_net_shot": 2,
    "forehand_clear": 3,
    "backhand_drive": 4,
    "backhand_net_shot": 5
}
INV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}
os.makedirs(PROCESS_SAVE_ROOT, exist_ok=True)

def normalize_skeleton(skel_seq):
    seq_len, feat_dim = skel_seq.shape
    skel_seq = skel_seq.reshape(seq_len, KEYPOINT_NUM, FEAT_PER_KPT)
    norm_seq = []
    for frame in skel_seq:
        hip_l = frame[23, :3]
        hip_r = frame[24, :3]
        hip_center = (hip_l + hip_r) / 2.0
        shoulder_l = frame[11, :3]
        shoulder_r = frame[12, :3]
        shoulder_width = np.linalg.norm(shoulder_l - shoulder_r)
        if shoulder_width < 1e-6:
            shoulder_width = 1.0
        frame[:, :3] = (frame[:, :3] - hip_center) / shoulder_width
        norm_seq.append(frame.reshape(-1))
    return np.array(norm_seq)

def resample_frames(seq, target_len):
    n_frames, feat_dim = seq.shape
    if n_frames == target_len:
        return seq
    old_idx = np.linspace(0, n_frames - 1, n_frames)
    new_idx = np.linspace(0, n_frames - 1, target_len)
    resampled = np.zeros((target_len, feat_dim))
    for d in range(feat_dim):
        resampled[:, d] = np.interp(new_idx, old_idx, seq[:, d])
    return resampled

def extract_video_skeleton(vid_path):
    detector = vision.PoseLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(vid_path)
    skeleton_frames = []
    ts = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    step_ts = int(1000 / fps) if fps > 0 else 33
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect_for_video(mp_img, ts)
        ts += step_ts
        if result.pose_landmarks:
            frame_feat = []
            for lm in result.pose_landmarks[0]:
                frame_feat.extend([lm.x, lm.y, lm.z, lm.visibility])
            skeleton_frames.append(frame_feat)
    cap.release()
    detector.close()
    if len(skeleton_frames) == 0:
        print(f"跳过视频 {os.path.basename(vid_path)}：未检测到有效人体骨架")
        return None
    skel_arr = np.array(skeleton_frames)
    skel_norm = normalize_skeleton(skel_arr)
    return resample_frames(skel_norm, TARGET_FRAMES)

def process_all_videos():
    all_samples = []
    all_labels = []
    for cls_name, cls_id in CLASS_MAP.items():
        cls_dir = os.path.join(RAW_DATA_ROOT, cls_name)
        if not os.path.exists(cls_dir):
            print(f"警告：{cls_dir} 文件夹不存在，跳过")
            continue
        vid_list = [f for f in os.listdir(cls_dir) if f.endswith((".mp4", ".avi", ".mov", ".mkv"))]
        print(f"\n处理类别 {cls_name}，共{len(vid_list)}个视频")
        if len(vid_list) == 0:
            print(f"提示：{cls_name} 文件夹无视频")
            continue
        for vid_name in tqdm(vid_list):
            skel_data = extract_video_skeleton(os.path.join(cls_dir, vid_name))
            if skel_data is not None:
                all_samples.append(skel_data)
                all_labels.append(cls_id)

    total_sample = len(all_samples)
    print(f"\n有效样本总数：{total_sample}")
    if total_sample == 0:
        raise RuntimeError("无任何有效人体骨架样本，请更换清晰全身视频！")

    X = np.array(all_samples)
    y = np.array(all_labels)

    if total_sample >= 5:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SPLIT_RATIO, random_state=42, stratify=y
        )
        np.save(os.path.join(PROCESS_SAVE_ROOT, "X_test.npy"), X_test)
        np.save(os.path.join(PROCESS_SAVE_ROOT, "y_test.npy"), y_test)
        print(f"自动划分：训练集{X_train.shape}，测试集{X_test.shape}")
    else:
        X_train = X
        y_train = y
        print(f"样本过少({total_sample}个)，不划分测试集，全部作为训练集")


    np.save(os.path.join(PROCESS_SAVE_ROOT, "X_train.npy"), X_train)
    np.save(os.path.join(PROCESS_SAVE_ROOT, "y_train.npy"), y_train)
    with open(os.path.join(PROCESS_SAVE_ROOT, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(INV_CLASS_MAP, f, ensure_ascii=False, indent=2)
    print("\n数据集文件保存完成")

if __name__ == "__main__":
    process_all_videos()