import os
import json
import torch
import numpy as np
import cv2
import mediapipe as mp
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from dataset import BadmintonSkeletonDataset
from torch.utils.data import DataLoader
from model import SkeletonTransformer
from preprocess import extract_video_skeleton, TARGET_FRAMES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "./best_model.pth"
DATA_PROCESS = "./data/processed"
LABEL_MAP_PATH = os.path.join(DATA_PROCESS, "label_map.json")


with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
    idx2cls = json.load(f)
idx2cls = {int(k): v for k, v in idx2cls.items()}
cls_names = list(idx2cls.values())


model = SkeletonTransformer().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

def evaluate_testset():
    """测试集完整评估：准确率、混淆矩阵、分类报告"""
    test_ds = BadmintonSkeletonDataset(
        os.path.join(DATA_PROCESS, "X_test.npy"),
        os.path.join(DATA_PROCESS, "y_test.npy")
    )
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for seq, label in test_loader:
            seq = seq.to(DEVICE)
            logits = model(seq)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(pred)
            all_labels.extend(label.numpy())

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"测试集整体准确率: {acc:.4f}")

    print("\n==== 分类报告 ====")
    print(classification_report(all_labels, all_preds, target_names=cls_names, labels=list(range(6))))

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(6)))
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=cls_names, yticklabels=cls_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("./confusion_matrix.png")
    print("混淆矩阵已保存 confusion_matrix.png")

def infer_single_video(vid_path):
    """单视频推理，输出类别与置信度"""
    skeleton_seq = extract_video_skeleton(vid_path)
    if skeleton_seq is None:
        print("视频提取骨架失败，无有效人体关键点")
        return
    seq_tensor = torch.from_numpy(skeleton_seq).unsqueeze(0).to(DEVICE)  # [1,30,132]
    with torch.no_grad():
        logits = model(seq_tensor)
        prob = torch.softmax(logits, dim=1)[0]
        pred_idx = torch.argmax(prob).item()
        conf = prob[pred_idx].item()
        pred_cls = idx2cls[pred_idx]
    print("==== 单视频推理结果 ====")
    print(f"Predicted class: {pred_cls}")
    print(f"Confidence: {conf:.2f}")
    return pred_cls, conf

if __name__ == "__main__":

    evaluate_testset()

    demo_video = "./demo_video.mp4"
    if os.path.exists(demo_video):
        infer_single_video(demo_video)