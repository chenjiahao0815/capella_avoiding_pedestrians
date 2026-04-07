#!/usr/bin/env python3
"""在容器中直接测试 ONNX 推理，不依赖 ROS"""
import cv2
import numpy as np

model_path = "/capella/lib/python3.10/site-packages/capella_avoiding_pedestrians/src/src/avoid_person.onnx"

image_path = "/capella/lib/python3.10/site-packages/capella_avoiding_pedestrians/src/src/1.png"

print(f"OpenCV version: {cv2.__version__}")
print(f"Build info (DNN): {[l.strip() for l in cv2.getBuildInformation().split(chr(10)) if 'dnn' in l.lower() or 'cuda' in l.lower()]}")

net = cv2.dnn.readNet(model_path)
img = cv2.imread(image_path)
assert img is not None, f"Failed to read {image_path}"

blob = cv2.dnn.blobFromImage(img, 1.0/255.0, (640, 640), (0,0,0), True, False)

# ========== 测试 CPU ==========
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
net.setInput(blob)
out_cpu = net.forward(net.getUnconnectedOutLayersNames())[0]

print(f"\n=== CPU Backend ===")
print(f"Shape: {out_cpu.shape}")
print(f"Non-zero: {np.count_nonzero(out_cpu)} / {out_cpu.size} ({100*np.count_nonzero(out_cpu)/out_cpu.size:.2f}%)")
print(f"Min: {out_cpu.min():.4f}, Max: {out_cpu.max():.4f}, Mean: {out_cpu.mean():.4f}")

# 转置并检查每个通道的非零数
if out_cpu.ndim == 3:
    data = out_cpu[0]  # [84, 8400]
    for ch in range(min(10, data.shape[0])):
        nz = np.count_nonzero(data[ch])
        print(f"  ch[{ch}]: non-zero={nz}/{data.shape[1]}, "
              f"min={data[ch].min():.4f}, max={data[ch].max():.4f}")

# 检测 person (class 0)
det = data.T  # [8400, 84]
class_scores = det[:, 4:]
max_scores = class_scores.max(axis=1)
max_classes = class_scores.argmax(axis=1)
person_mask = (max_classes == 0) & (max_scores > 0.15)
print(f"\nCPU Person detections (>0.15): {person_mask.sum()}")
if person_mask.sum() > 0:
    print(f"  Top scores: {sorted(max_scores[person_mask], reverse=True)[:5]}")

# ========== 测试 GPU ==========
try:
    net2 = cv2.dnn.readNet(model_path)
    net2.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net2.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    net2.setInput(blob)
    out_gpu = net2.forward(net2.getUnconnectedOutLayersNames())[0]

    print(f"\n=== CUDA Backend ===")
    print(f"Shape: {out_gpu.shape}")
    print(f"Non-zero: {np.count_nonzero(out_gpu)} / {out_gpu.size} ({100*np.count_nonzero(out_gpu)/out_gpu.size:.2f}%)")
    print(f"Min: {out_gpu.min():.4f}, Max: {out_gpu.max():.4f}, Mean: {out_gpu.mean():.4f}")

    if out_gpu.ndim == 3:
        data_gpu = out_gpu[0]
        for ch in range(min(10, data_gpu.shape[0])):
            nz = np.count_nonzero(data_gpu[ch])
            print(f"  ch[{ch}]: non-zero={nz}/{data_gpu.shape[1]}, "
                  f"min={data_gpu[ch].min():.4f}, max={data_gpu[ch].max():.4f}")

    det_gpu = data_gpu.T
    class_scores_gpu = det_gpu[:, 4:]
    max_scores_gpu = class_scores_gpu.max(axis=1)
    max_classes_gpu = class_scores_gpu.argmax(axis=1)
    person_mask_gpu = (max_classes_gpu == 0) & (max_scores_gpu > 0.15)
    print(f"\nGPU Person detections (>0.15): {person_mask_gpu.sum()}")
    if person_mask_gpu.sum() > 0:
        print(f"  Top scores: {sorted(max_scores_gpu[person_mask_gpu], reverse=True)[:5]}")

    # 对比差异
    diff = np.abs(out_cpu - out_gpu)
    print(f"\n=== CPU vs GPU Diff ===")
    print(f"Max diff: {diff.max():.6f}, Mean diff: {diff.mean():.6f}")

except Exception as e:
    print(f"\nCUDA backend failed: {e}")