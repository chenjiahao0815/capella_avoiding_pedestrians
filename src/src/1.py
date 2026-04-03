from pathlib import Path
from collections import Counter

import numpy as np
import onnx
import onnxruntime as ort


def pick_onnx_file():
    cwd = Path(__file__).resolve().parent
    onnx_files = sorted(cwd.glob("*.onnx"))
    if not onnx_files:
        raise FileNotFoundError(f"当前目录没有找到 .onnx 文件: {cwd}")
    if len(onnx_files) > 1:
        print("发现多个 onnx，默认使用第一个：")
        for f in onnx_files:
            print(" -", f.name)
    return onnx_files[0]


def main():
    onnx_path = pick_onnx_file()
    print(f"使用模型: {onnx_path.name}")

    model = onnx.load(str(onnx_path))
    print("\n=== 基本信息 ===")
    print("ir_version:", model.ir_version)
    print("producer_name:", repr(model.producer_name))
    print("producer_version:", repr(model.producer_version))
    print("domain:", repr(model.domain))
    print("model_version:", model.model_version)

    print("\n=== opset ===")
    for opset in model.opset_import:
        domain = opset.domain if opset.domain else "ai.onnx"
        print(f"- {domain}: {opset.version}")

    if model.metadata_props:
        print("\n=== metadata ===")
        for kv in model.metadata_props:
            print(f"- {kv.key}: {kv.value}")
    else:
        print("\n=== metadata ===")
        print("无")

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    print("\n=== 输入 ===")
    for i, inp in enumerate(sess.get_inputs()):
        print(f"[{i}] name={inp.name}, shape={inp.shape}, type={inp.type}")

    print("\n=== 输出 ===")
    for i, out in enumerate(sess.get_outputs()):
        print(f"[{i}] name={out.name}, shape={out.shape}, type={out.type}")

    inp = sess.get_inputs()[0]
    shape = inp.shape

    if len(shape) != 4:
        print("\n输入不是常见的 [N,C,H,W]，脚本先不继续推理。")
        return

    n = 1 if isinstance(shape[0], str) or shape[0] is None else int(shape[0])
    c = 3 if isinstance(shape[1], str) or shape[1] is None else int(shape[1])
    h = 640 if isinstance(shape[2], str) or shape[2] is None else int(shape[2])
    w = 640 if isinstance(shape[3], str) or shape[3] is None else int(shape[3])

    dummy = np.random.rand(n, c, h, w).astype(np.float32)
    outputs = sess.run(None, {inp.name: dummy})

    print("\n=== 推理输出统计 ===")
    for idx, arr in enumerate(outputs):
        arr = np.asarray(arr)
        print(f"\nOutput[{idx}]")
        print("shape =", arr.shape)
        print("dtype =", arr.dtype)
        print("min   =", float(arr.min()))
        print("max   =", float(arr.max()))
        print("mean  =", float(arr.mean()))
        print("nonzero =", int(np.count_nonzero(arr)), "/", arr.size)

        if arr.ndim == 3 and arr.shape[0] == 1:
            print("\n看起来像检测头输出")
            print("channels =", arr.shape[1], "candidates =", arr.shape[2])

            if arr.shape[1] >= 5:
                num_classes = arr.shape[1] - 4
                print("推测类别数 =", num_classes)

                det = arr[0].T
                class_scores = det[:, 4:]
                best_cls = np.argmax(class_scores, axis=1)
                best_score = np.max(class_scores, axis=1)

                print("best_score min/max/mean =",
                      float(best_score.min()),
                      float(best_score.max()),
                      float(best_score.mean()))

                print("\n=== best_class 分布（前10）===")
                counter = Counter(best_cls.tolist())
                for cls_id, cnt in counter.most_common(10):
                    print(f"class {cls_id}: {cnt}")

                for thr in [0.1, 0.25, 0.5]:
                    count = int(np.sum(best_score >= thr))
                    print(f"best_score >= {thr}: {count}")

                cls0 = (best_cls == 0)
                print("\n=== class 0 统计 ===")
                print("best_class == 0:", int(cls0.sum()))
                for thr in [0.1, 0.25, 0.5]:
                    count = int(np.sum(cls0 & (best_score >= thr)))
                    print(f"class0 且 score >= {thr}: {count}")


if __name__ == "__main__":
    main()