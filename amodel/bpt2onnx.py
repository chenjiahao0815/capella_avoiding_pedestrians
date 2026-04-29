#!/usr/bin/env python3
"""
Convert YOLO PT model to ONNX for capella_inspection_cpp node.
Usage:
    python3 convert_to_onnx.py
    python3 convert_to_onnx.py /capella/lib/python3.10/site-packages/capella_inspection_node/2-24.pt -o /home/linux/ws2/capella_inspection_cpp/config/2-24.onnx
"""

import argparse
import sys
import shutil
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError as e:
    print(f"Error: {e}\nInstall with:  pip install ultralytics")
    sys.exit(1)


def convert(pt_path: Path, onnx_path: Path, imgsz: int = 640):
    pt_path = pt_path.resolve()
    if not pt_path.exists():
        raise FileNotFoundError(f"PT model not found: {pt_path}")

    onnx_path = onnx_path.resolve()
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {pt_path}")
    model = YOLO(str(pt_path))

    print(f"\nExport config:")
    print(f"  Input size : {imgsz}x{imgsz}")
    print(f"  Precision  : FP32 (half=False)")
    print(f"  Simplify   : True (ultralytics built-in)")
    print(f"  Opset      : 12")
    print(f"  Dynamic    : False (static batch=1)")
    print(f"  Output     : {onnx_path}\n")

    model.export(
        format='onnx',
        imgsz=imgsz,
        half=False,        # FP32：火灾/烟雾检测建议保留全精度
        int8=False,
        simplify=True,     # 使用 ultralytics 内置简化（需 onnxsim）
        opset=12,
        dynamic=False
    )

    # ultralytics 默认生成与 .pt 同目录的 .onnx
    default_onnx = pt_path.with_suffix('.onnx')
    
    # 兼容：某些版本会生成带 _640x640 后缀的文件
    if not default_onnx.exists():
        candidates = sorted(pt_path.parent.glob(f"{pt_path.stem}*.onnx"))
        if candidates:
            default_onnx = candidates[0]
    
    if not default_onnx.exists():
        raise RuntimeError("Export failed: ONNX file not generated.")

    # 移动到目标路径
    if default_onnx.resolve() != onnx_path.resolve():
        shutil.move(str(default_onnx), str(onnx_path))
        print(f"Moved to: {onnx_path}")
    else:
        print(f"Saved to: {onnx_path}")

    # 验证并打印 I/O 信息（方便你写 C++ 预处理）
    try:
        import onnx
        m = onnx.load(str(onnx_path))
        onnx.checker.check_model(m)
        print("\nONNX validation passed.")
        
        print("\nModel I/O (for your C++ code):")
        for inp in m.graph.input:
            shape = [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]
            print(f"  Input : {inp.name}  -> {shape}")
        for out in m.graph.output:
            shape = [d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]
            print(f"  Output: {out.name}  -> {shape}")
    except ImportError:
        print("\nSkip validation (onnx package not installed).")
    
    return onnx_path


def main():
    parser = argparse.ArgumentParser(description="YOLO PT -> ONNX converter")
    parser.add_argument("pt", nargs="?", default=None, help="Path to .pt model")
    parser.add_argument("-o", "--output", default=None, help="Output ONNX path")
    parser.add_argument("-s", "--imgsz", type=int, default=640, help="Input image size")
    
    args = parser.parse_args()

    # 默认使用你原来的 2-24.pt
    if args.pt is None:
        args.pt = "/capella/lib/python3.10/site-packages/capella_inspection_node/2-24.pt"
    
    pt = Path(args.pt)
    
    if args.output is None:
        # 默认输出到你的 C++ ROS2 包 config 目录
        config_dir = Path("/home/linux/c++/amodel/")
        config_dir.mkdir(exist_ok=True)
        args.output = str(config_dir / f"{pt.stem}.onnx")
    
    try:
        convert(pt, Path(args.output), args.imgsz)
        print(f"\nDone. Reference this path in your C++ node:")
        print(f'  std::string model_path = "{Path(args.output)}";')
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()