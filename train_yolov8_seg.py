# /hy-tmp/Lingshu-7B/train_yolov8_seg.py
from ultralytics import YOLO
import os
import shutil
import torch

# 定义数据和模型路径
DATA_YAML_PATH = '/hy-tmp/Lingshu-7B/covid_segmentation.yaml'
PRETRAINED_MODEL = 'yolov8n-seg.pt' # YOLOv8 nano版本分割模型
FINE_TUNED_MODEL_SAVE_PATH = '/hy-tmp/Lingshu-7B/yolov8_finetuned_seg.pt' # 微调后模型保存路径

def train_yolov8_segmentation():
    print(f"开始 YOLOv8 分割模型微调...")
    print(f"使用数据集配置: {DATA_YAML_PATH}")
    print(f"使用预训练模型: {PRETRAINED_MODEL}")

    # 1. 加载预训练的YOLOv8分割模型
    # 如果本地没有，ultralytics会自动下载
    model = YOLO(PRETRAINED_MODEL)
    print("YOLOv8 分割模型加载成功！")

    # 2. 开始训练
    # data: 指向data.yaml文件
    # epochs: 训练轮次，可以根据需要调整。医疗影像数据集可能需要更多epochs。
    # imgsz: 输入图像尺寸，640是YOLOv8常用尺寸，可根据显存调整
    # batch: 批处理大小，根据显存调整。3060 12GB 应该可以运行 batch=8 或 16。
    # name: 训练结果保存的目录名 (在 runs/segment/ 目录下)
    # project: 训练结果保存的父目录
    # device: 指定训练设备，0表示第一个GPU
    # patience: 如果连续n个epoch验证集性能没有提升，则提前停止训练
    # cache: True/disk 缓存数据到RAM/磁盘，加速数据加载
    results = model.train(
        data=DATA_YAML_PATH,
        epochs=100, # 初始尝试100个epoch，可根据验证集性能调整
        imgsz=640,  # 常用尺寸，可尝试416或512以节省显存
        batch=8,    # 3060 12GB 显存，分割任务可能比分类更耗显存，先尝试8
        name='covid_segmentation_run',
        project='/hy-tmp/Lingshu-7B/runs', # 训练结果将保存在此目录下
        device=0 if torch.cuda.is_available() else 'cpu', # 使用GPU进行训练
        patience=20, # 如果20个epoch验证集性能没有提升，则提前停止
        cache='disk' # 缓存数据到RAM，加速训练
    )

    print("\nYOLOv8 分割模型微调完成！")

    # 3. 保存微调后的模型权重
    # Ultralytics通常会自动保存最佳模型到 runs/segment/name/weights/best.pt
    # 我们可以手动复制或指定保存路径
    best_model_path = os.path.join('/hy-tmp/Lingshu-7B/runs/segment/covid_segmentation_run/weights', 'best.pt')
    if os.path.exists(best_model_path):
        shutil.copy(best_model_path, FINE_TUNED_MODEL_SAVE_PATH)
        print(f"最佳模型权重已保存到: {FINE_TUNED_MODEL_SAVE_PATH}")
    else:
        print("警告: 未找到最佳模型权重。请检查训练日志。")

if __name__ == "__main__":
    # 确保 ultralytics 库已安装
    try:
        import ultralytics
    except ImportError:
        print("ultralytics 库未安装，正在安装...")
        os.system("pip install ultralytics")
        import ultralytics
    
    # 确保 PyTorch 已安装并支持CUDA
    if not torch.cuda.is_available():
        print("警告: 未检测到CUDA GPU。训练将在CPU上进行，速度会非常慢。")
    
    train_yolov8_segmentation()