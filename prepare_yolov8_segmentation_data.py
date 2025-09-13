# /hy-tmp/Lingshu-7B/prepare_yolov8_segmentation_data.py
import os
import shutil
import random
import cv2
import numpy as np

# 定义数据集根路径
DATASET_ROOT = '/hy-tmp/Lingshu-7B/COVID-19_Radiography_Dataset'
# 定义输出数据目录 (YOLOv8分割训练所需格式)
OUTPUT_DATA_ROOT = '/hy-tmp/Lingshu-7B/yolov8_segmentation_data'

# 划分比例
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1 # 剩余部分

# 确保比例总和为1
assert TRAIN_RATIO + VAL_RATIO + TEST_RATIO == 1.0, "划分比例总和必须为1.0"

# 类别名称到ID的映射 (根据文件夹名称的字母顺序)
# COVID -> 0, Lung_Opacity -> 1, Normal -> 2, Viral Pneumonia -> 3
CLASS_TO_ID = {
    'COVID': 0,
    'Lung_Opacity': 1,
    'Normal': 2,
    'Viral Pneumonia': 3
}

def convert_mask_to_yolo_segment(mask_path, image_width, image_height):
    """
    将二值化掩码图片转换为 YOLOv8 分割标签格式 (归一化多边形坐标)。
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"警告: 无法读取掩码文件: {mask_path}")
        return ""

    # 确保掩码是二值化的
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 查找轮廓
    # cv2.RETR_EXTERNAL 只检索最外层轮廓
    # cv2.CHAIN_APPROX_SIMPLE 压缩水平、垂直和对角线段，只保留它们的端点
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    yolo_segments = []
    for contour in contours:
        # 排除过小的轮廓 (可能是噪声)
        if cv2.contourArea(contour) < 10: # 面积阈值可调整
            continue

        # 将轮廓点转换为归一化坐标
        segment = contour.reshape(-1, 2).astype(np.float32)
        segment[:, 0] /= image_width
        segment[:, 1] /= image_height

        # 将归一化坐标展平为字符串
        yolo_segments.append(" ".join(map(str, segment.flatten())))
    
    return " ".join(yolo_segments)

def prepare_segmentation_data(dataset_root, output_root, train_ratio, val_ratio, test_ratio):
    print(f"开始准备 YOLOv8 分割训练数据，源目录: {dataset_root}")
    print(f"目标目录: {output_root}")

    # 清理旧的输出目录 (如果存在)
    if os.path.exists(output_root):
        print(f"检测到旧的输出目录 {output_root}，正在删除...")
        shutil.rmtree(output_root)
    
    # 创建训练、验证、测试子目录及其内部的 images/labels 目录
    for subset in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_root, subset, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_root, subset, 'labels'), exist_ok=True)

    # 遍历每个类别文件夹
    class_names = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]
    class_names = [c for c in class_names if c not in ['train', 'val', 'test', 'yolov8_classification_data']] # 排除其他目录
    class_names.sort() # 确保类别顺序一致
    print(f"检测到类别: {class_names}")

    if not class_names:
        print(f"错误: 在 {dataset_root} 中未找到任何类别文件夹。请检查数据集路径和结构。")
        return

    all_images_with_labels = [] # 存储所有图片及其对应的类别ID和路径

    for class_name in class_names:
        class_id = CLASS_TO_ID.get(class_name)
        if class_id is None:
            print(f"警告: 类别 '{class_name}' 未在 CLASS_TO_ID 中定义，跳过。")
            continue

        images_dir = os.path.join(dataset_root, class_name, 'images')
        masks_dir = os.path.join(dataset_root, class_name, 'masks')

        if not os.path.exists(images_dir):
            print(f"警告: 类别 '{class_name}' 下未找到 'images' 目录，跳过。")
            continue
        if not os.path.exists(masks_dir):
            print(f"警告: 类别 '{class_name}' 下未找到 'masks' 目录，跳过。")
            continue

        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_file in image_files:
            image_path = os.path.join(images_dir, img_file)
            mask_path = os.path.join(masks_dir, img_file) # 假设掩码文件名与图片文件名一致

            if not os.path.exists(mask_path):
                print(f"警告: 图片 {img_file} 缺少对应的掩码文件，跳过。")
                continue
            
            # 读取图片以获取尺寸
            img = cv2.imread(image_path)
            if img is None:
                print(f"警告: 无法读取图片文件: {image_path}，跳过。")
                continue
            h, w, _ = img.shape

            # 转换掩码为YOLO格式
            yolo_segment_str = convert_mask_to_yolo_segment(mask_path, w, h)
            
            if yolo_segment_str: # 只有当成功生成分割信息时才添加
                all_images_with_labels.append({
                    'image_path': image_path,
                    'class_id': class_id,
                    'yolo_segment': yolo_segment_str,
                    'original_filename': img_file # 原始文件名
                })
            else:
                print(f"警告: 图片 {img_file} 未生成有效分割信息，跳过。")

    random.shuffle(all_images_with_labels) # 打乱所有图片

    num_images = len(all_images_with_labels)
    num_train = int(num_images * train_ratio)
    num_val = int(num_images * val_ratio)

    train_data = all_images_with_labels[:num_train]
    val_data = all_images_with_labels[num_train : num_train + num_val]
    test_data = all_images_with_labels[num_train + num_val :]

    print(f"\n总计处理 {num_images} 张图片。")
    print(f"  训练集: {len(train_data)} 张")
    print(f"  验证集: {len(val_data)} 张")
    print(f"  测试集: {len(test_data)} 张")

    # 将数据复制到目标目录并生成标签文件
    def copy_and_create_labels(data_list, subset_name):
        for item in data_list:
            original_img_path = item['image_path']
            original_filename_no_ext = os.path.splitext(item['original_filename'])[0]
            
            # 目标图片路径
            target_img_dir = os.path.join(output_root, subset_name, 'images')
            target_img_path = os.path.join(target_img_dir, item['original_filename'])
            shutil.copy(original_img_path, target_img_path)

            # 目标标签路径
            target_label_dir = os.path.join(output_root, subset_name, 'labels')
            target_label_path = os.path.join(target_label_dir, f"{original_filename_no_ext}.txt")
            
            with open(target_label_path, 'w', encoding='utf-8') as f:
                # YOLOv8 分割标签格式: <class-id> <x1> <y1> <x2> <y2> ...
                f.write(f"{item['class_id']} {item['yolo_segment']}\n")

    print("\n正在复制训练集图片和生成标签...")
    copy_and_create_labels(train_data, 'train')
    print("正在复制验证集图片和生成标签...")
    copy_and_create_labels(val_data, 'val')
    print("正在复制测试集图片和生成标签...")
    copy_and_create_labels(test_data, 'test')

    print("\n数据准备完成！")
    print(f"数据已组织到 {output_root} 目录下，并生成了YOLOv8分割标签。")

if __name__ == "__main__":
    # 确保 opencv-python 已安装
    try:
        import cv2
    except ImportError:
        print("opencv-python 库未安装，正在安装...")
        os.system("pip install opencv-python")
        import cv2 # 重新导入
    
    prepare_segmentation_data(DATASET_ROOT, OUTPUT_DATA_ROOT, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)