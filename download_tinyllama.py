# /hy-tmp/Lingshu-7B/download_tinyllama.py
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def download_tinyllama_model():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    target_path = "./TinyLlama-1.1B-Chat" # 模型将下载到 /hy-tmp/Lingshu-7B/TinyLlama-1.1B-Chat

    # 检查目标路径是否已存在，避免重复下载
    if os.path.exists(target_path) and os.path.isdir(target_path) and len(os.listdir(target_path)) > 5:
        # 简单检查目录是否存在且非空，认为已下载
        print(f"模型目录 '{target_path}' 已存在且包含文件，跳过下载。")
        return

    print(f"正在下载 '{model_name}' 模型到 '{target_path}'...")

    # 确保目标路径存在
    os.makedirs(target_path, exist_ok=True)

    try:
        # 下载分词器
        print(f'Downloading tokenizer for {model_name}...')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(target_path)
        print('Tokenizer downloaded and saved.')

        # 下载模型
        print(f'Downloading model for {model_name}...')
        # 注意：这里我们只下载模型文件，不进行量化加载，因为量化加载会在 ai_pipeline_modules 中进行
        # device_map="auto" 可能会尝试将模型加载到GPU，但我们只保存文件，所以影响不大
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.save_pretrained(target_path)
        print('Model downloaded and saved.')

        print(f"'{model_name}' 模型已成功下载到 '{target_path}'")

    except Exception as e:
        print(f"下载模型时发生错误: {e}")
        print("请检查网络连接或模型名称是否正确。")
        # 如果下载失败，清理可能不完整的文件
        if os.path.exists(target_path):
            print(f"下载失败，正在清理不完整的目录: {target_path}")
            # import shutil
            # shutil.rmtree(target_path) # 谨慎操作，避免误删
    
    # 验证下载结果 (可选)
    print("\n验证下载内容:")
    if os.path.exists(target_path) and os.path.isdir(target_path):
        for root, dirs, files in os.walk(target_path):
            level = root.replace(target_path, '').count(os.sep)
            indent = ' ' * 4 * (level)
            print(f'{indent}{os.path.basename(root)}/')
            subindent = ' ' * 4 * (level + 1)
            for f in files:
                print(f'{subindent}{f}')
    else:
        print(f"目录 '{target_path}' 不存在或不是一个目录。")


if __name__ == "__main__":
    # 确保 transformers 库已安装
    try:
        import transformers
    except ImportError:
        print("transformers 库未安装，正在安装...")
        os.system("pip install transformers")
        import transformers # 重新导入
    
    # 确保 torch 库已安装
    try:
        import torch
    except ImportError:
        print("torch 库未安装，正在安装...")
        os.system("pip install torch") # 仅安装CPU版本，如果需要CUDA请使用官方命令
        import torch # 重新导入

    download_tinyllama_model()