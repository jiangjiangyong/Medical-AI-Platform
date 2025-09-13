# /hy-tmp/Lingshu-7B/minicpm_cop_generator.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import json

# 定义模型路径
minicpm_model_path = "/hy-tmp/Lingshu-7B/MiniCPM-2B-dpo-fp16"

print(f"正在加载 MiniCPM-2B-dpo-fp16 分词器和模型: {minicpm_model_path}...")

# 检查GPU是否可用
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"检测到设备: {device}")

# 加载分词器
tokenizer_minicpm = AutoTokenizer.from_pretrained(minicpm_model_path)

# 加载模型
# MiniCPM-2B-dpo-fp16 已经是fp16版本，可以直接加载，device_map="auto" 确保在GPU上
model_minicpm = AutoModelForCausalLM.from_pretrained(
    minicpm_model_path,
    torch_dtype=torch.float16, # 模型已经是fp16，保持一致
    device_map="auto",
    trust_remote_code=True 
)
model_minicpm.eval()
print("MiniCPM-2B-dpo-fp16 模型和分词器加载成功！")

# --- 模拟加载 Qwen 生成的专业报告 ---
# 在实际项目中，这个文件是 Qwen 生成的
professional_report_path = '/hy-tmp/Lingshu-7B/qwen_output/professional_report_draft_rag_kg.txt'
professional_report_content = ""
if os.path.exists(professional_report_path):
    with open(professional_report_path, 'r', encoding='utf-8') as f:
        professional_report_content = f.read()
    print(f"\n成功加载 Qwen 生成的专业报告：\n{professional_report_content[:200]}...") # 打印前200字
else:
    print(f"\n警告: 未找到 Qwen 生成的专业报告文件: {professional_report_path}。将使用示例内容。")
    professional_report_content = """
    影像所见：
    胸部X光片示右肺上叶可见一直径约1.5cm的类圆形高密度影，边缘毛糙，可见分叶征。左肺未见明显异常。心影大小形态正常，大血管无异常。

    初步诊断：
    右肺上叶结节，性质待定，恶性可能性不能排除。

    建议：
    1. 建议进一步行胸部高分辨率CT（HRCT）检查以明确结节性质。
    2. 必要时考虑PET-CT或穿刺活检。
    3. 定期随访。
    """

# --- 构建 MiniCPM 的 Prompt，用于生成科普报告 ---
messages_minicpm = [
    {"role": "user", "content": (
        "你是一名专业的医疗科普作者，擅长将复杂的医学报告转化为通俗易懂、面向患者的科普报告。请根据以下专业诊断报告，为患者撰写一份简洁明了、易于理解的科普报告。报告应解释病情、可能的治疗方案和注意事项，避免使用晦涩的医学术语。\n\n"
        "以下是专业诊断报告：\n"
        f"{professional_report_content}\n\n"
        "请开始撰写科普报告："
    )}
]

text_minicpm = tokenizer_minicpm.apply_chat_template(
    messages_minicpm,
    tokenize=False,
    add_generation_prompt=True
)

print("\n--- 构建的 MiniCPM Prompt ---")
print(text_minicpm)

# --- MiniCPM 生成科普报告 ---
model_inputs_minicpm = tokenizer_minicpm(text_minicpm, return_tensors="pt").to(device)

print("\n正在生成患者科普报告 (MiniCPM-2B-dpo-fp16)...")

generated_ids_minicpm = model_minicpm.generate(
    model_inputs_minicpm.input_ids,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1,
    pad_token_id=tokenizer_minicpm.eos_token_id
)

generated_text_minicpm = tokenizer_minicpm.decode(generated_ids_minicpm[0], skip_special_tokens=True)

# 提取模型生成的报告部分
response_start_index_minicpm = generated_text_minicpm.find(text_minicpm)
if response_start_index_minicpm != -1:
    cop_report = generated_text_minicpm[response_start_index_minicpm + len(text_minicpm):].strip()
else:
    cop_report = generated_text_minicpm.strip()

print("\n--- 生成的患者科普报告 ---")
print(cop_report)

# 将科普报告保存到文件
output_cop_report_path = '/hy-tmp/Lingshu-7B/minicpm_output/patient_cop_report.txt'
os.makedirs(os.path.dirname(output_cop_report_path), exist_ok=True)
with open(output_cop_report_path, 'w', encoding='utf-8') as f:
    f.write(cop_report)
print(f"\n患者科普报告已保存到: {output_cop_report_path}")