import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import json

# 新增导入 LangChain 相关的库
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceBgeEmbeddings # <-- 注释掉旧的导入
from langchain_huggingface import HuggingFaceEmbeddings # <-- 使用新的导入
from langchain.prompts import PromptTemplate

# --- 1. Qwen 模型加载 ---
# 定义模型路径
qwen_model_path = "/hy-tmp/Lingshu-7B/Qwen1.5-1.8B-Chat"

print(f"正在加载 Qwen1.5-1.8B-Chat 分词器和模型: {qwen_model_path}...")

# 检查GPU是否可用
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"检测到设备: {device}")

# 加载分词器
tokenizer_qwen = AutoTokenizer.from_pretrained(qwen_model_path)

# 加载模型
model_qwen = AutoModelForCausalLM.from_pretrained(
    qwen_model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)
model_qwen.eval()
print("Qwen1.5-1.8B-Chat 模型和分词器加载成功！")

# --- 2. 加载 YOLOv8 检测结果与模拟患者信息 ---
yolov8_output_json_path = '/hy-tmp/Lingshu-7B/yolov8_output/detection_results.json'
detected_objects_info = []
if os.path.exists(yolov8_output_json_path):
    with open(yolov8_output_json_path, 'r', encoding='utf-8') as f:
        detected_objects_info = json.load(f)
    print(f"成功加载 YOLOv8 检测结果，共检测到 {len(detected_objects_info)} 个物体。")
else:
    print(f"警告: 未找到 YOLOv8 检测结果文件: {yolov8_output_json_path}。将使用空列表。")

# 模拟患者信息 (在真实项目中，这些信息可能来自HIS系统或医生输入)
patient_info = {
    "patient_id": "P0012345",
    "age": 45,
    "gender": "男",
    "exam_type": "胸部X光片",
    "clinical_symptoms": "持续咳嗽，伴有轻微胸痛，无发热。"
}

# 将 YOLOv8 检测结果转换为 LLM 友好的文本描述
yolov8_text_description = ""
if detected_objects_info:
    yolov8_text_description += "影像分析结果（YOLOv8检测）：\n"
    for i, obj in enumerate(detected_objects_info):
        yolov8_text_description += (
            f"  {i+1}. 检测到 '{obj['class_name']}'，置信度 {obj['confidence']:.2f}，"
            f"位于图像区域 {obj['bbox']}。\n"
        )
    yolov8_text_description += "请注意，此处检测到的物体是通用模型识别结果，在真实医学场景中需替换为专业病灶检测。\n"
else:
    yolov8_text_description += "影像分析结果（YOLOv8检测）：未检测到明显异常。请注意，此处为通用模型识别结果。\n"

print("\n--- 整合后的输入信息 ---")
print(f"患者信息: {patient_info}")
print(f"YOLOv8文本描述:\n{yolov8_text_description}")

# --- 3. RAG 模块初始化 ---
print("\n--- RAG 模块初始化 ---")

# 3.1. 加载知识库文档
knowledge_base_dir = "/hy-tmp/Lingshu-7B/medical_knowledge_base"
documents = []
# 确保知识库目录存在
if not os.path.exists(knowledge_base_dir):
    print(f"错误: 知识库目录 {knowledge_base_dir} 不存在。请先创建并放置医学知识文件。")
else:
    for file_name in os.listdir(knowledge_base_dir):
        if file_name.endswith(".txt"):
            file_path = os.path.join(knowledge_base_dir, file_name)
            loader = TextLoader(file_path, encoding='utf-8')
            documents.extend(loader.load())
    print(f"加载了 {len(documents)} 篇医学知识文档。")

# 3.2. 文本分割
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)
print(f"文档分割后得到 {len(docs)} 个文本块。")

# 3.3. 选择嵌入模型 (现在使用 moka-ai/m3e-base)
# !!! 注意：这里已修改为从本地路径加载 moka-ai/m3e-base 嵌入模型 !!!
local_embedding_model_path = "/hy-tmp/Lingshu-7B/embeddings/m3e-base" # 新的本地路径
print(f"正在加载嵌入模型: {local_embedding_model_path}...")
# 使用新的 HuggingFaceEmbeddings 类
# m3e-base模型通常不需要normalize_embeddings=True，但为了通用性可以保留
embeddings = HuggingFaceEmbeddings(
    model_name=local_embedding_model_path, # 从本地路径加载
    model_kwargs={'device': device},
    encode_kwargs={'normalize_embeddings': False} # m3e-base默认不归一化，可根据需要调整
)
print("嵌入模型加载成功！")

# 3.4. 构建或加载向量数据库 (使用ChromaDB)
chroma_db_path = "/hy-tmp/Lingshu-7B/chroma_db"
if not os.path.exists(chroma_db_path) or not os.listdir(chroma_db_path):
    print(f"正在创建并持久化向量数据库到 {chroma_db_path}...")
    db = Chroma.from_documents(docs, embeddings, persist_directory=chroma_db_path)
    db.persist()
    print("向量数据库创建并持久化完成！")
else:
    print(f"从 {chroma_db_path} 加载现有向量数据库...")
    db = Chroma(persist_directory=chroma_db_path, embedding_function=embeddings)
    print("向量数据库加载完成！")

# 3.5. 定义检索器
retriever = db.as_retriever(search_kwargs={"k": 3})
print("RAG 模块初始化完成。")

# --- 4. 知识图谱模块初始化 (模拟) ---
print("\n--- 知识图谱模块初始化 (模拟) ---")

medical_knowledge_graph = {
    "肺结节": {
        "症状": ["无症状", "咳嗽", "胸痛"],
        "影像特征": ["圆形阴影", "毛糙边缘", "钙化"],
        "诊断方法": ["胸部CT", "X光片", "活检"],
        "相关疾病": ["肺癌", "结核", "炎性假瘤"]
    },
    "骨折": {
        "症状": ["疼痛", "肿胀", "畸形", "功能障碍"],
        "影像特征": ["骨折线", "移位"],
        "诊断方法": ["X光片", "CT扫描"],
        "治疗方法": ["复位", "固定", "手术"]
    },
    "肺炎": {
        "症状": ["发热", "咳嗽", "咳痰", "胸痛", "呼吸困难"],
        "影像特征": ["斑片状阴影", "实变", "肺间质浸润"],
        "诊断方法": ["胸部X光片", "CT", "血常规", "病原学检查"],
        "治疗方法": ["抗感染治疗", "抗病毒药物"]
    },
    "person": { # 模拟YOLOv8检测到的通用物体，并将其“映射”到医学语境中的“人体结构”
        "医学关联": ["人体结构", "解剖部位"]
    },
    "bus": {
        "医学关联": ["无直接医学关联", "交通工具"]
    },
    "skateboard": {
        "医学关联": ["无直接医学关联", "运动器械"]
    }
}

def validate_with_kg(text_to_validate, kg):
    """
    一个简单的函数，用于模拟根据知识图谱进行事实校验。
    检查文本中是否包含知识图谱中的核心实体，并尝试验证其关联性。
    """
    validation_results = []
    found_medical_entities = []

    for entity, data in kg.items():
        if entity in text_to_validate:
            found_medical_entities.append(entity)
            # 简单检查是否存在相关症状或特征
            if "症状" in data:
                for symptom in data["症状"]:
                    if symptom in text_to_validate:
                        validation_results.append(f"✅ 文本提及 {entity} 及其症状 '{symptom}'，与知识图谱一致。")
            if "影像特征" in data:
                for feature in data["影像特征"]:
                    if feature in text_to_validate:
                        validation_results.append(f"✅ 文本提及 {entity} 及其影像特征 '{feature}'，与知识图谱一致。")
            if "医学关联" in data:
                 validation_results.append(f"ℹ️ 文本提及 '{entity}'，知识图谱提示其关联：{', '.join(data['医学关联'])}。")
    
    if not found_medical_entities:
        validation_results.append("⚠️ 文本中未检测到与知识图谱直接匹配的医学实体。")

    return "\n".join(validation_results) if validation_results else "✅ 文本未检测到明显不一致，或未提及关键医学实体。"

print("知识图谱模块初始化完成。")

# --- 5. RAG 检索与 Prompt 构建 ---
# 构建检索查询，结合患者症状和YOLOv8检测到的主要物体
query_text = (
    f"患者症状: {patient_info['clinical_symptoms']}。 "
    f"影像检测到主要物体: {', '.join([obj['class_name'] for obj in detected_objects_info if obj['confidence'] > 0.5])}。"
    "请提供与这些信息相关的医学知识。"
)
print(f"\nRAG 检索查询: {query_text}")
retrieved_docs = retriever.invoke(query_text)

context_from_rag = "\n\n相关医学知识（来自RAG）：\n"
if retrieved_docs:
    for i, doc in enumerate(retrieved_docs):
        context_from_rag += f"--- 文档 {i+1} ---\n{doc.page_content}\n"
else:
    context_from_rag += "未检索到相关医学知识。\n"
print(context_from_rag)

# 构建 Prompt
messages = [
    {"role": "system", "content": "你是一名资深的医学影像诊断助手。你的任务是根据患者的临床信息、影像分析结果和提供的相关医学知识，生成一份专业的初步诊断报告草稿。请使用严谨的医学术语，并包含影像所见、初步诊断和建议。请务必参考提供的相关医学知识来减少幻觉。"},
    {"role": "user", "content": (
        f"患者ID: {patient_info['patient_id']}\n"
        f"年龄: {patient_info['age']}岁\n"
        f"性别: {patient_info['gender']}\n"
        f"检查类型: {patient_info['exam_type']}\n"
        f"临床症状: {patient_info['clinical_symptoms']}\n\n"
        f"{yolov8_text_description}\n"
        f"{context_from_rag}\n" # ！！！新增：RAG检索到的知识
        "请根据以上所有信息，生成一份专业的初步诊断报告草稿。报告应包含'影像所见'、'初步诊断'和'建议'三个部分。"
    )}
]

# 将messages转换为模型可接受的输入格式
text = tokenizer_qwen.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

print("\n--- 构建的 LLM Prompt ---")
print(text)

# --- 6. 生成报告草稿并进行知识图谱校验 ---
# 将 Prompt 编码为 token IDs
model_inputs = tokenizer_qwen(text, return_tensors="pt").to(device)

print("\n正在生成专业报告草稿 (Qwen1.5-1.8B-Chat)...")

# 生成文本
generated_ids = model_qwen.generate(
    model_inputs.input_ids,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1,
    pad_token_id=tokenizer_qwen.eos_token_id
)

# 解码生成的 token IDs 为文本
generated_text = tokenizer_qwen.decode(generated_ids[0], skip_special_tokens=True)

# 提取模型生成的报告部分
response_start_index = generated_text.find(text)
if response_start_index != -1:
    report_draft = generated_text[response_start_index + len(text):].strip()
else:
    report_draft = generated_text.strip()

print("\n--- 生成的专业报告草稿 (RAG增强版) ---")
print(report_draft)

# ！！！新增：对生成的报告进行知识图谱校验
print("\n--- 知识图谱初步校验结果 ---")
kg_validation_output = validate_with_kg(report_draft, medical_knowledge_graph)
print(kg_validation_output)

# 将报告保存到文件
output_report_path = '/hy-tmp/Lingshu-7B/qwen_output/professional_report_draft_rag_kg.txt'
os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
with open(output_report_path, 'w', encoding='utf-8') as f:
    f.write(report_draft)
print(f"\n专业报告草稿 (RAG增强版) 已保存到: {output_report_path}")