# /hy-tmp/Lingshu-7B/ai_pipeline_modules.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import json
import yaml # 用于读取YOLOv8的data.yaml
import uuid # 用于生成唯一文件名
import cv2 # 用于图像处理
import numpy as np # 用于图像处理
from ultralytics import YOLO # YOLOv8 导入
import re # 用于知识图谱校验中的正则匹配

# LangChain 相关的导入
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import PromptTemplate

# --- 全局变量：模型和数据库，只加载一次 ---
tokenizer_qwen = None
model_qwen_gpu = None
embeddings = None
retriever = None
medical_knowledge_graph = None
yolov8_seg_model = None # YOLOv8 分割模型
yolov8_seg_names = None # YOLOv8 分割模型类别名称

current_patient_llm_tokenizer = None
current_patient_llm_model = None
current_patient_llm_name = None # 记录当前激活的模型名称

device = "cpu" # 默认CPU，将在init_ai_modules中根据GPU情况更新

# 定义量化配置 (使用 4-bit 量化)
quantization_config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

PATIENT_LLM_MODELS_MAP = {
    "minicpm": "/hy-tmp/Lingshu-7B/MiniCPM-2B-dpo-fp16",
    "tinyllama": "/hy-tmp/Lingshu-7B/TinyLlama-1.1B-Chat"
}

def unload_patient_llm_model():
    global current_patient_llm_tokenizer, current_patient_llm_model, current_patient_llm_name
    if current_patient_llm_model is not None:
        print(f"正在卸载病人端模型: {current_patient_llm_name}...")
        del current_patient_llm_model
        del current_patient_llm_tokenizer
        torch.cuda.empty_cache() # 清理GPU显存
        current_patient_llm_model = None
        current_patient_llm_tokenizer = None
        current_patient_llm_name = None
        print("病人端模型已卸载，GPU显存已清理。")

def load_patient_llm_model(model_name: str):
    global current_patient_llm_tokenizer, current_patient_llm_model, current_patient_llm_name, device

    model_path = PATIENT_LLM_MODELS_MAP.get(model_name)
    if not model_path:
        print(f"错误: 未知的病人端模型名称 '{model_name}'。")
        return False

    if not os.path.exists(model_path):
        print(f"错误: 病人端模型路径 '{model_path}' 不存在。请检查模型是否已下载。")
        return False

    if current_patient_llm_name == model_name:
        print(f"病人端模型 '{model_name}' 已是当前激活模型，无需切换。")
        return True

    unload_patient_llm_model()

    print(f"正在加载病人端模型 '{model_name}' 从 '{model_path}' (4-bit 量化到GPU)...")
    try:
        current_patient_llm_tokenizer = AutoTokenizer.from_pretrained(model_path)
        current_patient_llm_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            quantization_config=quantization_config_4bit,
            trust_remote_code=True # 统一加上，兼容 MiniCPM
        )
        current_patient_llm_model.eval()
        current_patient_llm_name = model_name
        print(f"病人端模型 '{model_name}' 加载成功！")
        return True
    except Exception as e:
        print(f"加载病人端模型 '{model_name}' 时发生错误: {e}")
        import traceback
        traceback.print_exc()
        current_patient_llm_model = None
        current_patient_llm_tokenizer = None
        current_patient_llm_name = None
        return False


def init_ai_modules():
    """
    初始化所有AI模型、分词器、嵌入模型和向量数据库。
    此函数应在Flask应用启动时调用一次。
    """
    global tokenizer_qwen, model_qwen_gpu, yolov8_seg_model, yolov8_seg_names, \
           embeddings, retriever, medical_knowledge_graph, device

    print("--- 正在初始化所有AI模块 (全局加载一次) ---")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"检测到设备: {device}")

    # --- 1. Qwen 模型加载 (医生端专业报告生成) ---
    qwen_model_path = "/hy-tmp/Lingshu-7B/Qwen1.5-1.8B-Chat"
    print(f"正在加载 Qwen1.5-1.8B-Chat 分词器和模型 (4-bit 量化到GPU): {qwen_model_path}...")
    tokenizer_qwen = AutoTokenizer.from_pretrained(qwen_model_path)
    model_qwen_gpu = AutoModelForCausalLM.from_pretrained(
        qwen_model_path,
        device_map="auto",
        quantization_config=quantization_config_4bit
    )
    model_qwen_gpu.eval()
    print("Qwen1.5-1.8B-Chat 模型和分词器加载成功 (4-bit 量化到GPU)！")

    # --- 2. RAG 模块初始化 (嵌入模型强制加载到CPU) ---
    print("\n--- RAG 模块初始化 ---")
    knowledge_base_dir = "/hy-tmp/Lingshu-7B/medical_knowledge_base"
    documents = []
    if not os.path.exists(knowledge_base_dir):
        print(f"错误: 知识库目录 {knowledge_base_dir} 不存在。请先创建并放置医学知识文件。")
    else:
        for file_name in os.listdir(knowledge_base_dir):
            if file_name.endswith(".txt"):
                file_path = os.path.join(knowledge_base_dir, file_name)
                loader = TextLoader(file_path, encoding='utf-8')
                documents.extend(loader.load())
        print(f"加载了 {len(documents)} 篇医学知识文档。")

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    print(f"文档分割后得到 {len(docs)} 个文本块。")

    embedding_model_name = "moka-ai/m3e-base"
    local_embedding_model_path = "/hy-tmp/Lingshu-7B/embeddings/m3e-base"
    print(f"正在加载嵌入模型: {local_embedding_model_path}...")
    embeddings = HuggingFaceBgeEmbeddings(model_name=local_embedding_model_path,
                                          model_kwargs={'device': 'cpu'}) # 嵌入模型强制CPU
    print("嵌入模型加载成功 (已强制加载到CPU)！")

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

    retriever = db.as_retriever(search_kwargs={"k": 3})
    print("RAG 模块初始化完成。")

    # --- 3. 知识图谱模块初始化 (肺部专科) ---
    print("\n--- 知识图谱模块初始化 (肺部专科) ---")
    medical_knowledge_graph = {
        # --- 肺部疾病实体 ---
        "COVID-19": {
            "类型": "病毒感染",
            "影响器官": ["肺"],
            "病原体": "SARS-CoV-2",
            "常见症状": ["发热", "干咳", "乏力", "呼吸困难", "味觉嗅觉丧失", "肌痛", "咽痛"],
            "典型影像特征": ["磨玻璃影", "实变", "铺路石征", "多发斑片影", "胸膜下受累", "肺血管增粗"],
            "诊断方法": ["核酸检测", "胸部CT", "血常规", "抗体检测"],
            "治疗方法": ["抗病毒治疗", "氧疗", "对症支持治疗", "隔离"],
            "鉴别诊断": ["普通肺炎", "流感", "支气管炎"]
        },
        "病毒性肺炎": {
            "类型": "病毒感染",
            "影响器官": ["肺"],
            "病原体": ["流感病毒", "呼吸道合胞病毒", "腺病毒"],
            "常见症状": ["发热", "干咳", "乏力", "肌肉酸痛", "头痛"],
            "典型影像特征": ["肺间质浸润", "斑片状阴影", "磨玻璃影", "支气管壁增厚"],
            "诊断方法": ["病原学检测", "胸部X光片", "胸部CT", "血常规"],
            "治疗方法": ["抗病毒治疗", "对症支持治疗"]
        },
        "细菌性肺炎": {
            "类型": "细菌感染",
            "影响器官": ["肺"],
            "病原体": ["肺炎链球菌", "金黄色葡萄球菌", "肺炎克雷伯菌"],
            "常见症状": ["高热", "寒战", "咳脓痰", "胸痛", "呼吸急促"],
            "典型影像特征": ["肺叶实变", "支气管充气征", "胸腔积液", "肺脓肿"],
            "诊断方法": ["痰培养", "血培养", "胸部X光片", "胸部CT"],
            "治疗方法": ["抗生素治疗", "氧疗", "对症支持治疗"]
        },
        "肺部混浊区域": {
            "类型": "影像学发现",
            "可能原因": ["肺炎", "肺水肿", "肺肿瘤", "肺纤维化", "肺出血"],
            "影像特征": ["片状影", "模糊边界", "密度增高", "磨玻璃影", "实变"],
            "临床意义": "提示肺部存在病变，需进一步鉴别诊断"
        },
        "正常肺野": {
            "类型": "生理状态",
            "影像特征": ["清晰肺纹理", "无异常阴影", "肺门结构正常", "膈面光滑", "肋膈角锐利"],
            "临床意义": "肺部影像学检查未见明显异常"
        },
        "肺结节": {
            "类型": "影像学发现",
            "直径": "小于等于3厘米",
            "性质": ["良性", "恶性"],
            "常见症状": ["无症状", "咳嗽", "胸痛"],
            "典型影像特征": ["圆形阴影", "毛糙边缘", "分叶征", "胸膜凹陷征", "血管集束征", "钙化"],
            "诊断方法": ["胸部CT", "PET-CT", "穿刺活检", "手术切除"],
            "相关疾病": ["肺癌", "结核", "炎性假瘤", "错构瘤"]
        },

        # --- 症状实体 ---
        "发热": {"相关疾病": ["COVID-19", "病毒性肺炎", "细菌性肺炎", "流感"]},
        "干咳": {"相关疾病": ["COVID-19", "病毒性肺炎", "支气管炎", "过敏性咳嗽"]},
        "咳痰": {"相关疾病": ["细菌性肺炎", "病毒性肺炎", "支气管炎"], "特征": ["脓痰", "铁锈色痰"]},
        "呼吸困难": {"相关疾病": ["COVID-19", "肺炎", "哮喘", "心力衰竭"]},
        "乏力": {"相关疾病": ["COVID-19", "病毒性肺炎", "流感"]},
        "胸痛": {"相关疾病": ["COVID-19", "肺炎", "胸膜炎", "心绞痛"]},
        "味觉嗅觉丧失": {"相关疾病": ["COVID-19"], "特异性": "较高"},
        "肌肉酸痛": {"相关疾病": ["病毒性肺炎", "流感", "COVID-19"]},
        "咽痛": {"相关疾病": ["COVID-19", "感冒", "流感"]},
        "高热": {"相关疾病": ["细菌性肺炎"], "特征": "体温高于39℃"},
        "寒战": {"相关疾病": ["细菌性肺炎"], "伴随症状": "高热"},
        "咳脓痰": {"相关疾病": ["细菌性肺炎"], "特征": "痰液呈黄色或绿色"},

        # --- 影像学特征实体 ---
        "磨玻璃影": {"相关疾病": ["COVID-19", "病毒性肺炎", "肺水肿"], "特征": "肺部密度轻度增高，但血管和支气管影仍可见"},
        "实变": {"相关疾病": ["COVID-19", "肺炎", "肺肿瘤"], "特征": "肺泡腔被渗出物填充，密度均匀增高，血管和支气管影模糊或不可见"},
        "铺路石征": {"相关疾病": ["COVID-19", "ARDS"], "特征": "磨玻璃影中叠加网格状或小叶间隔增厚"},
        "多发斑片影": {"相关疾病": ["COVID-19", "肺炎", "肺结核"], "特征": "多个不规则的斑片状阴影"},
        "胸膜下受累": {"相关疾病": ["COVID-19"], "特征": "病变靠近胸膜"},
        "肺血管增粗": {"相关疾病": ["COVID-19", "肺动脉高压"], "特征": "肺部血管直径增加"},
        "肺间质浸润": {"相关疾病": ["病毒性肺炎", "间质性肺炎"], "特征": "肺间质炎症，肺纹理模糊或增粗"},
        "支气管壁增厚": {"相关疾病": ["支气管炎", "肺炎"], "特征": "支气管管壁增厚"},
        "肺叶实变": {"相关疾病": ["细菌性肺炎"], "特征": "整个肺叶或大部分肺叶密度增高"},
        "支气管充气征": {"相关疾病": ["实变", "肺炎"], "特征": "实变区内可见充气的支气管影"},
        "胸腔积液": {"相关疾病": ["肺炎", "心力衰竭", "肺癌"], "特征": "胸膜腔内液体积聚"},
        "肺脓肿": {"相关疾病": ["细菌性肺炎"], "特征": "肺部坏死性病变，形成含液体的空腔"},
        "清晰肺纹理": {"相关疾病": ["正常肺部"], "特征": "肺部血管和支气管影清晰"},
        "无异常阴影": {"相关疾病": ["正常肺部"], "特征": "肺部未见病变"},
        "肺门结构正常": {"相关疾病": ["正常肺部"], "特征": "肺门淋巴结无肿大，血管走行正常"},
        "膈面光滑": {"相关疾病": ["正常肺部"], "特征": "膈肌表面平滑"},
        "肋膈角锐利": {"相关疾病": ["正常肺部"], "特征": "肋骨与膈肌形成的夹角清晰锐利"},

        # --- 诊断方法实体 ---
        "核酸检测": {"用于诊断": ["COVID-19"], "特异性": "高"},
        "胸部CT": {"用于诊断": ["COVID-19", "肺炎", "肺结节", "骨折"], "优势": "提供详细三维图像"},
        "胸部X光片": {"用于诊断": ["肺炎", "骨折"], "优势": "快速、经济"},
        "血常规": {"用于诊断": ["肺炎", "感染"], "指标": ["白细胞", "C反应蛋白"]},
        "抗体检测": {"用于诊断": ["COVID-19"], "优势": "可判断既往感染"},
        "病原学检测": {"用于诊断": ["病毒性肺炎", "细菌性肺炎"], "方法": ["病毒分离", "核酸检测"]},
        "痰培养": {"用于诊断": ["细菌性肺炎"], "优势": "确定致病菌及药敏"},
        "血培养": {"用于诊断": ["细菌性肺炎"], "优势": "确定全身感染"},
        "PET-CT": {"用于诊断": ["肺结节", "肿瘤"], "优势": "评估代谢活性，鉴别良恶性"},
        "穿刺活检": {"用于诊断": ["肺结节", "肿瘤"], "优势": "组织病理学诊断金标准"},

        # --- 治疗方法实体 ---
        "抗病毒治疗": {"用于治疗": ["COVID-19", "病毒性肺炎"]},
        "氧疗": {"用于治疗": ["COVID-19", "肺炎", "呼吸困难"]},
        "对症支持治疗": {"用于治疗": ["COVID-19", "肺炎", "流感"], "包括": ["退热", "止咳", "化痰"]},
        "隔离": {"用于治疗": ["COVID-19"], "目的": "防止传播"},
        "抗生素治疗": {"用于治疗": ["细菌性肺炎"], "目的": "清除细菌感染"}
    }
    print("知识图谱模块初始化完成 (肺部专科)！")

    # --- 4. YOLOv8 分割模型加载 ---
    global yolov8_seg_model, yolov8_seg_names
    finetuned_yolov8_path = '/hy-tmp/Lingshu-7B/yolov8_finetuned_seg.pt'
    
    print(f"\n调试信息: 检查 YOLOv8 微调模型路径: {finetuned_yolov8_path}")
    if not os.path.exists(finetuned_yolov8_path):
        print(f"致命错误: YOLOv8 微调模型文件不存在于 {finetuned_yolov8_path}。请确认文件路径和名称。")
        yolov8_seg_model = None
        yolov8_seg_names = {0: "AI影像检测失败"} # 提供一个默认类别
    else:
        try:
            if yolov8_seg_model is None: # 只有当模型未加载时才尝试加载
                print(f"正在加载微调后的 YOLOv8 分割模型: {finetuned_yolov8_path}...")
                yolov8_seg_model = YOLO(finetuned_yolov8_path)
                yolov8_seg_model.eval() # 设置为评估模式
            
            data_yaml_path = '/hy-tmp/Lingshu-7B/covid_segmentation.yaml'
            print(f"调试信息: 尝试读取 YOLOv8 data.yaml: {data_yaml_path}")
            if not os.path.exists(data_yaml_path):
                print(f"致命错误: YOLOv8 data.yaml 文件不存在于 {data_yaml_path}。")
                yolov8_seg_model = None
                yolov8_seg_names = {0: "AI影像检测失败"}

            with open(data_yaml_path, 'r', encoding='utf-8') as f:
                data_config = yaml.safe_load(f)
            yolov8_seg_names = data_config.get('names', {0: "未知类别"}) # 使用.get避免keyError
            print(f"YOLOv8 分割模型加载成功，类别: {yolov8_seg_names}")
        except Exception as e:
            print(f"致命错误: 加载 YOLOv8 分割模型或读取 data.yaml 时发生异常: {e}")
            import traceback
            traceback.print_exc() # 打印完整错误堆栈
            yolov8_seg_model = None
            yolov8_seg_names = {0: "AI影像检测失败"}

    # 默认加载 MiniCPM 作为病人端模型
    print("\n--- 正在初始化病人端默认模型 (MiniCPM-2B-dpo-fp16) ---")
    if not load_patient_llm_model("minicpm"):
        print("错误: 默认病人端模型 MiniCPM 加载失败！请检查模型路径和配置。")
    print("所有AI模块初始化完毕。")


# 知识图谱校验函数 (已大幅修改)
def validate_with_kg(report_text, patient_info, detected_objects_info, rag_context, kg):
    validation_results = []
    
    # 1. 提取报告中的初步诊断
    report_diagnoses_raw = re.findall(r'初步诊断[:：\s]*(.*?)(?=\n|$)', report_text, re.DOTALL)
    report_diagnoses_keywords = []
    for diag_text in report_diagnoses_raw:
        for entity_name in kg.keys():
            if entity_name in diag_text:
                report_diagnoses_keywords.append(entity_name)
    report_diagnoses_keywords = list(set(report_diagnoses_keywords)) # 去重

    # 2. 提取AI检测到的主要病灶类别
    ai_detected_classes = [obj['class_name'] for obj in detected_objects_info if obj['confidence'] > 0.5 and obj['class_name'] != "未检测到明显病灶"]
    ai_detected_classes = list(set(ai_detected_classes)) # 去重

    # 3. 校验诊断与AI检测结果的一致性 (核心校验)
    if not report_diagnoses_keywords and ai_detected_classes: # 如果报告没诊断，但AI有发现
        validation_results.append("⚠️ 警告：AI检测到病灶，但报告未给出明确初步诊断。")
    elif report_diagnoses_keywords and not ai_detected_classes: # 如果报告有诊断，但AI没发现
        validation_results.append("❌ 警告：报告给出初步诊断，但AI影像检测未发现相关病灶，可能为幻觉。")
    
    for diagnosis in report_diagnoses_keywords:
        is_ai_supported = False
        if diagnosis in ai_detected_classes:
            is_ai_supported = True
        elif diagnosis == "肺炎" and any(c in ai_detected_classes for c in ["COVID", "Viral Pneumonia", "Lung_Opacity"]):
            is_ai_supported = True
        elif diagnosis == "肺部混浊区域" and any(c in ai_detected_classes for c in ["COVID", "Viral Pneumonia", "Lung_Opacity"]):
            is_ai_supported = True
        elif diagnosis == "正常肺野" and "Normal" in ai_detected_classes and len(ai_detected_classes) == 1:
            is_ai_supported = True
        
        if is_ai_supported:
            validation_results.append(f"✅ 报告诊断 '{diagnosis}' 与AI影像检测结果存在合理关联。")
        else:
            # 尝试在RAG或KG中寻找支持，作为次要验证
            found_support_in_rag_kg = False
            if rag_context and diagnosis in rag_context:
                found_support_in_rag_kg = True
            if diagnosis in kg and ("类型" in kg[diagnosis] or "临床意义" in kg[diagnosis]):
                found_support_in_rag_kg = True
            
            if not found_support_in_rag_kg:
                validation_results.append(f"❌ 警告：报告诊断 '{diagnosis}' 未被AI影像检测支持，且在RAG和知识图谱中未找到明确支持，**极可能存在幻觉**。")
            else:
                validation_results.append(f"⚠️ 注意：报告诊断 '{diagnosis}' 未被AI影像检测直接支持，但RAG或知识图谱提供了相关信息。请医生谨慎核实。")


    # 4. 校验报告中提及的症状是否在患者临床症状中提及或RAG/KG中有明确关联
    patient_symptoms_text = patient_info.get('clinical_symptoms', '')
    for entity, data in kg.items():
        if "常见症状" in data:
            for symptom in data["常见症状"]:
                if symptom in report_text and symptom not in patient_symptoms_text and symptom not in rag_context:
                    validation_results.append(f"⚠️ 报告提及症状 '{symptom}' (与 {entity} 相关)，但未在患者临床症状或RAG中明确提及。")
        elif "症状" in data and entity in report_text:
            for symptom_keyword in data["症状"]:
                if symptom_keyword in report_text and symptom_keyword not in patient_symptoms_text and symptom_keyword not in rag_context:
                    validation_results.append(f"⚠️ 报告提及症状 '{symptom_keyword}' (与 {entity} 相关)，但未在患者临床症状或RAG中明确提及。")


    # 5. 校验报告中提及的影像特征是否与AI检测结果或RAG/KG一致
    report_image_features = []
    for entity, data in kg.items():
        if "典型影像特征" in data:
            for feature in data["典型影像特征"]:
                if feature in report_text:
                    report_image_features.append(feature)
        elif "影像特征" in data:
            for feature in data["影像特征"]:
                if feature in report_text:
                    report_image_features.append(feature)
    report_image_features = list(set(report_image_features))

    for feature in report_image_features:
        is_ai_related = False
        for ai_class in ai_detected_classes:
            if ai_class in kg and ("典型影像特征" in kg[ai_class] and feature in kg[ai_class]["典型影像特征"] or \
                                    "影像特征" in kg[ai_class] and feature in kg[ai_class]["影像特征"]):
                is_ai_related = True
                break
        
        is_rag_supported = (rag_context and feature in rag_context)

        if not is_ai_related and not is_rag_supported:
            validation_results.append(f"⚠️ 报告提及影像特征 '{feature}'，但未被AI检测直接关联，也未在RAG中明确提及。")
        else:
            validation_results.append(f"✅ 报告提及影像特征 '{feature}'，与AI检测或RAG/KG存在关联。")


    # 6. 检查报告中是否存在未经证实的事实或编造信息 (例如日期、地点、核酸结果等)
    # 检查日期
    if re.search(r'\d{4}年\d{1,2}月\d{1,2}日', report_text):
        if not patient_info.get('exam_date') and not re.search(r'\d{4}年\d{1,2}月\d{1,2}日', rag_context):
            validation_results.append("❌ 警告：报告提及具体日期，但Prompt中未提供相关信息，可能为幻觉。")
    
    # 检查地点
    if re.search(r'(医院|诊所|北京市|上海市|广州市|深圳市)', report_text):
        if not patient_info.get('exam_location') and not re.search(r'(医院|诊所|北京市|上海市|广州市|深圳市)', rag_context):
            validation_results.append("❌ 警告：报告提及具体地点，但Prompt中未提供相关信息，可能为幻觉。")

    # 检查核酸检测结果
    if "核酸检测结果" in report_text or "核酸检测证实" in report_text:
        if "核酸检测" not in patient_info.get('clinical_symptoms', '') and "核酸检测" not in rag_context:
            validation_results.append("❌ 警告：报告提及'核酸检测结果'，但Prompt中未提供相关信息，可能为幻觉。")
    
    # 检查其他明显编造的专业术语或细节
    if patient_info.get('exam_type') == '胸部X光片' and re.search(r'(T1WI|T2WI|FLAIR|DWI)', report_text, re.IGNORECASE):
        validation_results.append("❌ 警告：检查类型为X光片，但报告提及MRI术语，可能为幻觉。")
    
    if re.search(r'（图[A-Z]）', report_text) and not patient_info.get('has_multiple_images_for_report'):
        validation_results.append("❌ 警告：报告提及图示标记（如'图A'），但Prompt中未提供多图信息，可能为幻觉。")


    # 最终总结
    final_status = "✅ 知识图谱校验：未检测到明显错误或幻觉。"
    if any("❌" in r for r in validation_results):
        final_status = "❌ 知识图谱校验：发现严重错误或幻觉，请医生**立即核实**！"
    elif any("⚠️" in r for r in validation_results):
        final_status = "⚠️ 知识图谱校验：发现潜在问题，请医生仔细核实。"
    
    validation_results.insert(0, final_status)

    return "\n".join(validation_results)


def run_yolov8_segmentation(image_path):
    """
    使用微调后的 YOLOv8 分割模型对医学影像进行病灶检测和分割。
    """
    global yolov8_seg_model, yolov8_seg_names

    if yolov8_seg_model is None:
        print("错误: YOLOv8 分割模型未加载。请检查模型路径和初始化。")
        return [{"class_name": "AI影像检测失败", "confidence": 0.0, "bbox": [0,0,0,0], "mask_coords": []}], None

    print(f"\n正在使用 YOLOv8 分割模型对图片 {image_path} 进行检测和分割...")

    if not os.path.exists(image_path):
        print(f"调试信息: 图片文件不存在于路径: {image_path}")
        return [{"class_name": "图片文件不存在", "confidence": 0.0, "bbox": [0,0,0,0], "mask_coords": []}], None
    
    results = yolov8_seg_model(image_path, imgsz=640, conf=0.25, iou=0.7, verbose=True)

    detected_objects_info = []
    annotated_image_filename = None

    if results and results[0].boxes.data.numel() > 0:
        r = results[0]
        print(f"调试信息: YOLOv8 原始检测结果数量: {len(r.boxes.data)}")
        for i, box in enumerate(r.boxes):
            class_id = int(box.cls)
            class_name = yolov8_seg_names.get(class_id, f"未知类别ID:{class_id}")
            confidence = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            mask_coords = []
            if r.masks is not None and i < len(r.masks.xy):
                mask_coords = r.masks.xy[i].tolist()

            detected_objects_info.append({
                "class_name": class_name,
                "confidence": confidence,
                "bbox": [x1, y1, x2, y2],
                "mask_coords": mask_coords
            })
        print(f"YOLOv8 分割模型检测到 {len(detected_objects_info)} 个病灶。")

        annotated_img_array = r.plot()
        
        original_base_filename = os.path.splitext(os.path.basename(image_path))[0]
        annotated_image_filename = f"annotated_{original_base_filename}_{str(uuid.uuid4())[:8]}.jpg"
        
        save_dir = os.path.dirname(image_path) 
        annotated_image_filepath = os.path.join(save_dir, annotated_image_filename)
        
        cv2.imwrite(annotated_image_filepath, annotated_img_array)
        print(f"带有标注的图片已保存到: {annotated_image_filepath}")

    else:
        print("YOLOv8 分割模型未检测到任何病灶 (可能置信度过低或无物体)。")
        detected_objects_info = [{"class_name": "未检测到明显病灶", "confidence": 1.0, "bbox": [0,0,0,0], "mask_coords": []}]
        annotated_image_filename = None
    
    return detected_objects_info, annotated_image_filename


def generate_professional_report(patient_info, detected_objects_info):
    """
    使用Qwen模型生成专业报告草稿，结合RAG和知识图谱。
    """
    global model_qwen_gpu

    if not all([tokenizer_qwen, model_qwen_gpu, retriever, medical_knowledge_graph]):
        raise RuntimeError("AI模块未初始化，请先调用 init_ai_modules()")

    yolov8_text_description = ""
    main_detected_classes_for_prompt = []
    if detected_objects_info:
        yolov8_text_description += "影像分析结果（AI辅助检测）：\n"
        for i, obj in enumerate(detected_objects_info):
            segment_info = ""
            if obj['mask_coords']:
                segment_info = f"并伴有像素级分割掩码，覆盖区域为[{obj['bbox'][0]},{obj['bbox'][1]}]到[{obj['bbox'][2]},{obj['bbox'][3]}]"
            
            yolov8_text_description += (
                f"  {i+1}. 检测到 '{obj['class_name']}'，置信度 {obj['confidence']:.2f}，"
                f"位于图像区域 {obj['bbox']}。{segment_info}\n"
            )
            if obj['confidence'] > 0.5 and obj['class_name'] != "未检测到明显病灶":
                main_detected_classes_for_prompt.append(obj['class_name'])
    else:
        yolov8_text_description += "影像分析结果（AI辅助检测）：未检测到明显异常。\n"

    query_text = (
        f"患者症状: {patient_info['clinical_symptoms']}。 "
        f"影像AI检测到主要发现: {', '.join(main_detected_classes_for_prompt) if main_detected_classes_for_prompt else '无明确病灶'}。"
        "请提供与这些AI发现和症状相关的肺部医学知识，特别是关于诊断、影像特征和鉴别诊断。"
    )
    print(f"\nRAG 检索查询: {query_text}")
    retrieved_docs_list = retriever.invoke(query_text)
    
    rag_context = "\n\n相关医学知识（来自RAG）：\n"
    if retrieved_docs_list:
        for i, doc in enumerate(retrieved_docs_list):
            rag_context += f"--- 文档 {i+1} ---\n{doc.page_content}\n"
    else:
        rag_context += "未检索到相关医学知识。\n"
    print(rag_context)

    # 构建 Prompt (Qwen) - 极致强化幻觉抑制
    messages = [
        {"role": "system", "content": (
            "你是一名经验丰富的放射科医生，擅长解读肺部影像并撰写专业的诊断报告。\n"
            "你的任务是根据患者的临床信息、AI辅助影像分析结果和提供的最新肺部医学知识，生成一份严谨、准确的初步诊断报告草稿。\n"
            "**核心原则：**\n"
            "1. **严格依据事实：** 报告内容必须完全基于以下三类信息：\n"
            "   - 患者提供的**临床症状**。\n"
            "   - AI辅助影像检测到的**具体影像发现**（病灶类别、置信度、位置、边界框、分割掩码）。\n"
            "   - RAG检索到的**相关医学知识**（仅作为解释AI发现和症状的参考）。\n"
            "2. **严禁幻觉与编造：**\n"
            "   - **严禁提及任何AI影像检测结果中未明确报告的疾病或影像特征（如磨玻璃影、实变、肺纹理增多、肺门增大、具体解剖位置、尺寸等），除非这些信息在RAG中被明确提及并与AI检测结果直接关联。**\n"
            "   - **严禁编造任何未经Prompt（包括患者信息、AI检测、RAG上下文）明确提供的日期、地点、检查结果（如核酸检测结果）、医院名称、图示标记（如图A/B/C）、MRI术语等具体事实。**\n"
            "   - **严禁进行过度推断或引入与AI检测结果不符的诊断。**\n"
            "3. **报告结构：** 报告必须清晰地包含'影像所见'、'初步诊断'和'建议'三个部分，并确保每个部分完整且内容聚焦。\n"
            "4. **无明确病灶处理：** 如果AI影像检测结果中未检测到明确病灶，则初步诊断应倾向于排除重大影像学异常，并建议结合临床进一步检查，而非编造诊断。\n"
            "5. **置信度引用：** 在提及AI检测结果时，可以引用其置信度，但不要将其解释为“RAG确认率”或进行其他误导性解读。\n"
            "请严格遵守以上原则，生成专业的初步诊断报告草稿。不要重复提示词"
        )},
        {"role": "user", "content": (
            f"患者ID: {patient_info['patient_id']}\n"
            f"年龄: {patient_info['age']}岁\n"
            f"性别: {patient_info['gender']}\n"
            f"检查类型: {patient_info['exam_type']}\n"
            f"临床症状: {patient_info['clinical_symptoms']}\n\n"
            f"{yolov8_text_description}\n"
            f"{rag_context}\n" # RAG检索到的知识
            "请根据以上所有信息，**严格遵循核心原则，重点围绕AI检测到的影像发现**，生成一份专业的初步诊断报告草稿。报告应包含'影像所见'、'初步诊断'和'建议'三个部分。"
        )}
    ]

    text = tokenizer_qwen.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    print("\n--- 构建的 LLM Prompt (Qwen) ---")
    print(text)

    model_inputs = tokenizer_qwen(text, return_tensors="pt").to(device)
    
    print("\n正在生成专业报告草稿 (Qwen1.5-1.8B-Chat)...")

    generated_ids = None
    try:
        generated_ids = model_qwen_gpu.generate(
            model_inputs.input_ids,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer_qwen.eos_token_id
        )
    except Exception as e:
        print(f"错误: Qwen 模型生成文本时发生异常: {e}")
        import traceback
        traceback.print_exc()
        return "Qwen 模型生成专业报告失败，请联系技术支持。", "校验失败：模型生成异常。"

    if generated_ids is None or generated_ids.numel() == 0:
        print("错误: Qwen 模型未生成任何有效的 token ID。")
        return "Qwen 模型未能生成专业报告，可能输入过长或模型状态异常。", "校验失败：模型未生成内容。"

    # 关键修改：基于输入 token ID 长度进行解码
    input_token_len = model_inputs.input_ids.shape[1]
    generated_new_tokens_ids = generated_ids[0, input_token_len:]
    
    report_draft = tokenizer_qwen.decode(generated_new_tokens_ids, skip_special_tokens=True).strip()

    print("\n--- 生成的专业报告草稿 (RAG增强版) ---")
    print(report_draft)

    kg_validation_output = validate_with_kg(
        report_text=report_draft,
        patient_info=patient_info,
        detected_objects_info=detected_objects_info,
        rag_context=rag_context, # 传递RAG原始文本
        kg=medical_knowledge_graph
    )
    print("\n--- 知识图谱初步校验结果 ---")
    print(kg_validation_output)

    return report_draft, kg_validation_output


def generate_cop_report(professional_report_content):
    """
    使用当前激活的病人端LLM模型生成患者科普报告。
    """
    global current_patient_llm_tokenizer, current_patient_llm_model, current_patient_llm_name

    if not all([current_patient_llm_tokenizer, current_patient_llm_model]):
        print("错误: 病人端LLM模型未加载或未激活，无法生成科普报告。")
        return "病人端LLM模型未加载，无法生成科普报告，请联系技术支持。"

    # Few-shot 示例
    few_shot_example_professional = """
    **影像所见**
    根据AI辅助检测结果，在右肺下叶检测到 'COVID' 病灶，置信度 0.95，位于图像区域 [100, 200, 300, 400]，并伴有像素级分割掩码。
    **初步诊断**
    根据AI影像检测结果和患者临床症状（干咳、乏力），初步判断为新冠病毒肺炎。
    **建议**
    建议进行核酸检测以确诊，并结合临床症状进行对症支持治疗。
    """

    few_shot_example_cop = """
    **您的科普报告**
    亲爱的患者朋友，

    根据您最近的检查，AI辅助影像分析显示您的右肺下叶有一个**新冠病毒感染的迹象**。结合您目前的干咳、乏力症状，医生初步判断您可能患有**新冠病毒肺炎**。

    **下一步建议：**
    我们建议您尽快进行**核酸检测**以确认诊断。同时，请您根据医生的指导，进行适当的休息和对症治疗，例如多喝水、保持舒适等。

    请记住，这份报告是基于AI辅助分析的初步结果，最终诊断和治疗方案请务必咨询您的主治医生。
    """

    messages_patient_llm = [
        {"role": "system", "content": (
            "你是一名专业的医疗科普作者，擅长将复杂的医学报告转化为通俗易懂、面向患者的科普报告。\n"
            "**核心原则：**\n"
            "1. **忠实原文，严禁编造：** 你的任务是**严格忠实地翻译和总结**提供的专业诊断报告，**严禁引入任何新的医学信息、诊断、症状、疾病（如HIV、肺结核、肺动脉高压、肺部空洞、肿瘤等）或未经原文提及的细节。**\n"
            "2. **通俗易懂：** 将专业术语转化为日常语言，解释病情、可能的治疗方案和注意事项。\n"
            "3. **语气友好：** 保持对患者友好的语气，避免使用晦涩的医学词汇。\n"
            "4. **结构清晰：** 报告应有清晰的标题和分段，便于患者阅读。\n"
            "请严格遵守以上原则，为患者撰写一份简洁明了、易于理解的科普报告。不要重复提示词"
        )},
        # Few-shot 示例：专业报告
        {"role": "user", "content": f"以下是专业诊断报告：\n{few_shot_example_professional}\n\n请根据以上专业诊断报告，**严格遵循核心原则**，为患者撰写一份简洁明了、易于理解的科普报告："},
        # Few-shot 示例：科普报告
        {"role": "assistant", "content": few_shot_example_cop},
        # 实际的用户请求
        {"role": "user", "content": (
            "以下是专业诊断报告：\n"
            f"{professional_report_content}\n\n"
            "请根据以上专业诊断报告，**严格遵循核心原则**，为患者撰写一份简洁明了、易于理解的科普报告："
        )}
    ]

    text_patient_llm = current_patient_llm_tokenizer.apply_chat_template(
        messages_patient_llm,
        tokenize=False,
        add_generation_prompt=True
    )

    print(f"\n--- 构建的病人端LLM Prompt ({current_patient_llm_name}) ---")
    print(text_patient_llm)

    model_inputs_patient_llm = current_patient_llm_tokenizer(text_patient_llm, return_tensors="pt").to(device)

    print(f"\n正在生成患者科普报告 ({current_patient_llm_name})...")

    generated_ids = None
    try:
        generated_ids = current_patient_llm_model.generate(
            model_inputs_patient_llm.input_ids,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.01,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=current_patient_llm_tokenizer.eos_token_id
        )
    except Exception as e:
        print(f"错误: 病人端LLM模型 '{current_patient_llm_name}' 生成文本时发生异常: {e}")
        import traceback
        traceback.print_exc()
        return f"病人端LLM模型 '{current_patient_llm_name}' 生成科普报告失败，请联系技术支持。"

    if generated_ids is None or generated_ids.numel() == 0:
        print(f"错误: 病人端LLM模型 '{current_patient_llm_name}' 未生成任何有效的 token ID。")
        return f"病人端LLM模型 '{current_patient_llm_name}' 未能生成科普报告，可能输入过长或模型状态异常。"

    # 关键修改：基于输入 token ID 长度进行解码
    input_token_len_minicpm = model_inputs_patient_llm.input_ids.shape[1]
    generated_new_tokens_ids_minicpm = generated_ids[0, input_token_len_minicpm:]
    
    cop_report = current_patient_llm_tokenizer.decode(generated_new_tokens_ids_minicpm, skip_special_tokens=True).strip()

    print(f"\n--- 生成的患者科普报告 ({current_patient_llm_name}) ---")
    print(cop_report)

    return cop_report