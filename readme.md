### **`README.md`**

```markdown
# 智能肺部影像辅助诊断与科普平台

## 🚀 项目概述

本项目旨在构建一个**智能肺部影像辅助诊断与患者科普平台**。它利用前沿的AI技术，辅助医生对胸部X光片进行初步的**肺部病灶实例分割**，自动生成**专业的诊断报告草稿**。同时，系统还能将这份专业报告转化为**通俗易懂的患者科普报告**，从而提升医疗效率，优化医患沟通。

作为一个为求职量身定制的AI算法项目，它全面展示了在多模态AI、大型语言模型（LLM）应用、检索增强生成（RAG）、知识图谱、模型优化及全栈开发等方面的综合实力。

## ✨ 核心功能与亮点

1.  **多模态AI驱动的肺部影像分析**
    *   **YOLOv8n-seg 真实微调：** 集成了在 `COVID-19 Radiography Database` 上微调的 **YOLOv8n-seg** 模型，实现对胸部X光片中 **COVID-19、肺部混浊、病毒性肺炎**等病灶的**精确实例分割**。
    *   **影像可视化：** 医生端可直观查看原始上传影像和带有AI标注（边界框与分割掩码）的影像。

2.  **智能诊断报告与科普报告生成**
    *   **专业报告（医生端）：** 基于 **Qwen1.5-1.8B-Chat** (4-bit 量化) 模型，结合AI影像分析结果、患者临床信息及RAG检索的肺部专科知识，生成结构严谨、内容专业的初步诊断报告草稿。
    *   **科普报告（患者端）：** 基于 **MiniCPM-2B-dpo-fp16** (4-bit 量化) 模型，将专业报告转化为通俗易懂、语气友好的患者科普报告，解释病情、治疗方案和注意事项。

3.  **RAG与知识图谱增强，有效减少AI幻觉**
    *   **肺部专科RAG：** 构建了包含COVID-19、肺炎、肺结节等详细信息的**肺部专科知识库**，通过 **moka-ai/m3e-base** 嵌入模型和 **ChromaDB** 向量数据库，为LLM提供精准的上下文。
    *   **肺部专科知识图谱：** 深度扩展的知识图谱用于对LLM生成的报告进行**事实校验和逻辑引导**，显著提升报告的准确性和可靠性，**从源头抑制AI幻觉**。

4.  **多角色前后端交互与报告管理**
    *   **医生工作台：** 提供影像上传、报告列表（含图片预览）、查看详细报告、**软删除报告**等功能。
    *   **患者查询：** 患者可通过报告ID直接查询并**只查看**自己的科普报告。
    *   **技术信息页：** 展示项目技术栈和系统状态。
    *   **报告生命周期管理：** 报告支持软删除，并由**后台调度任务**自动清理超过一天的已删除报告及关联文件。

## ⚙️ 技术栈

*   **后端框架：** Flask
*   **前端框架：** Vue3 (使用 Vue Router 进行多页面管理)
*   **深度学习框架：** PyTorch
*   **计算机视觉模型：** YOLOv8n-seg (在 `COVID-19 Radiography Database` 上微调)
*   **大型语言模型：**
    *   Qwen1.5-1.8B-Chat (专业报告生成，4-bit 量化)
    *   MiniCPM-2B-dpo-fp16 (科普报告生成，4-bit 量化)
*   **检索增强生成 (RAG)：**
    *   嵌入模型：moka-ai/m3e-base (部署在CPU)
    *   向量数据库：ChromaDB
    *   LLM集成框架：LangChain
*   **知识图谱：** Python Dictionary (肺部专科知识)
*   **后台任务调度：** APScheduler
*   **其他：** Hugging Face Transformers, OpenCV, NumPy, werkzeug

## 📂 项目目录结构

```
.
├── chroma_db/                         # ChromaDB 向量数据库持久化目录
├── COVID-19_Radiography_Dataset/      # 原始数据集，用于YOLOv8微调 (包含 images/ 和 masks/ 子目录)
├── embeddings/                        # moka-ai/m3e-base 嵌入模型的本地存储
├── medical_knowledge_base/            # RAG 知识库文本文件 (肺部专科)
├── minicpm_output/                    # MiniCPM 生成的科普报告输出目录
├── MiniCPM-2B-dpo-fp16/               # MiniCPM-2B-dpo-fp16 LLM 模型文件
├── qwen_output/                       # Qwen 生成的专业报告输出目录
├── Qwen1.5-1.8B-Chat/                 # Qwen1.5-1.8B-Chat LLM 模型文件
├── runs/                              # YOLOv8 训练日志和权重保存目录
├── uploads/                           # 用户上传的原始影像和AI标注影像的存储目录
├── yolov8_output/                     # (旧的YOLOv8模拟输出，可清理)
├── yolov8_segmentation_data/          # 经过预处理、划分为train/val/test的YOLOv8分割训练数据
├── ai_pipeline_modules.py             # 核心AI逻辑模块 (LLM、RAG、KG、YOLOv8推理)
├── app.py                             # Flask 后端应用，定义API接口和后台任务
├── covid_segmentation.yaml            # YOLOv8 分割模型训练的数据配置文件
├── prepare_yolov8_segmentation_data.py # 数据准备脚本：将原始数据集转换为YOLOv8分割格式
├── reports.json                       # 报告数据持久化文件 (JSON模拟数据库)
├── sample_image.jpg                   # 示例图片
├── train_yolov8_seg.py                # YOLOv8 分割模型微调训练脚本
├── yolov8_finetuned_seg.pt            # 微调后的YOLOv8分割模型权重
└── yolov8n-seg.pt                     # YOLOv8n-seg 预训练模型权重
```

## 🚀 快速启动

### **1. 环境准备 (Linux 服务器)**

1.  **克隆项目仓库 (如果适用):**
    ```bash
    git clone <your-repo-url>
    cd <project-root-directory> # 例如 /hy-tmp/Lingshu-7B
    ```
2.  **创建并激活Python虚拟环境 (推荐):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **安装Python依赖:**
    ```bash
    pip install -r requirements.txt # 如果有requirements.txt文件
    # 或者手动安装核心依赖 (请确保PyTorch版本与CUDA兼容)
    pip install torch==2.6.0+cu124 torchvision==0.19.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
    pip install transformers accelerate bitsandbytes sentence-transformers langchain-community langchain-text-splitters chromadb ultralytics opencv-python apscheduler Flask Flask-Cors werkzeug pyyaml
    ```
    *   **注意：** `torch` 版本 `2.6.0+cu124` 对应 CUDA 12.4。请根据您的CUDA版本调整。
    *   `bitsandbytes` 可能需要特定CUDA版本支持，如果安装失败，请查阅其官方文档。

4.  **下载预训练模型:**
    *   **Qwen1.5-1.8B-Chat:**
        ```bash
        huggingface-cli download Qwen/Qwen1.5-1.8B-Chat --local-dir Qwen1.5-1.8B-Chat
        ```
    *   **MiniCPM-2B-dpo-fp16:**
        ```bash
        huggingface-cli download openbmb/MiniCPM-2B-dpo-fp16 --local-dir MiniCPM-2B-dpo-fp16
        ```
    *   **moka-ai/m3e-base (嵌入模型):**
        ```bash
        mkdir -p embeddings
        huggingface-cli download moka-ai/m3e-base --local-dir embeddings/m3e-base
        ```
    *   **YOLOv8n-seg 预训练模型:**
        ```bash
        wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-seg.pt
        ```
    *   **YOLOv8 辅助资产 (yolo11n.pt):**
        ```bash
        wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
        mkdir -p ~/.cache/ultralytics/assets/
        mv yolo11n.pt ~/.cache/ultralytics/assets/
        ```
    *   **COVID-19 Radiography Database:** 确保数据集已下载并放置在 `COVID-19_Radiography_Dataset/` 目录下。

### **2. 数据准备与YOLOv8微调 (Linux 服务器)**

1.  **准备YOLOv8分割训练数据:**
    ```bash
    python prepare_yolov8_segmentation_data.py
    ```
    *   此脚本将处理 `COVID-19_Radiography_Dataset/`，生成 `yolov8_segmentation_data/` 目录，包含训练所需的图片和标签。
2.  **创建YOLOv8数据配置文件:**
    确保 `covid_segmentation.yaml` 文件存在于项目根目录，内容如下：
    ```yaml
    # covid_segmentation.yaml
    path: /hy-tmp/Lingshu-7B/yolov8_segmentation_data
    train: train/images
    val: val/images
    test: test/images
    names:
      0: COVID
      1: Lung_Opacity
      2: Normal
      3: Viral Pneumonia
    ```
3.  **微调YOLOv8分割模型:**
    ```bash
    python train_yolov8_seg.py
    ```
    *   此步骤将使用您的GPU对YOLOv8n-seg模型进行微调。训练可能需要数小时。
    *   微调后的模型将保存为 `yolov8_finetuned_seg.pt`。

### **3. RAG知识库准备 (Linux 服务器)**

1.  **创建RAG知识库目录和文件:**
    ```bash
    mkdir -p medical_knowledge_base
    ```
    *   在 `medical_knowledge_base/` 目录下创建 `covid19_overview.txt`, `viral_pneumonia_details.txt`, `lung_opacity_causes.txt`, `normal_lung_anatomy.txt`, `lung_nodule_management.txt` 等肺部专科知识文件。
    *   确保 `reports.json` 文件存在 (初始为空)。
    *   确保 `uploads/` 目录存在。

### **4. 启动Flask后端 (Linux 服务器)**

1.  **确保 `ai_pipeline_modules.py` 和 `app.py` 是最新版本。**
2.  **启动Flask应用:**
    ```bash
    python app.py
    ```
    *   Flask应用将启动，并初始化所有AI模型和RAG模块。它将监听 `http://0.0.0.0:8080`。
    *   后台清理任务将启动，每小时检查并清理过期报告。

### **5. 前端设置与启动 (本地开发环境)**

1.  **安装Node.js和npm (或yarn)。**
2.  **创建Vue项目 (如果尚未创建):**
    ```bash
    npm create vite@latest my-ai-frontend -- --template vue
    cd my-ai-frontend
    npm install
    npm install axios vue-router@4
    ```
3.  **更新前端代码:**
    *   将 `src/App.vue`, `src/main.js`, `src/router/index.js` 和 `src/views/` 目录下的所有 `.vue` 文件更新为最新代码。
    *   **重要：** 确保所有前端文件中 `API_BASE_URL` 变量已替换为您的 **Flask 后端暴露网站地址** (例如 `http://i-2.gpushare.com:44953`)。
4.  **启动Vue前端:**
    ```bash
    npm run dev
    ```
    *   前端应用通常会运行在 `http://localhost:5173`。

## 🚀 使用指南

1.  **访问前端应用:** 在浏览器中打开前端应用的地址 (例如 `http://localhost:5173`)。
2.  **医生工作台:**
    *   在导航栏点击“医生工作台”。
    *   上传一张胸部X光片 (推荐使用 `COVID-19_Radiography_Dataset` 中的图片)。
    *   填写患者信息。
    *   点击“开始AI分析”，系统将进行影像检测、报告生成。
    *   下方报告列表将显示新生成的报告摘要，包含原始影像和AI标注影像的缩略图。
    *   点击“查看详细报告”可查看完整的专业报告、科普报告和AI检测详情。
    *   点击“删除报告”可将报告标记为删除，报告将从列表中移除。
3.  **患者查询:**
    *   在导航栏点击“患者查询”。
    *   输入报告ID (可从医生工作台获取)。
    *   点击“查询报告”，将只显示患者科普报告。
4.  **技术信息页:**
    *   在导航栏点击“技术信息”，查看项目技术栈和系统状态。

## 🌟 项目亮点与未来展望

请参阅 [最终项目总结报告](#最终项目总结报告：智能肺部影像辅助诊断与科普平台) 以获取详细的项目亮点、克服的挑战和未来规划。

---