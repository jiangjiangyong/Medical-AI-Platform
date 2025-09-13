这是我做的第一个全流医疗项目，我使用的配置是3060 12G，如果有疑问可以互相交流交流 1950532548@qq.com
### **`README.md`**

```markdown
# 智能肺部影像辅助诊断与科普平台

## 🚀 项目概述

本项目旨在构建一个**智能肺部影像辅助诊断与患者科普平台**。它利用前沿的AI技术，辅助医生对胸部X光片进行初步的**肺部病灶实例分割**，自动生成**专业的诊断报告草稿**。同时，系统还能将这份专业报告转化为**通俗易懂的患者科普报告**，从而提升医疗效率，优化医患沟通。

作为一个为求职量身定制的AI算法项目，它全面展示了在多模态AI、大型语言模型（LLM）应用、检索增强生成（RAG）、知识图谱、模型优化、MLOps实践及全栈开发等方面的综合实力。

## ✨ 核心功能与亮点

1.  **端到端多模态AI驱动的肺部影像分析**
    *   **YOLOv8n-seg 真实微调：** 集成了在 `COVID-19 Radiography Database` 上微调的 **YOLOv8n-seg** 模型，实现对胸部X光片中 **COVID-19、肺部混浊、病毒性肺炎**等病灶的**精确实例分割**。模型输出病灶的类别、置信度、边界框及像素级分割掩码。
    *   **影像可视化：** 医生工作台和报告详情页可直观查看原始上传影像和带有AI标注（边界框与分割掩码）的影像，增强医生对AI分析结果的信任。

2.  **智能诊断报告与患者科普报告生成**
    *   **专业报告（医生端）：** 基于 **Qwen1.5-1.8B-Chat** (4-bit 量化) 模型，结合AI影像分析结果、患者临床信息及RAG检索的肺部专科知识，生成结构严谨、内容专业的初步诊断报告草稿。
    *   **科普报告（患者端）：** 基于 **MiniCPM-2B-dpo-fp16** (4-bit 量化) 模型（默认激活，**可动态热切换至 TinyLlama-1.1B-Chat**），将专业报告转化为通俗易懂、语气友好的患者科普报告，解释病情、治疗方案和注意事项。

3.  **RAG与知识图谱深度增强，强力抑制AI幻觉**
    *   **肺部专科RAG：** 构建了包含COVID-19、肺炎、肺结节等详细信息的**肺部专科知识库**，通过 **moka-ai/m3e-base** 嵌入模型和 **ChromaDB** 向量数据库，为LLM提供精准的上下文。
    *   **肺部专科知识图谱：** 深度扩展的知识图谱用于对LLM生成的报告进行**事实校验和逻辑引导**。它能主动检测报告中的逻辑冲突（如诊断与AI发现不符）、识别未经Prompt提供的事实，并校验与检查类型不符的医学描述，**显著抑制AI幻觉，极大提升报告的准确性和可靠性**。

4.  **多角色前后端交互与报告生命周期管理**
    *   **医生工作台：** 提供影像上传、报告列表（含原始及AI标注图片预览）、查看详细报告、以及**软删除报告**（移入回收站）功能。
    *   **患者查询：** 患者可通过报告ID直接查询并**只查看**自己的科普报告和基本信息。
    *   **技术信息页：** 展示项目技术栈、AI模型配置、**当前激活的病人端LLM模型**，并提供**模型动态切换**功能。
    *   **报告生命周期管理：** 报告支持软删除，并由**后台调度任务**自动进行**过期报告及其关联图片文件的永久清理**，确保数据存储的效率和合规性。

5.  **生产级MLOps实践：轻量级性能监控与模型动态管理**
    *   **系统与AI性能监控：** 后端实时收集CPU、内存、GPU使用率，以及API和AI模块（YOLOv8、Qwen、病人端LLM）的推理延迟，并将数据**持久化到CSV文件**。前端技术信息页面通过 **Chart.js** 绘制**实时仪表盘和历史趋势图**，直观展示系统健康状况和AI性能。
    *   **模型动态热切换：** 实现了病人端LLM模型（MiniCPM与TinyLlama）在不中断Flask服务的情况下进行**动态加载和卸载**，优化了资源利用和模型灵活性。

## ⚙️ 技术栈

*   **后端框架：** Flask
*   **前端框架：** Vue3 (使用 Vue Router 进行多页面管理，Chart.js / vue-chartjs 进行数据可视化)
*   **深度学习框架：** PyTorch
*   **计算机视觉模型：** YOLOv8n-seg (在 `COVID-19 Radiography Database` 上微调，用于肺部病灶实例分割)
*   **大型语言模型 (LLM)：**
    *   Qwen1.5-1.8B-Chat (专业报告生成，4-bit 量化)
    *   MiniCPM-2B-dpo-fp16 (科普报告生成，4-bit 量化，默认激活)
    *   TinyLlama-1.1B-Chat (科普报告生成，4-bit 量化，可动态切换)
*   **检索增强生成 (RAG)：**
    *   嵌入模型：moka-ai/m3e-base (部署在CPU)
    *   向量数据库：ChromaDB
    *   LLM集成框架：LangChain
*   **知识图谱：** Python Dictionary (肺部专科知识，用于事实校验和逻辑引导)
*   **后台任务调度：** APScheduler
*   **系统监控：** psutil, subprocess (调用 `nvidia-smi`), csv (轻量级数据持久化)
*   **其他：** Hugging Face Transformers, OpenCV, NumPy, werkzeug

## 📂 项目目录结构

```
.
├── chroma_db/                         # ChromaDB 向量数据库持久化目录
├── COVID-19_Radiography_Dataset/      # 原始数据集，用于YOLOv8微调 (包含 images/ 和 masks/ 子目录)
├── embeddings/                        # moka-ai/m3e-base 嵌入模型的本地存储
├── medical_knowledge_base/            # RAG 知识库文本文件 (肺部专科)
├── metrics_history.csv                # 系统与AI性能监控数据的轻量级持久化文件
├── my-ai-frontend/                    # Vue3 前端项目目录
├── minicpm_output/                    # MiniCPM 生成的科普报告输出目录 (历史遗留，可清理)
├── MiniCPM-2B-dpo-fp16/               # MiniCPM-2B-dpo-fp16 LLM 模型文件
├── qwen_output/                       # Qwen 生成的专业报告输出目录 (历史遗留，可清理)
├── Qwen1.5-1.8B-Chat/                 # Qwen1.5-1.8B-Chat LLM 模型文件
├── reports.json                       # 报告数据持久化文件 (JSON模拟数据库)
├── runs/                              # YOLOv8 训练日志和权重保存目录
├── TinyLlama-1.1B-Chat/               # TinyLlama-1.1B-Chat LLM 模型文件
├── uploads/                           # 用户上传的原始影像和AI标注影像的存储目录
├── yolov8_output/                     # (旧的YOLOv8模拟输出，可清理)
├── yolov8_segmentation_data/          # 经过预处理、划分为train/val/test的YOLOv8分割训练数据
├── ai_pipeline_modules.py             # 核心AI逻辑模块 (LLM、RAG、KG、YOLOv8推理、模型切换)
├── app.py                             # Flask 后端应用，定义API接口、后台任务、性能指标暴露
├── covid_segmentation.yaml            # YOLOv8 分割模型训练的数据配置文件
├── download_tinyllama.py              # 下载 TinyLlama 模型的辅助脚本
├── prepare_yolov8_segmentation_data.py # 数据准备脚本：将原始数据集转换为YOLOv8分割格式
├── requirements.txt                   # Python 项目依赖列表
├── train_yolov8_seg.py                # YOLOv8 分割模型微调训练脚本
├── yolov8_finetuned_seg.pt            # 微调后的YOLOv8分割模型权重
└── yolov8n-seg.pt                     # YOLOv8n-seg 预训练模型权重
```

## 🚀 快速启动

### **1. 环境准备 (Linux 服务器)**

1.  **克隆项目仓库:**
    ```bash
    git clone https://github.com/jiangjiangyong/Medical-AI-Platform.git
    cd Medical-AI-Platform # 进入项目根目录，例如 /hy-tmp/Lingshu-7B
    ```
2.  **创建并激活Python虚拟环境 (推荐):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **安装Python依赖:**
    *   **首先安装PyTorch (请根据您的CUDA版本选择正确的`--index-url`):**
        ```bash
        # 示例：CUDA 12.1
        pip install torch==2.4.0+cu121 torchvision==0.19.0+cu121 torchaudio==2.4.0+cu121 --index-url https://download.pytorch.org/whl/cu121
        # 如果您的CUDA版本是12.4，请将cu121改为cu124
        # pip install torch==2.6.0+cu124 torchvision==0.19.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
        ```
    *   **然后安装其他所有依赖:**
        ```bash
        pip install -r requirements.txt
        ```
        *   **注意：** `bitsandbytes` 可能需要特定CUDA版本支持，如果安装失败，请查阅其官方文档。

4.  **下载预训练模型:**
    *   **Qwen1.5-1.8B-Chat:**
        ```bash
        huggingface-cli download Qwen/Qwen1.5-1.8B-Chat --local-dir Qwen1.5-1.8B-Chat
        ```
    *   **MiniCPM-2B-dpo-fp16:**
        ```bash
        huggingface-cli download openbmb/MiniCPM-2B-dpo-fp16 --local-dir MiniCPM-2B-dpo-fp16
        ```
    *   **TinyLlama-1.1B-Chat:**
        ```bash
        python download_tinyllama.py
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
    *   **COVID-19 Radiography Database:** 确保数据集已下载并放置在 `COVID-19_Radiography_Dataset/` 目录下，其内部结构应为 `类别/images/` 和 `类别/masks/`。

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
    *   此步骤将使用您的GPU对YOLOv8n-seg模型进行微调。训练可能需要数小时，微调后的模型将保存为 `yolov8_finetuned_seg.pt`。

### **3. RAG知识库与辅助文件准备 (Linux 服务器)**

1.  **创建RAG知识库目录和文件:**
    ```bash
    mkdir -p medical_knowledge_base
    ```
    *   在 `medical_knowledge_base/` 目录下创建 `covid19_overview.txt`, `viral_pneumonia_details.txt`, `lung_opacity_causes.txt`, `normal_lung_anatomy.txt`, `lung_nodule_management.txt` 等肺部专科知识文件。
2.  **创建辅助目录和文件:**
    ```bash
    touch reports.json # 报告数据持久化文件 (初始为空)
    mkdir -p uploads # 用户上传的原始影像和AI标注影像的存储目录
    ```

### **4. 启动Flask后端 (Linux 服务器)**

1.  **确保 `ai_pipeline_modules.py` 和 `app.py` 是最新版本。**
2.  **启动Flask应用:**
    ```bash
    python app.py
    ```
    *   Flask应用将启动，并初始化所有AI模型和RAG模块。它将监听 `http://0.0.0.0:8080`。
    *   后台清理任务和系统指标更新调度器将启动。

### **5. 前端设置与启动 (本地开发环境)**

1.  **安装Node.js和npm (或yarn)。**
2.  **进入前端项目目录:**
    ```bash
    cd my-ai-frontend
    ```
3.  **安装前端依赖:**
    ```bash
    npm install
    npm install axios vue-router@4 chart.js vue-chartjs
    ```
4.  **更新前端代码:**
    *   将 `src/App.vue`, `src/main.js`, `src/router/index.js` 和 `src/views/` 目录下的所有 `.vue` 文件更新为最新代码。
    *   **重要：** 确保所有前端文件中 `API_BASE_URL` 变量已替换为您的 **Flask 后端暴露网站地址** (例如 `http://i-2.gpushare.com:44953`)。
5.  **启动Vue前端:**
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
    *   在导航栏点击“技术信息”。
    *   查看项目技术栈、系统状态、**当前激活的病人端LLM模型**。
    *   通过下拉菜单和按钮，可以**动态切换**病人端LLM模型（MiniCPM或TinyLlama）。
    *   查看**实时系统资源监控**（CPU、内存、GPU使用率）和**AI模块性能延迟的历史趋势图**。

## 🌟 项目亮点与未来展望

本项目“智能肺部影像辅助诊断与科普平台”是您在AI算法、系统架构和工程实践方面综合能力的集大成者。它不仅成功地将前沿的AI技术应用于肺部疾病的辅助诊断这一高价值领域，更在实践中克服了从模型优化、幻觉抑制到系统运维等一系列复杂挑战。

**未来展望：** 平台仍有广阔的优化空间，例如引入更高级的LLM微调（LoRA/QLoRA）、集成图数据库进行更复杂的医学知识推理、开发交互式影像标注工具、实现用户认证与权限管理，以及进一步完善生产级容器化部署等，这些都将是项目持续演进的方向。

---
```
