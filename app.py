# /hy-tmp/Lingshu-7B/app.py
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
import os
import uuid
import json
from werkzeug.utils import secure_filename
import datetime
from apscheduler.schedulers.background import BackgroundScheduler
import time
import subprocess
import psutil
import csv # ！！！新增导入 ！！！

# 导入我们封装的AI模块
import ai_pipeline_modules

app = Flask(__name__)
CORS(app)

# 配置上传文件目录
UPLOAD_FOLDER = '/hy-tmp/Lingshu-7B/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# 允许上传的图片类型
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# 报告数据存储文件
REPORTS_FILE = '/hy-tmp/Lingshu-7B/reports.json'

# ！！！新增：监控指标相关全局变量 ！！！
METRICS_CSV_FILE = '/hy-tmp/Lingshu-7B/metrics_history.csv'
metrics_history = [] # 存储最近的性能数据
MAX_METRICS_HISTORY_LENGTH = 120 # 最多保留 300 条记录 (5秒一条，约25分钟)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_reports():
    """从reports.json加载所有报告"""
    if not os.path.exists(REPORTS_FILE) or os.path.getsize(REPORTS_FILE) == 0:
        return []
    with open(REPORTS_FILE, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            print(f"警告: {REPORTS_FILE} 文件内容损坏，将返回空列表。")
            return []

def save_reports(reports):
    """将报告保存到reports.json"""
    with open(REPORTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(reports, f, ensure_ascii=False, indent=4)

def clean_old_deleted_reports():
    """
    后台任务：清理超过一天的已删除报告。
    """
    print(f"\n[{datetime.datetime.now().isoformat()}] 正在运行后台清理任务...")
    reports = load_reports()
    
    updated_reports = []
    files_to_delete = []
    one_day_ago = datetime.datetime.now() - datetime.timedelta(days=1)

    for r in reports:
        if r.get("status") == "deleted":
            deleted_timestamp_str = r.get("deleted_timestamp")
            if deleted_timestamp_str:
                try:
                    deleted_time = datetime.datetime.fromisoformat(deleted_timestamp_str)
                    if deleted_time < one_day_ago:
                        if r.get("uploaded_image_filename"):
                            files_to_delete.append(os.path.join(app.config['UPLOAD_FOLDER'], r["uploaded_image_filename"]))
                        if r.get("annotated_image_filename"):
                            files_to_delete.append(os.path.join(app.config['UPLOAD_FOLDER'], r["annotated_image_filename"]))
                        print(f"报告 {r['report_id']} 已删除超过一天，将被永久清理。")
                        continue
                except ValueError:
                    print(f"警告: 报告 {r['report_id']} 的 deleted_timestamp 格式错误，跳过清理。")
            
        updated_reports.append(r)
    
    for f_path in files_to_delete:
        if os.path.exists(f_path):
            try:
                os.remove(f_path)
                print(f"已删除文件: {f_path}")
            except Exception as e:
                print(f"删除文件 {f_path} 时发生错误: {e}")

    if len(reports) != len(updated_reports):
        save_reports(updated_reports)
        print(f"后台清理完成。清理了 {len(reports) - len(updated_reports)} 份过期报告。")
    else:
        print("后台清理完成。没有发现需要清理的过期报告。")

# ！！！新增：保存指标到CSV ！！！
def save_metrics_to_csv():
    if not metrics_history:
        return

    fieldnames = metrics_history[0].keys()
    with open(METRICS_CSV_FILE, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_history)

# ！！！新增：从CSV加载指标 ！！！
def load_metrics_from_csv():
    global metrics_history
    if not os.path.exists(METRICS_CSV_FILE) or os.path.getsize(METRICS_CSV_FILE) == 0:
        return
    
    with open(METRICS_CSV_FILE, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        loaded_metrics = []
        for row in reader:
            # 尝试将数字字符串转换回数字类型
            for key, value in row.items():
                if key != 'timestamp' and value:
                    try:
                        row[key] = float(value)
                    except ValueError:
                        pass # 保持为字符串
            loaded_metrics.append(row)
        metrics_history = loaded_metrics[-MAX_METRICS_HISTORY_LENGTH:] # 只加载最近的
    print(f"从 {METRICS_CSV_FILE} 加载了 {len(metrics_history)} 条历史指标。")



def update_metrics():
    """
    后台任务：更新系统资源和应用级指标。
    """
    current_metrics = {
        "timestamp": datetime.datetime.now().isoformat(),
        "cpu_usage_percent": psutil.cpu_percent(interval=None),
        "memory_usage_percent": psutil.virtual_memory().percent,
        "gpu_usage_percent": 0.0,
        "gpu_memory_usage_percent": 0.0,
        "gpu_memory_total_mib": 0.0,
        "gpu_memory_used_mib": 0.0,
        "last_request_latency_seconds": 0.0, # 最近一次请求的延迟
        "last_ai_yolov8_latency_seconds": 0.0,
        "last_ai_qwen_latency_seconds": 0.0,
        "last_ai_patient_llm_latency_seconds": 0.0,
    }

    # GPU Usage (using nvidia-smi)
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.total,memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True
        )
        output_lines = result.stdout.strip().split('\n')
        if output_lines and output_lines[0].strip(): # 确保有输出且非空
            gpu_util, gpu_mem_total, gpu_mem_used = map(lambda x: float(x.strip().replace(' MiB', '').replace('%', '')), output_lines[0].split(','))
            
            current_metrics["gpu_usage_percent"] = gpu_util
            current_metrics["gpu_memory_total_mib"] = gpu_mem_total
            current_metrics["gpu_memory_used_mib"] = gpu_mem_used
            if gpu_mem_total > 0:
                current_metrics["gpu_memory_usage_percent"] = (gpu_mem_used / gpu_mem_total) * 100
            else:
                current_metrics["gpu_memory_usage_percent"] = 0
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        # print(f"警告: 无法获取GPU指标 (nvidia-smi error: {e})。请确保NVIDIA驱动和nvidia-smi已正确安装。")
        pass # 静默处理，避免频繁打印警告
    except Exception as e:
        print(f"获取GPU指标时发生未知错误: {e}")
        pass
    # 将当前指标添加到历史记录
    metrics_history.append(current_metrics)
    if len(metrics_history) > MAX_METRICS_HISTORY_LENGTH:
        metrics_history.pop(0) # 移除最旧的记录
    # 保存到 CSV
    save_metrics_to_csv()
    # print(f"[{datetime.datetime.now().isoformat()}] 指标已更新并保存到CSV。")
@app.route('/process_image', methods=['POST'])
def process_image():
    start_time = time.time()
    
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_filename = str(uuid.uuid4()) + "_" + filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        print(f"图片已保存到: {filepath}")

        try:
            patient_info = {
                "patient_id": request.form.get('patient_id', 'P' + str(uuid.uuid4())[:8]),
                "age": int(request.form.get('age', 45)),
                "gender": request.form.get('gender', '男'),
                "exam_type": request.form.get('exam_type', '胸部X光片'),
                "clinical_symptoms": request.form.get('clinical_symptoms', '持续咳嗽，伴有轻微胸痛，无发热。')
            }

            # --- 调用 AI 管道并记录延迟 ---
            ai_start_time_yolov8 = time.time()
            detected_objects_info, annotated_image_filename = ai_pipeline_modules.run_yolov8_segmentation(filepath)
            latency_yolov8 = time.time() - ai_start_time_yolov8
            
            if not detected_objects_info:
                detected_objects_info = [{"class_name": "未检测到明显病灶", "confidence": 1.0, "bbox": [0,0,0,0], "mask_coords": []}]
                print("YOLOv8 分割模型未检测到任何病灶，返回默认正常信息。")

            ai_start_time_qwen = time.time()
            professional_report_draft, kg_validation_output = \
                ai_pipeline_modules.generate_professional_report(patient_info, detected_objects_info)
            latency_qwen = time.time() - ai_start_time_qwen
            
            ai_start_time_patient_llm = time.time()
            patient_cop_report = ai_pipeline_modules.generate_cop_report(professional_report_draft)
            latency_patient_llm = time.time() - ai_start_time_patient_llm

            report_id = str(uuid.uuid4())
            new_report = {
                "report_id": report_id,
                "patient_info": patient_info,
                "uploaded_image_filename": unique_filename,
                "annotated_image_filename": annotated_image_filename,
                "detected_objects": detected_objects_info,
                "professional_report": professional_report_draft,
                "kg_validation_output": kg_validation_output,
                "patient_cop_report": patient_cop_report,
                "created_timestamp": datetime.datetime.now().isoformat(),
                "status": "active",
                "deleted_timestamp": None
            }
            reports = load_reports()
            reports.append(new_report)
            save_reports(reports)
            print(f"报告 {report_id} 已保存。")

            # ！！！更新最近一次请求的延迟指标 ！！！
            if metrics_history:
                metrics_history[-1]["last_request_latency_seconds"] = time.time() - start_time
                metrics_history[-1]["last_ai_yolov8_latency_seconds"] = latency_yolov8
                metrics_history[-1]["last_ai_qwen_latency_seconds"] = latency_qwen
                metrics_history[-1]["last_ai_patient_llm_latency_seconds"] = latency_patient_llm
                save_metrics_to_csv() # 立即保存，确保前端能拉取到最新数据

            return jsonify({
                "status": "success",
                "report_id": report_id,
                "professional_report": professional_report_draft,
                "kg_validation_output": kg_validation_output,
                "patient_cop_report": patient_cop_report,
                "detected_objects": detected_objects_info,
                "uploaded_image_filename": unique_filename,
                "annotated_image_filename": annotated_image_filename
            }), 200

        except Exception as e:
            print(f"处理请求时发生错误: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({"error": f"AI处理失败: {str(e)}"}), 500
        finally:
            pass
    else:
        return jsonify({"error": "File type not allowed"}), 400

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/reports', methods=['GET'])
def get_all_reports():
    reports = load_reports()
    active_reports = [r for r in reports if r.get("status", "active") == "active"]
    report_summaries = [
        {
            "report_id": r["report_id"],
            "patient_id": r["patient_info"]["patient_id"],
            "age": r["patient_info"]["age"],
            "gender": r["patient_info"]["gender"],
            "exam_type": r["patient_info"]["exam_type"],
            "timestamp": r.get("created_timestamp", "未知时间"),
            "uploaded_image_filename": r.get("uploaded_image_filename"),
            "annotated_image_filename": r.get("annotated_image_filename"),
            "status": r.get("status", "active")
        } for r in active_reports
    ]
    report_summaries.sort(key=lambda x: x["timestamp"], reverse=True)
    return jsonify(report_summaries), 200

@app.route('/api/reports/<report_id>', methods=['GET'])
def get_single_report(report_id):
    reports = load_reports()
    report = next((r for r in reports if r["report_id"] == report_id), None)
    if report:
        return jsonify(report), 200
    return jsonify({"error": "Report not found"}), 404

@app.route('/api/reports/<report_id>', methods=['DELETE'])
def delete_report(report_id):
    reports = load_reports()
    report_found = False
    for r in reports:
        if r["report_id"] == report_id:
            r["status"] = "deleted"
            r["deleted_timestamp"] = datetime.datetime.now().isoformat()
            report_found = True
            break
    
    if report_found:
        save_reports(reports)
        print(f"报告 {report_id} 已标记为 'deleted'。")
        return jsonify({"status": "success", "message": f"Report {report_id} marked as deleted."}), 200
    return jsonify({"error": "Report not found"}), 404

@app.route('/api/patient_llm_status', methods=['GET'])
def get_patient_llm_status():
    return jsonify({
        "current_patient_llm": ai_pipeline_modules.current_patient_llm_name
    }), 200

@app.route('/api/admin/switch_patient_llm', methods=['POST'])
def switch_patient_llm():
    print("警告: 正在尝试切换病人端LLM模型。在生产环境中，此API需要严格认证。")

    data = request.get_json()
    model_name = data.get('model_name')

    if not model_name:
        return jsonify({"status": "error", "message": "Missing 'model_name' parameter."}), 400
    
    if model_name not in ai_pipeline_modules.PATIENT_LLM_MODELS_MAP:
        return jsonify({"status": "error", "message": f"Unknown model name: {model_name}. Available: {list(ai_pipeline_modules.PATIENT_LLM_MODELS_MAP.keys())}"}), 400

    success = ai_pipeline_modules.load_patient_llm_model(model_name)

    if success:
        return jsonify({"status": "success", "message": f"Switched patient LLM to {model_name}.", "current_patient_llm": ai_pipeline_modules.current_patient_llm_name}), 200
    else:
        return jsonify({"status": "error", "message": f"Failed to switch patient LLM to {model_name}.", "current_patient_llm": ai_pipeline_modules.current_patient_llm_name}), 500


@app.route('/api/technical_info', methods=['GET'])
def get_technical_info():
    # ！！！修改：返回实时和历史指标 ！！！
    current_metrics_data = metrics_history[-1] if metrics_history else {}
    history_metrics_data = metrics_history # 返回所有历史记录

    return jsonify({
        "project_name": "智能影像报告辅助生成与患者科普系统",
        "backend_framework": "Flask",
        "frontend_framework": "Vue3",
        "ai_models": {
            "image_detection": "YOLOv8n-seg (真实微调，肺部专科)",
            "professional_report_llm": "Qwen1.5-1.8B-Chat (4-bit 量化)",
            "patient_cop_llm": ai_pipeline_modules.current_patient_llm_name,
            "embedding_model": "moka-ai/m3e-base (CPU)",
            "rag_vector_db": "ChromaDB",
            "knowledge_graph_type": "Python Dictionary (肺部专科)"
        },
        "system_metrics_current": current_metrics_data, # 实时指标
        "system_metrics_history": history_metrics_data, # 历史指标
        "status": "All AI modules loaded and operational.",
        "last_update": datetime.datetime.now().isoformat()
    }), 200

# ！！！新增：获取所有历史指标的API ！！！
@app.route('/api/monitor/metrics_history', methods=['GET'])
def get_metrics_history():
    return jsonify(metrics_history), 200


if __name__ == '__main__':
    ai_pipeline_modules.init_ai_modules()
    
    # ！！！加载历史指标 ！！！
    load_metrics_from_csv()

    scheduler = BackgroundScheduler()
    scheduler.add_job(clean_old_deleted_reports, 'interval', hours=1)
    # ！！！修改：每隔5秒更新一次指标并保存到CSV ！！！
    scheduler.add_job(update_metrics, 'interval', seconds=10) 
    scheduler.start()
    print("后台清理任务和系统指标更新调度器已启动。")

    print("Flask应用启动，AI模块已初始化完毕。")
    app.run(host='0.0.0.0', port=8080, debug=False)