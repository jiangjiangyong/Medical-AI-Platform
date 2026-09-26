<script setup>
import { computed, onBeforeUnmount, onMounted, reactive, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ArrowLeft, CircleCheck, Clock, Document, EditPen, Upload, Warning } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import { casesApi } from '../services/api'

const route = useRoute()
const router = useRouter()
const detail = ref(null)
const loading = ref(true)
const verifying = ref(false)
const uploading = ref(false)
const analyzing = ref(false)
const savingReport = ref(false)
const file = ref(null)
const reviewNote = ref('')
const editingReport = ref(false)
const intakeCode = ref(sessionStorage.getItem(`case-intake-code-${route.params.id}`) || '')
const pollTimer = ref(null)
const pollInFlight = ref(false)
const currentUser = JSON.parse(localStorage.getItem('medical_platform_user') || '{}')
const role = currentUser.role
const isPatient = role === 'patient'
const isStaff = computed(() => ['admin', 'doctor', 'health_manager'].includes(role))
const identityForm = reactive({ confirmed_name: currentUser.display_name || '', intake_code: '', confirm_identity: false })
const editForm = reactive({ imaging_findings_text: '', preliminary_assessment: '', recommendations_text: '', risk_level: 'moderate', change_note: '' })
const statusText = {
  created: '待核验', uploaded: '资料已上传', processing: '分析处理中', pending_review: '待医生审核',
  needs_revision: '待修改', patient_explanation_ready: '生成患者说明', approved: '已审核', published: '已发布',
  analysis_failed: '分析失败',
}
const workflowSteps = [
  { title: '身份核验', note: '确认检查归属' },
  { title: '上传检查', note: '校验影像资料' },
  { title: '辅助分析', note: '输出结构化结果' },
  { title: '医生发布', note: '编辑并确认报告' },
  { title: '患者随访', note: '查看说明与任务' },
]

const professionalReport = computed(() => detail.value?.reports?.find((item) => item.report_type === 'professional'))
const patientReport = computed(() => detail.value?.reports?.find((item) => item.report_type === 'patient' && item.status === 'published'))
const professionalData = computed(() => professionalReport.value?.structured_data || {})
const patientData = computed(() => patientReport.value?.structured_data || {})
const isProcessing = computed(() => detail.value?.status === 'processing')
const workflowStep = computed(() => {
  if (!detail.value) return 0
  if (detail.value.status === 'published') return 5
  if (['approved', 'patient_explanation_ready'].includes(detail.value.status)) return 4
  if (['pending_review', 'needs_revision'].includes(detail.value.status) || professionalReport.value) return 3
  if (detail.value.status === 'processing') return 2
  if (detail.value.studies?.length) return 2
  if (detail.value.identity_verified) return 1
  return 0
})

async function load(silent = false) {
  if (!silent) loading.value = true
  try {
    detail.value = (await casesApi.detail(route.params.id)).data
    if (detail.value.status === 'processing') startPolling()
    else stopPolling()
    if (professionalReport.value && !editingReport.value) syncEditForm()
  } catch (error) {
    ElMessage.error(error.response?.data?.detail || '病例加载失败')
  } finally {
    if (!silent) loading.value = false
  }
}

function startPolling() {
  if (pollTimer.value) return
  pollTimer.value = window.setInterval(async () => {
    if (pollInFlight.value || !detail.value || detail.value.status !== 'processing') return
    pollInFlight.value = true
    try { await load(true) } finally { pollInFlight.value = false }
  }, 3000)
}

function stopPolling() {
  if (pollTimer.value) window.clearInterval(pollTimer.value)
  pollTimer.value = null
}

function chooseFile(event) { file.value = event.target.files?.[0] || null }

async function verifyIdentity() {
  if (!identityForm.confirm_identity) return ElMessage.warning('请先确认这是您本人的检查资料')
  verifying.value = true
  try {
    await casesApi.verifyIdentity(route.params.id, identityForm)
    ElMessage.success('身份核验完成，现在可以上传检查资料')
    await load()
  } catch (error) { ElMessage.error(error.response?.data?.detail || '身份核验失败') } finally { verifying.value = false }
}

async function upload() {
  if (!file.value) return ElMessage.warning('请先选择影像或检查报告')
  if (isPatient && !detail.value.identity_verified) return ElMessage.warning('请先完成身份核验')
  uploading.value = true
  try {
    await casesApi.upload(route.params.id, file.value)
    file.value = null
    ElMessage.success('资料校验通过，系统已开始分析')
    await load()
  } catch (error) { ElMessage.error(error.response?.data?.detail || '上传失败') } finally { uploading.value = false }
}

async function runDemo() {
  analyzing.value = true
  try {
    await casesApi.analyze(route.params.id)
    ElMessage.success('演示分析已完成，报告进入医生审核')
    await load()
  } catch (error) { ElMessage.error(error.response?.data?.detail || '演示分析失败') } finally { analyzing.value = false }
}

function syncEditForm() {
  const data = professionalData.value
  editForm.imaging_findings_text = (data.imaging_findings || []).join('\n')
  editForm.preliminary_assessment = data.preliminary_assessment || ''
  editForm.recommendations_text = (data.recommendations || []).join('\n')
  editForm.risk_level = data.risk_level || 'moderate'
}

function beginEdit() {
  syncEditForm()
  editForm.change_note = ''
  editingReport.value = true
}

async function saveReport() {
  if (!professionalReport.value || !editForm.preliminary_assessment.trim()) return ElMessage.warning('请补充初步判断')
  savingReport.value = true
  const lines = (value) => value.split('\n').map((item) => item.trim()).filter(Boolean)
  try {
    await casesApi.editReport(route.params.id, professionalReport.value.id, {
      imaging_findings: lines(editForm.imaging_findings_text),
      preliminary_assessment: editForm.preliminary_assessment.trim(),
      recommendations: lines(editForm.recommendations_text),
      risk_level: editForm.risk_level,
      change_note: editForm.change_note.trim(),
    })
    editingReport.value = false
    ElMessage.success('报告草稿已保存，等待重新审核')
    await load()
  } catch (error) { ElMessage.error(error.response?.data?.detail || '报告保存失败') } finally { savingReport.value = false }
}

async function review(status) {
  if (!professionalReport.value) return
  try {
    await casesApi.review(route.params.id, professionalReport.value.id, { status, note: reviewNote.value })
    reviewNote.value = ''
    ElMessage.success(status === 'approved' ? '报告已发布，患者说明正在生成' : '报告已退回修改')
    await load()
  } catch (error) { ElMessage.error(error.response?.data?.detail || '审核失败') }
}

onMounted(load)
onBeforeUnmount(stopPolling)
</script>

<template>
  <div v-if="detail" v-loading="loading" class="case-detail-page">
    <div class="detail-toolbar"><button class="soft-action" @click="router.push('/cases')"><el-icon><ArrowLeft /></el-icon>返回病例中心</button><span class="detail-id">CASE / {{ detail.id.slice(0, 8) }}</span></div>

    <section class="case-hero">
      <div><span class="eyebrow" style="color:#ecb866">{{ isPatient ? 'MY STUDY' : 'CASE REVIEW' }}</span><h2>{{ detail.title }}</h2><p>{{ detail.symptoms || '未填写症状或检查背景' }}</p></div>
      <div class="hero-facts"><div class="hero-fact"><span>检查类型</span><strong>胸部 X 光</strong></div><div class="hero-fact"><span>当前状态</span><strong class="hero-status">{{ statusText[detail.status] || detail.status }}</strong></div></div>
    </section>

    <section class="workflow-strip">
      <div v-for="(step, index) in workflowSteps" :key="step.title" class="workflow-step" :class="{ complete: index < workflowStep, active: index === workflowStep }">
        <span class="workflow-step-number"><el-icon v-if="index < workflowStep"><CircleCheck /></el-icon><template v-else>{{ index + 1 }}</template></span>
        <div><strong>{{ step.title }}</strong><small>{{ step.note }}</small></div>
      </div>
    </section>

    <div v-if="isStaff && intakeCode" class="intake-code-banner"><div><strong>患者核验码</strong><span>请通过受控渠道提供给对应患者，本码仅在本次检查建档时显示。</span></div><code>{{ intakeCode }}</code></div>

    <div v-if="isStaff && ['pending_review', 'needs_revision'].includes(professionalReport?.status)" class="review-bar review-bar-enhanced">
      <div class="review-copy"><el-icon><Warning /></el-icon><div><strong>{{ professionalReport.status === 'needs_revision' ? '报告已退回，需要修改后重新审核' : '报告等待专业审核' }}</strong><span>患者只能看到医生确认后的最终说明。</span></div></div>
      <el-input v-model="reviewNote" class="review-note" placeholder="可选：填写审核备注" clearable />
      <div class="review-actions"><button class="reject-button" @click="review('needs_revision')">退回修改</button><button class="approve-button" @click="review('approved')">确认并发布</button></div>
    </div>

    <div class="case-layout">
      <div>
        <section v-if="isPatient && detail.requires_identity_verification" class="panel intake-panel">
          <div class="panel-header panel-header-stack"><div><span class="panel-step-label">STEP 01</span><h3>先核验这项检查</h3><span>核对姓名和检查归属后，才能上传资料</span></div><span class="simulation-badge"><span class="status-dot"></span>待确认</span></div>
          <div class="panel-body">
            <el-form label-position="top" @submit.prevent="verifyIdentity">
              <el-form-item label="患者姓名"><el-input v-model="identityForm.confirmed_name" placeholder="请输入当前账号姓名" /></el-form-item>
              <el-form-item label="检查编号 / 一次性核验码"><el-input v-model="identityForm.intake_code" placeholder="如工作人员提供，请输入核验码；自建演示病例可留空" /></el-form-item>
              <label class="consent-check"><input v-model="identityForm.confirm_identity" type="checkbox" />我确认这份检查资料属于本人，并同意平台用于本次辅助分析</label>
              <button class="primary-action form-submit-action" type="submit" :disabled="verifying"><el-icon><CircleCheck /></el-icon>{{ verifying ? '核验中…' : '完成身份核验' }}</button>
            </el-form>
          </div>
        </section>

        <section v-if="isStaff || detail.identity_verified" class="panel workflow-panel">
          <div class="panel-header panel-header-stack"><div><span class="panel-step-label">STEP 02 / 03</span><h3>上传并提交影像资料</h3><span>{{ isPatient ? '资料校验通过后会自动进入分析队列' : '上传后自动触发模拟视觉适配器和报告草稿生成' }}</span></div><span v-if="isProcessing" class="simulation-badge"><span class="status-dot"></span>处理中</span></div>
          <div class="panel-body">
            <div v-if="isPatient" class="patient-upload-note"><el-icon><Document /></el-icon><span>当前只接收 JPG、PNG 或 PDF，单个文件不超过 16 MB。患者端不会展示未经医生确认的模型结论。</span></div>
            <div v-else class="simulation-notice"><el-icon><Warning /></el-icon><div><strong>当前视觉模型为演示适配器</strong><p>上传资料会按固定 JSON 契约生成候选结果，后续可替换为私有部署的医学影像模型。</p></div></div>
            <div class="upload-workflow-row">
              <div class="upload-zone compact-upload"><input id="study-file" type="file" accept=".jpg,.jpeg,.png,.pdf" @change="chooseFile" /><label for="study-file"><el-icon><Upload /></el-icon>{{ file ? file.name : '选择胸片或检查报告' }}</label><p>{{ detail.studies.length ? `已归档 ${detail.studies.length} 份资料` : '请选择本次检查资料' }}</p></div>
              <div class="upload-side-actions"><button class="primary-action" :disabled="uploading || isProcessing" @click="upload"><el-icon><Upload /></el-icon>{{ uploading ? '上传中…' : '上传并开始分析' }}</button><button v-if="isStaff" class="soft-action" :disabled="analyzing || isProcessing" @click="runDemo"><el-icon><CircleCheck /></el-icon>{{ analyzing ? '演示分析中…' : '无影像运行演示' }}</button></div>
            </div>
            <div v-if="isProcessing" class="processing-note"><span class="processing-spinner"></span><div><strong>系统正在处理本次检查</strong><span>影像校验、结构化分析和医生报告草稿完成后，页面会自动更新。</span></div></div>
            <div v-if="detail.studies.length" class="study-file-list"><div v-for="study in detail.studies" :key="study.id" class="study-file-item"><el-icon><Document /></el-icon><span>{{ study.original_name }}</span><small>{{ (study.file_size / 1024 / 1024).toFixed(2) }} MB</small></div></div>
          </div>
        </section>

        <section v-if="detail.vision_analysis && isStaff" class="panel result-panel">
          <div class="panel-header"><div><span class="panel-step-label">STEP 03</span><h3>结构化视觉结果</h3></div><span class="simulation-badge"><span class="status-dot"></span>{{ detail.vision_analysis.model_name }}</span></div>
          <div class="panel-body"><div class="analysis-box"><div class="result-heading"><h4>{{ detail.vision_analysis.result.impression }}</h4><span class="risk-pill" :class="detail.vision_analysis.result.risk_level">{{ detail.vision_analysis.result.risk_level === 'high' ? '高优先级' : detail.vision_analysis.result.risk_level === 'moderate' ? '需复核' : '低风险' }}</span></div><div class="finding-list"><div v-for="finding in detail.vision_analysis.result.findings" :key="finding.name" class="finding-item"><div><strong>{{ finding.name }}</strong><div class="finding-evidence">{{ finding.evidence }}</div></div><span>{{ finding.location }} · {{ Math.round(finding.confidence * 100) }}%</span></div><div v-if="!detail.vision_analysis.result.findings.length" class="finding-item"><strong>未发现明确异常征象</strong><span>模型结果</span></div></div><div class="result-limitations"><el-icon><Warning /></el-icon><span>{{ detail.vision_analysis.result.simulated ? '当前为演示结果，不能作为临床诊断依据。' : '模型结果仅供专业人员复核。' }}</span></div></div></div>
        </section>

        <section v-if="professionalReport && isStaff" class="panel report-panel">
          <div class="panel-header"><div><span class="panel-step-label">STEP 04</span><h3>医生端辅助报告</h3></div><div class="report-header-actions"><span class="status-tag" :class="professionalReport.status">{{ statusText[professionalReport.status] || professionalReport.status }}</span><button v-if="!editingReport" class="icon-button report-edit-button" title="编辑报告" @click="beginEdit"><el-icon><EditPen /></el-icon></button></div></div>
          <div v-if="!editingReport" class="panel-body structured-report">
            <div v-if="professionalData.imaging_findings?.length" class="report-section"><h4>影像所见</h4><ul class="report-list"><li v-for="item in professionalData.imaging_findings" :key="item">{{ item }}</li></ul></div>
            <div v-if="professionalData.preliminary_assessment" class="report-section"><h4>初步判断</h4><p>{{ professionalData.preliminary_assessment }}</p></div>
            <div v-if="professionalData.recommendations?.length" class="report-section"><h4>建议</h4><ul class="report-list"><li v-for="item in professionalData.recommendations" :key="item">{{ item }}</li></ul></div>
            <div class="report-footer"><span>风险等级：{{ professionalData.risk_level || '待确认' }}</span><span>报告版本：v{{ professionalReport.version || 1 }}</span><span v-if="professionalReport.evidence?.length">已关联 {{ professionalReport.evidence.length }} 条知识证据</span></div>
          </div>
          <div v-else class="panel-body report-editor">
            <div class="editor-note">保存后报告会回到待审核状态，患者端不会看到未重新确认的内容。</div>
            <label>影像所见（每行一条）<el-input v-model="editForm.imaging_findings_text" type="textarea" :rows="4" /></label>
            <label>初步判断<el-input v-model="editForm.preliminary_assessment" type="textarea" :rows="4" /></label>
            <label>建议（每行一条）<el-input v-model="editForm.recommendations_text" type="textarea" :rows="4" /></label>
            <label>风险等级<select v-model="editForm.risk_level" class="filter-select editor-select"><option value="low">低风险</option><option value="moderate">需复核</option><option value="high">高优先级</option></select></label>
            <label>修改说明（可选）<el-input v-model="editForm.change_note" placeholder="例如：补充既往影像对比要求" /></label>
            <div class="editor-actions"><button class="soft-action" @click="editingReport = false">取消</button><button class="primary-action" :disabled="savingReport" @click="saveReport"><el-icon><CircleCheck /></el-icon>{{ savingReport ? '保存中…' : '保存并重新审核' }}</button></div>
          </div>
        </section>
      </div>

      <div>
        <section class="panel patient-report-panel">
          <div class="panel-header"><div><span class="panel-step-label">STEP 05</span><h3>患者端说明</h3></div><span v-if="patientReport" class="status-tag published">已发布</span></div>
          <div v-if="patientReport" class="panel-body structured-report">
            <div v-if="patientData.summary" class="patient-summary-box"><el-icon><CircleCheck /></el-icon><span>{{ patientData.summary }}</span></div>
            <div v-if="patientData.what_it_means" class="report-section"><h4>目前看到什么</h4><p>{{ patientData.what_it_means }}</p></div>
            <div v-if="patientData.what_to_do?.length" class="report-section"><h4>接下来可以做什么</h4><ul class="report-list"><li v-for="item in patientData.what_to_do" :key="item">{{ item }}</li></ul></div>
            <div v-if="patientData.questions_for_clinician?.length" class="report-section question-section"><h4>可以和医生确认的问题</h4><ul class="report-list"><li v-for="item in patientData.questions_for_clinician" :key="item">{{ item }}</li></ul></div>
            <div v-if="patientData.when_to_seek_help" class="attention-note"><el-icon><Warning /></el-icon><div><strong>需要注意</strong><p>{{ patientData.when_to_seek_help }}</p></div></div>
            <div v-if="patientData.safety_note" class="report-safety">{{ patientData.safety_note }}</div>
          </div>
          <div v-else class="empty-state report-empty"><el-icon><Clock /></el-icon><strong>{{ isPatient && !detail.identity_verified ? '完成身份核验后继续' : '医生发布后会生成患者说明' }}</strong><span>{{ isPatient && !detail.identity_verified ? '核验当前检查归属后，上传资料并等待医生确认。' : '患者端只会看到基于医生最终报告生成的通俗解释。' }}</span></div>
        </section>

        <section class="panel followup-panel">
          <div class="panel-header"><div><h3>随访任务</h3><span>{{ detail.follow_ups.length ? `${detail.follow_ups.length} 项安排` : '医生发布后生成' }}</span></div><el-icon class="panel-header-icon"><Clock /></el-icon></div>
          <div v-if="detail.follow_ups.length" class="task-card-list"><div v-for="task in detail.follow_ups" :key="task.id" class="task-card"><div><strong>{{ task.title }}</strong><small>{{ task.description }}</small></div><div class="task-card-bottom"><span class="status-tag" :class="task.status">{{ task.status === 'completed' ? '已完成' : task.status === 'in_progress' ? '进行中' : '待处理' }}</span><span>{{ task.due_at ? new Date(task.due_at).toLocaleDateString('zh-CN') : '待确认' }}</span></div></div></div>
          <div v-else class="empty-state">医生确认并发布后，会自动生成下一步任务。</div>
        </section>
      </div>
    </div>
  </div>
</template>
