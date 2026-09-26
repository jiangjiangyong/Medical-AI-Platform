<script setup>
import { computed, onMounted, reactive, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ArrowRight, DocumentAdd, Search } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import { casesApi } from '../services/api'

const router = useRouter()
const user = JSON.parse(localStorage.getItem('medical_platform_user') || '{}')
const role = user.role
const isStaff = ['admin', 'doctor', 'health_manager'].includes(role)
const cases = ref([])
const patients = ref([])
const loading = ref(true)
const dialogVisible = ref(false)
const creating = ref(false)
const searchQuery = ref('')
const statusFilter = ref('all')
const form = reactive({ patient_id: '', title: '胸部 X 光辅助决策病例', symptoms: '', clinical_context: {} })
const statusText = { created: '待核验', uploaded: '资料已上传', processing: '分析处理中', pending_review: '待审核', needs_revision: '需修改', approved: '已审核', published: '已发布', analysis_failed: '分析失败' }
const statusOptions = [
  { value: 'all', label: '全部状态' },
  { value: 'created', label: '待核验' },
  { value: 'processing', label: '分析处理中' },
  { value: 'pending_review', label: '待审核' },
  { value: 'needs_revision', label: '需修改' },
  { value: 'published', label: '已发布' },
]

const filteredCases = computed(() => {
  const keyword = searchQuery.value.trim().toLowerCase()
  return cases.value.filter((item) => {
    const matchesKeyword = !keyword || item.title.toLowerCase().includes(keyword) || item.id.toLowerCase().includes(keyword)
    const matchesStatus = statusFilter.value === 'all' || item.status === statusFilter.value
    return matchesKeyword && matchesStatus
  })
})

async function load() {
  loading.value = true
  try { cases.value = (await casesApi.list()).data } catch (error) { ElMessage.error(error.response?.data?.detail || '病例加载失败') } finally { loading.value = false }
}

async function loadPatients() {
  if (!isStaff) return
  try { patients.value = (await casesApi.patients()).data } catch (error) { ElMessage.error(error.response?.data?.detail || '患者列表加载失败') }
}

function openCreate() {
  form.patient_id = ''
  form.title = '胸部 X 光辅助决策病例'
  form.symptoms = ''
  dialogVisible.value = true
}

async function create() {
  if (isStaff && !form.patient_id) return ElMessage.warning('请先选择需要建立病例的患者')
  creating.value = true
  try {
    const { data } = await casesApi.create({ ...form, patient_id: isStaff ? form.patient_id : undefined })
    if (data.intake_code) sessionStorage.setItem(`case-intake-code-${data.id}`, data.intake_code)
    dialogVisible.value = false
    router.push(`/cases/${data.id}`)
  } catch (error) { ElMessage.error(error.response?.data?.detail || '病例创建失败') } finally { creating.value = false }
}

onMounted(() => { load(); loadPatients() })
</script>

<template>
  <div class="page-heading">
    <div>
      <span class="section-kicker">{{ role === 'patient' ? 'MY STUDIES' : 'CASE CENTER' }}</span>
      <h2>{{ role === 'patient' ? '我的检查' : '病例中心' }}</h2>
      <p>{{ role === 'patient' ? '查看检查进度、审核后的说明和随访安排。' : '从患者资料到报告审核，集中处理每个影像病例。' }}</p>
    </div>
    <div class="heading-actions">
      <button class="primary-action" @click="openCreate"><el-icon><DocumentAdd /></el-icon>{{ role === 'patient' ? '提交检查资料' : '新建病例' }}</button>
    </div>
  </div>

  <section class="panel">
    <div class="panel-header case-list-header"><div><h3>{{ role === 'patient' ? '检查记录' : '病例列表' }}</h3><span>{{ filteredCases.length }} / {{ cases.length }} 条记录</span></div><span class="panel-header-mark"><el-icon><Search /></el-icon></span></div>
    <div class="filter-toolbar">
      <div class="filter-controls">
        <el-input v-model="searchQuery" class="case-search" clearable placeholder="搜索标题或病例编号" :prefix-icon="Search" />
        <select v-model="statusFilter" class="filter-select" aria-label="筛选病例状态">
          <option v-for="option in statusOptions" :key="option.value" :value="option.value">{{ option.label }}</option>
        </select>
      </div>
      <button v-if="searchQuery || statusFilter !== 'all'" type="button" class="filter-reset" @click="searchQuery = ''; statusFilter = 'all'">清除筛选</button>
    </div>
    <div v-loading="loading" class="table-wrap">
      <table v-if="filteredCases.length" class="data-table case-table"><thead><tr><th>{{ role === 'patient' ? '检查' : '病例' }}</th><th>检查类型</th><th>优先级</th><th>状态</th><th>更新时间</th><th></th></tr></thead>
        <tbody><tr v-for="item in filteredCases" :key="item.id"><td><span class="case-title">{{ item.title }}</span><span class="case-sub">{{ item.id }}</span></td><td>胸部 X 光</td><td><span class="status-tag" :class="item.priority === 'high' ? 'high' : 'created'">{{ item.priority === 'high' ? '高' : '常规' }}</span></td><td><span class="status-tag" :class="item.status">{{ statusText[item.status] || item.status }}</span></td><td>{{ new Date(item.updated_at).toLocaleString('zh-CN') }}</td><td><button class="link-button" @click="router.push(`/cases/${item.id}`)">打开 <el-icon><ArrowRight /></el-icon></button></td></tr></tbody>
      </table>
      <div v-if="filteredCases.length" class="case-mobile-list">
        <button v-for="item in filteredCases" :key="`mobile-${item.id}`" type="button" class="case-mobile-card" @click="router.push(`/cases/${item.id}`)">
          <div class="case-mobile-card-top"><strong>{{ item.title }}</strong><span class="status-tag" :class="item.status">{{ statusText[item.status] || item.status }}</span></div>
          <div class="case-mobile-card-meta"><span>胸部 X 光</span><span>{{ item.priority === 'high' ? '高优先级' : '常规' }}</span><span>{{ new Date(item.updated_at).toLocaleDateString('zh-CN') }}</span><el-icon><ArrowRight /></el-icon></div>
        </button>
      </div>
      <div v-if="!loading && !filteredCases.length" class="empty-state">{{ cases.length ? '没有符合当前筛选条件的记录。' : '暂无记录，点击上方按钮创建第一条病例。' }}</div>
    </div>
  </section>

  <el-dialog v-model="dialogVisible" :title="role === 'patient' ? '提交一项检查' : '新建辅助决策病例'" width="520px">
    <el-form label-position="top" @submit.prevent="create">
      <div v-if="isStaff" class="form-context-note">先选择患者，再补充症状和检查背景，后续报告会自动归档到该患者名下。</div>
      <el-form-item v-if="isStaff" label="关联患者" required>
        <el-select v-model="form.patient_id" filterable clearable placeholder="搜索并选择患者" style="width:100%">
          <el-option v-for="patient in patients" :key="patient.id" :label="`${patient.display_name} · ${patient.email}`" :value="patient.id" />
        </el-select>
        <span v-if="!patients.length" class="form-help">当前没有可用的患者账号，请先创建患者账号。</span>
      </el-form-item>
      <el-form-item label="记录名称"><el-input v-model="form.title" /></el-form-item>
      <el-form-item label="症状或检查背景"><el-input v-model="form.symptoms" type="textarea" :rows="4" placeholder="例如：持续咳嗽，无明显发热；可留空后补充" /></el-form-item>
      <div class="dialog-actions"><el-button @click="dialogVisible = false">取消</el-button><el-button type="primary" :loading="creating" @click="create">创建并继续</el-button></div>
    </el-form>
  </el-dialog>
</template>
