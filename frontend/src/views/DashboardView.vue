<script setup>
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ArrowRight, Calendar, CircleCheck, DocumentAdd, Warning } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import { dashboardApi } from '../services/api'

const router = useRouter()
const summary = ref({ total_cases: 0, pending_review: 0, active_followups: 0, high_priority: 0, recent_cases: [] })
const loading = ref(true)
const user = JSON.parse(localStorage.getItem('medical_platform_user') || '{}')
const isPatient = computed(() => user.role === 'patient')

const statusText = { created: '待核验', uploaded: '资料已上传', processing: '分析处理中', pending_review: '待审核', needs_revision: '需修改', approved: '已审核', published: '已发布', analysis_failed: '分析失败' }
const metrics = computed(() => isPatient.value
  ? [
      { label: '我的检查', value: summary.value.total_cases, foot: '已归档的影像病例', tone: 'teal' },
      { label: '待查看报告', value: summary.value.pending_review, foot: '等待专业人员审核', tone: 'amber' },
      { label: '待办随访', value: summary.value.active_followups, foot: '需要按计划完成', tone: 'blue' },
      { label: '需优先关注', value: summary.value.high_priority, foot: '建议尽快查看详情', tone: 'rose' },
    ]
  : [
      { label: '病例总数', value: summary.value.total_cases, foot: '持续管理中的影像病例', tone: 'teal' },
      { label: '待审核', value: summary.value.pending_review, foot: '需要专业人员确认', tone: 'amber' },
      { label: '进行中随访', value: summary.value.active_followups, foot: '按计划保持跟进', tone: 'blue' },
      { label: '高优先级', value: summary.value.high_priority, foot: '建议尽快人工复核', tone: 'rose' },
    ])

async function load() {
  loading.value = true
  try { summary.value = (await dashboardApi.summary()).data } catch (error) { ElMessage.error(error.response?.data?.detail || '概览加载失败') } finally { loading.value = false }
}

onMounted(load)
</script>

<template>
  <div class="page-heading dashboard-heading">
    <div>
      <span class="section-kicker">{{ isPatient ? 'PERSONAL HEALTH' : 'CASE OPERATIONS' }}</span>
      <h2>{{ isPatient ? `你好，${user.display_name || '朋友'}` : `早上好，${user.display_name || '团队'}` }}</h2>
      <p>{{ isPatient ? '这里集中展示你的检查、审核报告和下一步随访安排。' : '把需要关注的病例、审核和随访集中在这里。' }}</p>
    </div>
    <div class="heading-actions">
      <button v-if="isPatient" class="soft-action" @click="router.push('/follow-ups')"><el-icon><Calendar /></el-icon>查看我的随访</button>
      <button v-else class="primary-action" @click="router.push('/cases')"><el-icon><DocumentAdd /></el-icon>创建病例</button>
    </div>
  </div>

  <div v-loading="loading" class="metric-grid">
    <div v-for="metric in metrics" :key="metric.label" class="metric-card" :class="`metric-${metric.tone}`">
      <span class="metric-label">{{ metric.label }}</span>
      <div class="metric-value">{{ metric.value }}</div>
      <div class="metric-foot">{{ metric.foot }}</div>
    </div>
  </div>

  <section class="panel quick-panel">
    <div class="panel-header">
      <div><h3>{{ isPatient ? '现在先做什么？' : '优先处理' }}</h3><span>{{ isPatient ? '从最近一次检查开始查看' : '从最需要人工确认的事项开始' }}</span></div>
      <span class="panel-header-mark"><el-icon><CircleCheck /></el-icon></span>
    </div>
    <div class="quick-action-grid">
      <button class="quick-action-card" type="button" @click="router.push('/cases')">
        <span class="quick-action-icon"><el-icon><DocumentAdd /></el-icon></span>
        <span><strong>{{ isPatient ? '查看我的检查' : '新建或打开病例' }}</strong><small>{{ isPatient ? '阅读已审核的检查说明' : '上传资料并启动辅助分析' }}</small></span>
        <el-icon class="quick-action-arrow"><ArrowRight /></el-icon>
      </button>
      <button class="quick-action-card" type="button" @click="router.push(isPatient ? '/follow-ups' : '/cases')">
        <span class="quick-action-icon" :class="{ 'quick-action-alert': !isPatient && summary.pending_review }"><el-icon><Warning /></el-icon></span>
        <span><strong>{{ isPatient ? '查看下一步随访' : (summary.pending_review ? `处理 ${summary.pending_review} 个待审核病例` : '查看病例状态') }}</strong><small>{{ isPatient ? '确认需要完成的健康任务' : '确认报告后才能向患者发布' }}</small></span>
        <el-icon class="quick-action-arrow"><ArrowRight /></el-icon>
      </button>
      <button class="quick-action-card" type="button" @click="router.push('/follow-ups')">
        <span class="quick-action-icon"><el-icon><Calendar /></el-icon></span>
        <span><strong>{{ isPatient ? '打开随访清单' : '跟进任务清单' }}</strong><small>{{ summary.active_followups ? `${summary.active_followups} 项任务进行中` : '当前没有进行中的任务' }}</small></span>
        <el-icon class="quick-action-arrow"><ArrowRight /></el-icon>
      </button>
    </div>
  </section>

  <div class="dashboard-grid">
    <section class="panel">
      <div class="panel-header"><div><h3>{{ isPatient ? '最近的检查' : '最近病例' }}</h3><span>{{ summary.recent_cases.length ? `${summary.recent_cases.length} 条最近记录` : '暂无记录' }}</span></div><button class="link-button" @click="router.push('/cases')">查看全部 <el-icon><ArrowRight /></el-icon></button></div>
      <div v-if="summary.recent_cases.length" class="table-wrap">
        <table class="data-table"><thead><tr><th>{{ isPatient ? '检查' : '病例' }}</th><th>检查类型</th><th>状态</th><th>更新时间</th></tr></thead>
          <tbody><tr v-for="item in summary.recent_cases" :key="item.id" class="clickable-row" @click="router.push(`/cases/${item.id}`)"><td><span class="case-title">{{ item.title }}</span><span class="case-sub">{{ item.id.slice(0, 8) }}</span></td><td>胸部 X 光</td><td><span class="status-tag" :class="item.status">{{ statusText[item.status] || item.status }}</span></td><td>{{ new Date(item.updated_at).toLocaleDateString('zh-CN') }}</td></tr></tbody>
        </table>
      </div>
      <div v-else class="empty-state">{{ isPatient ? '还没有检查记录。' : '还没有病例，从创建第一条病例开始。' }}</div>
    </section>

    <section v-if="!isPatient" class="panel">
      <div class="panel-header"><div><h3>风险分布</h3><span>当前病例</span></div><span class="panel-header-mark"><el-icon><Warning /></el-icon></span></div>
      <div class="panel-body risk-stack">
        <div class="risk-row"><i class="risk-bar"><span class="risk-high" :style="{ width: `${summary.total_cases ? Math.min(100, summary.high_priority / summary.total_cases * 100) : 0}%` }"></span></i><span>高优先级复核</span><strong>{{ summary.high_priority }}</strong></div>
        <div class="risk-row"><i class="risk-bar"><span class="risk-medium" :style="{ width: `${summary.total_cases ? Math.min(100, summary.pending_review / summary.total_cases * 100) : 0}%` }"></span></i><span>待审核病例</span><strong>{{ summary.pending_review }}</strong></div>
        <div class="risk-row"><i class="risk-bar"><span class="risk-low" :style="{ width: `${summary.total_cases ? Math.max(20, 100 - summary.high_priority / Math.max(summary.total_cases, 1) * 100) : 0}%` }"></span></i><span>其余病例</span><strong>{{ Math.max(0, summary.total_cases - summary.high_priority) }}</strong></div>
      <div class="analysis-box" style="margin-top: 10px"><h4><el-icon><CircleCheck /></el-icon> 当前工作流</h4><span>身份核验 → 上传校验 → 影像分析 → 医生发布 → 患者说明与随访</span></div>
      </div>
    </section>

    <section v-else class="panel patient-next-panel">
      <div class="panel-header"><div><h3>你的下一步</h3><span>每一项都以审核后的信息为准</span></div><span class="panel-header-mark"><el-icon><CircleCheck /></el-icon></span></div>
      <div class="patient-next-list">
        <div class="patient-next-item"><span>01</span><div><strong>查看审核后的检查说明</strong><small>报告发布后，可以在“我的检查”中查看。</small></div></div>
        <div class="patient-next-item"><span>02</span><div><strong>按照随访任务完成下一步</strong><small>{{ summary.active_followups ? `目前有 ${summary.active_followups} 项任务待处理。` : '暂无进行中的随访任务。' }}</small></div></div>
        <div class="patient-next-item"><span>03</span><div><strong>有疑问时和专业人员确认</strong><small>平台内容用于辅助理解，不能替代面对面诊疗。</small></div></div>
      </div>
    </section>
  </div>
</template>
