<script setup>
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ArrowRight, Calendar, CircleCheck, Clock } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import { followupsApi } from '../services/api'

const router = useRouter()
const tasks = ref([])
const loading = ref(true)
const isPatient = JSON.parse(localStorage.getItem('medical_platform_user') || '{}').role === 'patient'
const statusText = { pending: '待处理', in_progress: '进行中', completed: '已完成', cancelled: '已取消' }
const counts = computed(() => ({
  pending: tasks.value.filter((task) => task.status === 'pending').length,
  inProgress: tasks.value.filter((task) => task.status === 'in_progress').length,
  completed: tasks.value.filter((task) => task.status === 'completed').length,
}))

async function load() { loading.value = true; try { tasks.value = (await followupsApi.list()).data } catch (error) { ElMessage.error(error.response?.data?.detail || '随访任务加载失败') } finally { loading.value = false } }
async function update(task) {
  const status = task.status === 'pending' ? 'in_progress' : 'completed'
  try { await followupsApi.update(task.id, status); task.status = status; ElMessage.success(status === 'completed' ? '任务已完成' : '任务已开始') } catch (error) { ElMessage.error(error.response?.data?.detail || '更新失败') }
}
function actionLabel(task) { return task.status === 'pending' ? '开始处理' : '标记完成' }
onMounted(load)
</script>

<template>
  <div class="page-heading">
    <div><span class="section-kicker">{{ isPatient ? 'MY FOLLOW-UP' : 'FOLLOW-UP' }}</span><h2>{{ isPatient ? '我的随访' : '随访任务' }}</h2><p>{{ isPatient ? '按审核后的建议完成每一步，并及时记录进展。' : '把报告中的下一步建议变成可追踪的任务。' }}</p></div>
    <button class="soft-action" @click="router.push('/cases')"><el-icon><ArrowRight /></el-icon>{{ isPatient ? '查看我的检查' : '回到病例中心' }}</button>
  </div>

  <div v-loading="loading" class="status-summary">
    <div class="status-summary-card"><span class="status-summary-icon pending"><el-icon><Clock /></el-icon></span><div><small>待处理</small><strong>{{ counts.pending }}</strong></div></div>
    <div class="status-summary-card"><span class="status-summary-icon progress"><el-icon><Calendar /></el-icon></span><div><small>进行中</small><strong>{{ counts.inProgress }}</strong></div></div>
    <div class="status-summary-card"><span class="status-summary-icon done"><el-icon><CircleCheck /></el-icon></span><div><small>已完成</small><strong>{{ counts.completed }}</strong></div></div>
  </div>

  <section class="panel">
    <div class="panel-header"><div><h3>任务清单</h3><span>{{ tasks.length }} 项安排</span></div><span class="panel-header-mark"><el-icon><Calendar /></el-icon></span></div>
    <div v-loading="loading" class="table-wrap">
      <table v-if="tasks.length" class="data-table followup-table"><thead><tr><th>任务</th><th>截止时间</th><th>优先级</th><th>状态</th><th>操作</th></tr></thead><tbody><tr v-for="task in tasks" :key="task.id"><td><span class="case-title">{{ task.title }}</span><span class="case-sub">{{ task.description }}</span></td><td>{{ task.due_at ? new Date(task.due_at).toLocaleDateString('zh-CN') : '待确认' }}</td><td><span class="status-tag" :class="task.priority === 'high' ? 'high' : 'created'">{{ task.priority === 'high' ? '高优先级' : '常规' }}</span></td><td><span class="status-tag" :class="task.status">{{ statusText[task.status] || task.status }}</span></td><td><button v-if="task.status !== 'completed' && task.status !== 'cancelled'" class="approve-button" @click="update(task)"><el-icon><CircleCheck /></el-icon>{{ actionLabel(task) }}</button><span v-else class="task-complete-label">{{ statusText[task.status] }}</span></td></tr></tbody></table>
      <div v-if="tasks.length" class="task-mobile-list"><article v-for="task in tasks" :key="`mobile-${task.id}`" class="task-mobile-card"><div class="task-mobile-card-top"><span class="status-tag" :class="task.status">{{ statusText[task.status] || task.status }}</span><span>{{ task.due_at ? new Date(task.due_at).toLocaleDateString('zh-CN') : '待确认' }}</span></div><strong>{{ task.title }}</strong><p>{{ task.description }}</p><div class="task-mobile-card-bottom"><span class="status-tag" :class="task.priority === 'high' ? 'high' : 'created'">{{ task.priority === 'high' ? '高优先级' : '常规' }}</span><button v-if="task.status !== 'completed' && task.status !== 'cancelled'" class="approve-button" @click="update(task)">{{ actionLabel(task) }}</button></div></article></div>
      <div v-if="!loading && !tasks.length" class="empty-state">暂无随访任务。审核通过后，系统会根据风险等级生成下一步安排。</div>
    </div>
  </section>
</template>
