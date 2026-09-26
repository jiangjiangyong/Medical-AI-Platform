<script setup>
import { computed, onMounted, reactive, ref } from 'vue'
import { DataAnalysis, Document, Refresh, Upload, User } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import { adminApi } from '../services/api'

const overview = ref({ users: 0, cases: 0, followups: 0, knowledge_chunks: 0, vision_adapter: 'mock', vision_target: '' })
const auditLogs = ref([])
const loading = ref(true)
const indexing = ref(false)
const form = reactive({ source_name: 'admin-note.md', content: '' })
const lastIndexResult = ref('')
const actionText = {
  'case.created': '创建病例',
  'study.uploaded': '归档资料',
  'case.analyzed': '完成辅助分析',
  'report.reviewed': '审核报告',
  'followup.updated': '更新随访任务',
}
const serviceCards = computed(() => [
  { title: '视觉分析', value: overview.value.vision_adapter === 'mock' ? '模拟适配器' : overview.value.vision_adapter, note: '等待本地医学模型接入', icon: DataAnalysis, tone: 'amber' },
  { title: '文本报告', value: 'DeepSeek API', note: '专业报告与患者解释', icon: Document, tone: 'teal' },
  { title: '知识检索', value: 'BGE 小块向量', note: '约 280 字符切分', icon: Refresh, tone: 'blue' },
  { title: '账号体系', value: `${overview.value.users} 个账号`, note: '医生、患者和管理角色', icon: User, tone: 'slate' },
])

async function load() {
  loading.value = true
  try {
    const [overviewResponse, auditResponse] = await Promise.all([adminApi.overview(), adminApi.auditLogs()])
    overview.value = overviewResponse.data
    auditLogs.value = auditResponse.data
  } catch (error) { ElMessage.error(error.response?.data?.detail || '管理数据加载失败') } finally { loading.value = false }
}

async function indexDefaults() {
  indexing.value = true
  try {
    const { data } = await adminApi.indexDefaults()
    lastIndexResult.value = `已处理 ${data.items?.length || 0} 份默认资料`
    ElMessage.success('默认知识已重新索引')
    await load()
  } catch (error) { ElMessage.error(error.response?.data?.detail || '索引失败') } finally { indexing.value = false }
}

async function indexCustom() {
  if (!form.source_name.trim()) return ElMessage.warning('请填写资料来源名称')
  if (!form.content.trim()) return ElMessage.warning('请输入知识内容')
  indexing.value = true
  try {
    const { data } = await adminApi.index({ source_name: form.source_name, content: form.content })
    lastIndexResult.value = `已切分 ${data.chunks} 个知识块，${data.embedding_ready} 个向量就绪`
    ElMessage.success('知识片段已写入索引')
    form.content = ''
    await load()
  } catch (error) { ElMessage.error(error.response?.data?.detail || '写入失败') } finally { indexing.value = false }
}

function formatAction(action) { return actionText[action] || action }
function formatTime(value) { return value ? new Date(value).toLocaleString('zh-CN', { month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : '' }
onMounted(load)
</script>

<template>
  <div class="page-heading"><div><span class="section-kicker">CONTROL CENTER</span><h2>系统管理</h2><p>在这里确认平台状态、维护知识库，并查看最近的关键操作。</p></div><button class="soft-action" @click="load"><el-icon><Refresh /></el-icon>刷新数据</button></div>

  <div v-loading="loading" class="admin-status-grid">
    <div v-for="service in serviceCards" :key="service.title" class="admin-status-card"><span class="admin-status-icon" :class="service.tone"><el-icon><component :is="service.icon" /></el-icon></span><div><strong>{{ service.title }}</strong><span>{{ service.value }}</span><small>{{ service.note }}</small></div></div>
  </div>

  <div class="admin-grid" v-loading="loading">
    <section class="panel"><div class="panel-header"><div><h3>运行概览</h3><span>当前开发环境</span></div><span class="panel-header-mark"><el-icon><DataAnalysis /></el-icon></span></div><div class="panel-body"><div class="admin-stat-list"><div class="admin-stat"><span>用户总数</span><strong>{{ overview.users }}</strong></div><div class="admin-stat"><span>病例总数</span><strong>{{ overview.cases }}</strong></div><div class="admin-stat"><span>随访任务</span><strong>{{ overview.followups }}</strong></div><div class="admin-stat"><span>知识块</span><strong>{{ overview.knowledge_chunks }}</strong></div></div><div class="admin-guide"><div><span>01</span><p>先维护知识资料</p></div><div><span>02</span><p>再运行病例分析</p></div><div><span>03</span><p>最后查看操作记录</p></div></div></div></section>
    <section class="panel"><div class="panel-header"><div><h3>知识库维护</h3><span>小块切分后再调用向量模型</span></div><span class="panel-header-mark"><el-icon><Upload /></el-icon></span></div><div class="panel-body"><div class="knowledge-actions"><button class="primary-action" :disabled="indexing" @click="indexDefaults"><el-icon><Refresh /></el-icon>{{ indexing ? '索引中…' : '索引默认知识' }}</button><span>重新建立项目内置医学资料索引</span></div><div class="knowledge-form"><input v-model="form.source_name" placeholder="来源名称，例如 guideline-note.md" /><textarea v-model="form.content" maxlength="100000" placeholder="粘贴一段需要加入知识库的医学资料"></textarea><div class="knowledge-form-foot"><span>{{ form.content.length }} / 100000 字符</span><button class="soft-action" :disabled="indexing" @click="indexCustom"><el-icon><Upload /></el-icon>写入知识库</button></div></div><div v-if="lastIndexResult" class="index-result"><el-icon><DataAnalysis /></el-icon>{{ lastIndexResult }}</div></div></section>
  </div>

  <section class="panel admin-audit-panel"><div class="panel-header"><div><h3>最近操作</h3><span>用于快速确认病例和知识库发生了什么变化</span></div><span>{{ auditLogs.length }} 条记录</span></div><div v-if="auditLogs.length" class="audit-list"><div v-for="item in auditLogs" :key="item.id" class="audit-row"><span class="audit-action-dot"></span><div><strong>{{ formatAction(item.action) }}</strong><small>{{ item.actor_name }} · {{ item.resource_type }} {{ item.resource_id ? item.resource_id.slice(0, 8) : '' }}</small></div><time>{{ formatTime(item.created_at) }}</time></div></div><div v-else class="empty-state">暂时没有操作记录。</div></section>
</template>
