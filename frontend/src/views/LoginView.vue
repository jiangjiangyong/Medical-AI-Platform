<script setup>
import { reactive, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { DataAnalysis, Lock, User } from '@element-plus/icons-vue'
import { authApi } from '../services/api'

const router = useRouter()
const loading = ref(false)
const form = reactive({ email: 'doctor@medical.local', password: 'Doctor123!' })
const demoAccounts = [
  { label: '医生端', email: 'doctor@medical.local', password: 'Doctor123!' },
  { label: '患者端', email: 'patient@medical.local', password: 'Patient123!' },
  { label: '管理员', email: 'admin@medical.local', password: 'Admin123!' },
  { label: '健康管理', email: 'manager@medical.local', password: 'Manager123!' },
]

function selectDemo(account) {
  form.email = account.email
  form.password = account.password
}

async function submit() {
  loading.value = true
  try {
    const { data } = await authApi.login(form)
    localStorage.setItem('medical_platform_token', data.access_token)
    localStorage.setItem('medical_platform_user', JSON.stringify(data.user))
    ElMessage.success('已进入工作空间')
    router.push('/dashboard')
  } catch (error) {
    ElMessage.error(error.response?.data?.detail || '登录失败，请检查后端服务')
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="login-page">
    <section class="login-visual">
      <div class="login-brand">
        <div class="brand-mark"><el-icon><DataAnalysis /></el-icon></div>
        <div class="brand-copy"><strong>影像决策台</strong><span>Medical Insight</span></div>
      </div>
      <div class="login-kicker">
        <h1>让每一次影像复核，都有迹可循。</h1>
        <p>围绕病例、证据与随访任务组织医学影像辅助决策，让医生更快审阅，让患者更清楚地理解下一步。</p>
      </div>
      <div class="login-signal">
        <span><i class="pulse-dot"></i>本地数据工作区</span>
        <span>AI 草稿 · 人工审核</span>
      </div>
    </section>
    <section class="login-panel">
      <div class="login-card">
        <h2>欢迎回来</h2>
        <p>登录你的医疗辅助决策工作空间。</p>
        <el-form :model="form" @submit.prevent="submit">
          <el-form-item>
            <el-input v-model="form.email" placeholder="邮箱地址" size="large">
              <template #prefix><el-icon><User /></el-icon></template>
            </el-input>
          </el-form-item>
          <el-form-item>
            <el-input v-model="form.password" placeholder="密码" type="password" size="large" show-password @keyup.enter="submit">
              <template #prefix><el-icon><Lock /></el-icon></template>
            </el-input>
          </el-form-item>
          <button class="login-submit" type="submit" :disabled="loading">{{ loading ? '正在验证…' : '进入工作台' }}</button>
        </el-form>
        <div class="demo-accounts">
          <strong>快速进入演示环境</strong>
          <div class="demo-account-grid">
            <button v-for="account in demoAccounts" :key="account.email" type="button" class="demo-account-button" @click="selectDemo(account)">
              <span>{{ account.label }}</span><small>{{ account.email }}</small>
            </button>
          </div>
        </div>
      </div>
    </section>
  </div>
</template>
