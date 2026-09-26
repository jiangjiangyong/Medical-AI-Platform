<script setup>
import { computed, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import {
  Bell, Calendar, DataAnalysis, Document, Fold, House, Monitor, Setting, Expand,
} from '@element-plus/icons-vue'

const route = useRoute()
const router = useRouter()
const collapsed = ref(false)
const mobileMenuOpen = ref(false)
const user = ref(readUser())

function readUser() {
  const raw = localStorage.getItem('medical_platform_user')
  return raw ? JSON.parse(raw) : null
}

watch(() => route.fullPath, () => {
  user.value = readUser()
  mobileMenuOpen.value = false
})

const isLogin = computed(() => route.name === 'login')
const roleLabel = computed(() => ({ admin: '系统管理员', doctor: '医生端', health_manager: '健康管理师', patient: '患者端' }[user.value?.role] || '访客'))
const pageTitle = computed(() => {
  if (route.name === 'dashboard') {
    if (user.value?.role === 'admin') return '系统概览'
    if (user.value?.role === 'patient') return '健康概览'
    return '工作台'
  }
  return route.meta.title || ({ cases: '病例中心', 'case-detail': '病例详情', 'follow-ups': '随访任务', admin: '系统管理' }[route.name] || '影像决策台')
})
const navItems = computed(() => {
  const patient = user.value?.role === 'patient'
  const items = [
    { path: '/dashboard', label: patient ? '健康概览' : user.value?.role === 'admin' ? '系统概览' : '工作台', icon: House },
    { path: '/cases', label: patient ? '我的检查' : '病例中心', icon: Document },
    { path: '/follow-ups', label: patient ? '我的随访' : '随访任务', icon: Calendar },
  ]
  if (user.value?.role === 'admin') items.push({ path: '/admin', label: '系统管理', icon: Setting })
  return items
})

function logout() {
  localStorage.removeItem('medical_platform_token')
  localStorage.removeItem('medical_platform_user')
  user.value = null
  router.push('/login')
}

function openMobileMenu() {
  collapsed.value = false
  mobileMenuOpen.value = true
}
</script>

<template>
  <div v-if="isLogin" class="auth-frame">
    <router-view @authenticated="user = readUser()" />
  </div>
  <div v-else class="app-shell">
    <button v-if="mobileMenuOpen" class="mobile-nav-backdrop" type="button" aria-label="关闭导航" @click="mobileMenuOpen = false"></button>
    <aside class="side-panel" :class="{ collapsed, 'mobile-open': mobileMenuOpen }">
      <div class="brand-block">
        <div class="brand-mark"><DataAnalysis /></div>
        <div v-if="!collapsed" class="brand-copy">
          <strong>影像决策台</strong>
          <span>Medical Insight</span>
        </div>
      </div>
      <div v-if="!collapsed" class="workspace-label">工作空间</div>
      <nav class="nav-list">
        <router-link v-for="item in navItems" :key="item.path" :to="item.path" class="nav-item" @click="mobileMenuOpen = false">
          <el-icon><component :is="item.icon" /></el-icon>
          <span v-if="!collapsed">{{ item.label }}</span>
        </router-link>
      </nav>
      <div class="side-bottom">
        <div v-if="!collapsed" class="trust-note"><span class="pulse-dot"></span>辅助决策服务在线</div>
        <button class="collapse-button" type="button" :title="collapsed ? '展开导航' : '收起导航'" @click="collapsed = !collapsed">
          <el-icon><component :is="collapsed ? Expand : Fold" /></el-icon>
          <span v-if="!collapsed">收起导航</span>
        </button>
      </div>
    </aside>
    <main class="main-panel">
      <header class="top-bar">
        <div class="top-bar-main">
          <button class="mobile-menu-button" type="button" title="打开导航" aria-label="打开导航" @click="openMobileMenu"><el-icon><Expand /></el-icon></button>
          <div>
          <span class="eyebrow">{{ roleLabel }}</span>
          <h1>{{ pageTitle }}</h1>
          </div>
        </div>
        <div class="top-actions">
          <button class="icon-button" title="查看待办随访" @click="router.push('/follow-ups')"><el-icon><Bell /></el-icon><span class="notification-dot"></span></button>
          <div class="user-chip">
            <div class="avatar">{{ user?.display_name?.slice(0, 1) || 'U' }}</div>
            <div class="user-meta"><strong>{{ user?.display_name || '未登录' }}</strong><span>{{ user?.email }}</span></div>
          </div>
          <button class="logout-button" type="button" @click="logout">退出</button>
        </div>
      </header>
      <div class="content-area">
        <router-view />
      </div>
    </main>
  </div>
</template>
