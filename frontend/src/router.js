import { createRouter, createWebHistory } from 'vue-router'
import LoginView from './views/LoginView.vue'
import DashboardView from './views/DashboardView.vue'
import CasesView from './views/CasesView.vue'
import CaseDetailView from './views/CaseDetailView.vue'
import FollowUpsView from './views/FollowUpsView.vue'
import AdminView from './views/AdminView.vue'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/login', name: 'login', component: LoginView },
    { path: '/', redirect: '/dashboard' },
    { path: '/dashboard', name: 'dashboard', component: DashboardView, meta: { auth: true } },
    { path: '/cases', name: 'cases', component: CasesView, meta: { auth: true } },
    { path: '/cases/:id', name: 'case-detail', component: CaseDetailView, meta: { auth: true } },
    { path: '/follow-ups', name: 'follow-ups', component: FollowUpsView, meta: { auth: true } },
    { path: '/admin', name: 'admin', component: AdminView, meta: { auth: true, roles: ['admin'] } },
  ],
})

router.beforeEach((to) => {
  const token = localStorage.getItem('medical_platform_token')
  const rawUser = localStorage.getItem('medical_platform_user')
  const user = rawUser ? JSON.parse(rawUser) : null
  if (to.meta.auth && !token) return '/login'
  if (to.name === 'login' && token) return '/dashboard'
  if (to.meta.roles && (!user || !to.meta.roles.includes(user.role))) return '/dashboard'
})

export default router

