import axios from 'axios'

const api = axios.create({
  // Same-origin requests work through the Vite dev proxy and through a
  // reverse proxy in a remote deployment. Override this for a standalone
  // static build with VITE_API_BASE_URL.
  baseURL: import.meta.env.VITE_API_BASE_URL || '/api/v1',
  timeout: 60000,
})

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('medical_platform_token')
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('medical_platform_token')
      localStorage.removeItem('medical_platform_user')
      window.location.href = '/login'
    }
    return Promise.reject(error)
  },
)

export const authApi = {
  login: (payload) => api.post('/auth/login', payload),
  register: (payload) => api.post('/auth/register', payload),
  me: () => api.get('/auth/me'),
}

export const dashboardApi = {
  summary: () => api.get('/dashboard/summary'),
}

export const casesApi = {
  list: () => api.get('/cases'),
  patients: () => api.get('/cases/patients'),
  create: (payload) => api.post('/cases', payload),
  detail: (id) => api.get(`/cases/${id}`),
  verifyIdentity: (id, payload) => api.post(`/cases/${id}/verify-identity`, payload),
  upload: (id, file) => {
    const form = new FormData()
    form.append('file', file)
    return api.post(`/cases/${id}/studies`, form, { headers: { 'Content-Type': 'multipart/form-data' } })
  },
  analyze: (id, scenario = 'opacity') => api.post(`/cases/${id}/analyze`, { scenario }),
  editReport: (caseId, reportId, payload) => api.patch(`/cases/${caseId}/reports/${reportId}`, payload),
  review: (caseId, reportId, payload) => api.post(`/cases/${caseId}/reports/${reportId}/review`, payload),
  revisions: (caseId, reportId) => api.get(`/cases/${caseId}/reports/${reportId}/revisions`),
  studyUrl: (studyId) => `${api.defaults.baseURL}/cases/studies/${studyId}/file`,
}

export const followupsApi = {
  list: () => api.get('/follow-ups'),
  update: (id, status) => api.patch(`/follow-ups/${id}`, { status }),
}

export const adminApi = {
  overview: () => api.get('/admin/overview'),
  auditLogs: () => api.get('/admin/audit-logs'),
  indexDefaults: () => api.post('/admin/knowledge/index-defaults'),
  index: (payload) => api.post('/admin/knowledge/index', payload),
}

export default api
