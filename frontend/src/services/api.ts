import axios from 'axios'
import type {
  TripFormData,
  TripPlanResponse,
  TripPlan,
  StatefulPlanRequest,
  HITLDecision,
  TripEditRequest,
  AgentRunResponse,
  ThreadStateResponse
} from '@/types'

export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || ''

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 600000,
  headers: {
    'Content-Type': 'application/json'
  }
})

apiClient.interceptors.request.use(
  (config) => {
    console.log(
      '发送请求:',
      config.method?.toUpperCase(),
      config.url
    )
    return config
  },
  (error) => {
    console.error('请求错误:', error)
    return Promise.reject(error)
  }
)

apiClient.interceptors.response.use(
  (response) => {
    console.log(
      '收到响应:',
      response.status,
      response.config.url
    )
    return response
  },
  (error) => {
    console.error(
      '响应错误:',
      error.response?.status,
      error.message
    )
    return Promise.reject(error)
  }
)

function getApiErrorMessage(
  error: any,
  fallback: string
): string {
  const detail = error?.response?.data?.detail

  if (typeof detail === 'string') {
    return detail
  }

  if (Array.isArray(detail)) {
    return detail
      .map((item: any) => item?.msg || JSON.stringify(item))
      .join('; ')
  }

  return error?.message || fallback
}

// ============================================================
// Stateful Agent API
// ============================================================

export async function generateStatefulTripPlan(
  payload: StatefulPlanRequest
): Promise<AgentRunResponse> {
  try {
    const response = await apiClient.post<AgentRunResponse>(
      '/api/trip-agent/plan',
      payload
    )

    return response.data
  } catch (error: any) {
    console.error('Stateful旅行规划失败:', error)

    throw new Error(
      getApiErrorMessage(
        error,
        '生成旅行计划失败'
      )
    )
  }
}

export async function resumeTrip(
  threadId: string,
  decision: HITLDecision
): Promise<AgentRunResponse> {
  try {
    const response = await apiClient.post<AgentRunResponse>(
      `/api/trip-agent/resume/${encodeURIComponent(threadId)}`,
      decision
    )

    return response.data
  } catch (error: any) {
    console.error('恢复HITL失败:', error)

    throw new Error(
      getApiErrorMessage(
        error,
        '恢复旅行规划失败'
      )
    )
  }
}

export async function editTrip(
  threadId: string,
  payload: TripEditRequest
): Promise<AgentRunResponse> {
  try {
    const response = await apiClient.post<AgentRunResponse>(
      `/api/trip-agent/edit/${encodeURIComponent(threadId)}`,
      payload
    )

    return response.data
  } catch (error: any) {
    console.error('增量修改旅行计划失败:', error)

    throw new Error(
      getApiErrorMessage(
        error,
        '修改旅行计划失败'
      )
    )
  }
}

export async function getThreadState(
  threadId: string
): Promise<ThreadStateResponse> {
  try {
    const response = await apiClient.get<ThreadStateResponse>(
      `/api/trip-agent/thread/${encodeURIComponent(threadId)}`
    )

    return response.data
  } catch (error: any) {
    console.error('读取Thread状态失败:', error)

    throw new Error(
      getApiErrorMessage(
        error,
        '读取旅行线程失败'
      )
    )
  }
}

// 后续 Result.vue 获取景点图片统一使用这个函数，
// 避免请求错误地发到 localhost:5173。
export async function getPoiPhoto(
  name: string
): Promise<any> {
  try {
    const response = await apiClient.get(
      '/api/poi/photo',
      {
        params: { name }
      }
    )

    return response.data
  } catch (error: any) {
    console.error(`获取${name}图片失败:`, error)
    throw error
  }
}

// ============================================================
// Legacy API
// 暂时保留，Home.vue 切换完成后再决定是否删除
// ============================================================

export async function generateTripPlan(
  formData: TripFormData
): Promise<TripPlanResponse> {
  try {
    const response = await apiClient.post<TripPlanResponse>(
      '/api/trip/plan',
      formData
    )

    return response.data
  } catch (error: any) {
    console.error('生成旅行计划失败:', error)

    throw new Error(
      getApiErrorMessage(
        error,
        '生成旅行计划失败'
      )
    )
  }
}

export interface StreamEvent {
  type: 'init' | 'node_complete' | 'complete' | 'error'
  message: string
  progress: number
  node?: string
  data?: TripPlan
}

export interface StreamOptions {
  timeout?: number
  signal?: AbortSignal
}

export async function generateTripPlanStream(
  formData: TripFormData,
  onEvent: (event: StreamEvent) => void,
  options?: StreamOptions
): Promise<void> {
  const timeout = options?.timeout || 180000
  const controller = new AbortController()
  const timeoutId = setTimeout(
    () => controller.abort(),
    timeout
  )

  const signal = options?.signal
    ? AbortSignal.any([
        options.signal,
        controller.signal
      ])
    : controller.signal

  let response: Response

  try {
    response = await fetch(
      `${API_BASE_URL}/api/trip/plan/stream`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(formData),
        signal
      }
    )
  } catch (error: any) {
    clearTimeout(timeoutId)

    if (error.name === 'AbortError') {
      throw new Error('请求已取消或超时')
    }

    throw error
  }

  clearTimeout(timeoutId)

  if (!response.ok) {
    throw new Error(`请求失败: ${response.status}`)
  }

  const reader = response.body?.getReader()

  if (!reader) {
    throw new Error('无法获取响应流')
  }

  const decoder = new TextDecoder()
  let buffer = ''

  try {
    while (true) {
      const { done, value } = await reader.read()

      if (done) {
        break
      }

      buffer += decoder.decode(
        value,
        { stream: true }
      )

      const lines = buffer.split('\n')
      buffer = lines.pop() || ''

      for (const line of lines) {
        const trimmed = line.trim()

        if (!trimmed.startsWith('data: ')) {
          continue
        }

        try {
          const event: StreamEvent =
            JSON.parse(trimmed.slice(6))

          onEvent(event)

          if (
            event.type === 'complete' ||
            event.type === 'error'
          ) {
            return
          }
        } catch (error) {
          console.warn(
            '解析SSE事件失败:',
            trimmed,
            error
          )
        }
      }
    }
  } finally {
    reader.cancel().catch(() => {})
  }
}

export async function healthCheck(): Promise<any> {
  try {
    const response = await apiClient.get('/health')
    return response.data
  } catch (error: any) {
    console.error('健康检查失败:', error)

    throw new Error(
      error.message || '健康检查失败'
    )
  }
}

export default apiClient

