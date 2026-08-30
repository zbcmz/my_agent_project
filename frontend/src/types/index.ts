// 类型定义

export interface Location {
  longitude: number
  latitude: number
}

export interface Attraction {
  name: string
  address: string
  location?: Location
  visit_duration: number
  description: string
  category?: string
  rating?: number
  image_url?: string
  ticket_price?: number
}

export interface Meal {
  type: 'breakfast' | 'lunch' | 'dinner' | 'snack'
  name: string
  address?: string
  location?: Location
  description?: string
  cuisine?: string
  rating?: number
  avg_cost?: number
  distance?: string
  poi_id?: string
  source?: 'nearby' | 'popular'
  estimated_cost?: number
}

export interface Hotel {
  name: string
  address: string
  location?: Location
  price_range: string
  rating: string
  distance: string
  type: string
  estimated_cost?: number
}

export interface Budget {
  total_attractions: number
  total_hotels: number
  total_meals: number
  total_transportation: number
  total: number
}

export interface RouteSegment {
  from_name: string
  to_name: string
  distance: string
  duration: string
  mode: string
  detail: string
}

export interface DayPlan {
  date: string
  day_index: number
  description: string
  transportation: string
  accommodation: string
  hotel?: Hotel
  attractions: Attraction[]
  meals: Meal[]
  route_segments: RouteSegment[]
}

export interface WeatherInfo {
  date: string
  day_weather: string | null
  night_weather: string | null
  day_temp: number | null
  night_temp: number | null
  wind_direction: string | null
  wind_power: string | null
}

export interface TripPlan {
  city: string
  start_date: string
  end_date: string
  days: DayPlan[]
  weather_info: WeatherInfo[]
  overall_suggestions: string
  budget?: Budget
}

export interface TripFormData {
  city: string
  start_date: string
  end_date: string
  travel_days: number
  transportation: string
  accommodation: string
  preferences: string[]
  food_preference: string
  free_text_input: string
}

// ------------------------------------------------------------
// Legacy API
// ------------------------------------------------------------

export interface TripPlanResponse {
  success: boolean
  message: string
  data?: TripPlan
}

// ------------------------------------------------------------
// Stateful Agent API
// ------------------------------------------------------------

export type AgentStatus =
  | 'completed'
  | 'waiting_human'
  | 'failed'

export interface AgentViolation {
  code?: string
  message?: string
  severity?: 'hard' | 'soft' | string
  affected_days?: number[]
  [key: string]: unknown
}

export interface AgentInterrupt {
  type?: string
  message?: string
  trip_plan?: TripPlan | null
  violations?: AgentViolation[]
  revision_count?: number
  plan_version?: number
  [key: string]: unknown
}


export interface StatefulPlanRequest {
  request: TripFormData
  user_id: string
  thread_id?: string
  constraints?: Record<string, unknown> | null
  enable_human_review: boolean
}

export interface HITLDecision {
  action: 'approve' | 'edit'
  feedback?: string
}

export interface TripEditRequest {
  feedback: string
  enable_human_review: boolean
}

export interface AgentRunResponse {
  success: boolean
  status: AgentStatus
  message: string
  thread_id: string
  data?: TripPlan | null
  interrupt?: AgentInterrupt | null
  constraints?: Record<string, unknown> | null
  violations: AgentViolation[]
  revision_count: number
  plan_version: number
  supervisor?: Record<string, unknown> | null
}


export interface ThreadStateResponse {
  values: {
    trip_plan?: TripPlan
    request?: TripFormData
    user_id?: string
    revision_count?: number
    plan_version?: number
    violations?: AgentViolation[]
    human_review_enabled?: boolean
    [key: string]: unknown
  }
  next: string[]
}


