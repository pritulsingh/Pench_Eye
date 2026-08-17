import axios from 'axios';
import type {
  Alert,
  AlertSummary,
  AnalyticsOverview,
  CameraDetail,
  CameraStation,
  DashboardStats,
  DemoStatus,
  ImageRecord,
  MapOverview,
  MapSighting,
  MovementTrack,
  Observation,
  PaginatedResponse,
  ReviewQueueItem,
  SimulationEvent,
  Tiger,
  TigerProfile,
  TriageReport,
  TriageRun,
  Zone,
} from '@/types';

const BASE_URL = import.meta.env.VITE_API_URL || '';

export const API_ORIGIN = BASE_URL;

export const api = axios.create({
  baseURL: `${BASE_URL}/api/v1`,
  timeout: 60000,
  headers: { 'Content-Type': 'application/json' },
});

/** Absolute URL for a backend-relative asset path (e.g. an image URL). */
export const assetUrl = (path?: string | null): string | undefined =>
  path ? `${BASE_URL}${path}` : undefined;

api.interceptors.response.use(
  (response) => response,
  (error) => {
    const detail = error.response?.data?.detail;
    error.userMessage =
      (typeof detail === 'string' && detail) || error.message || 'Request failed';
    return Promise.reject(error);
  }
);

export const dashboardApi = {
  getStats: () => api.get<DashboardStats>('/dashboard/stats'),
};

export const imagesApi = {
  upload: (file: File, cameraId?: string) => {
    const formData = new FormData();
    formData.append('file', file);
    if (cameraId) formData.append('camera_id', cameraId);
    return api.post('/images/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },
  batch: (files: File[], cameraId?: string) => {
    const formData = new FormData();
    files.forEach((f) => formData.append('files', f));
    if (cameraId) formData.append('camera_id', cameraId);
    return api.post('/images/batch', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },
  list: (params?: Record<string, unknown>) =>
    api.get<PaginatedResponse<ImageRecord>>('/images', { params }),
  get: (id: string) => api.get<ImageRecord>(`/images/${id}`),
  restore: (id: string) => api.post(`/images/${id}/restore`),
  delete: (id: string) => api.post(`/images/${id}/delete`),
};

export const triageApi = {
  runTriage: (data: { batch_id?: string; camera_id?: string } = {}) =>
    api.post<TriageRun>('/triage/run', data),
  getRuns: () => api.get<TriageRun[]>('/triage/runs'),
  getRun: (id: string) => api.get<TriageRun>(`/triage/runs/${id}`),
  getReport: () => api.get<TriageReport>('/triage/report'),
  getQuarantine: (params?: Record<string, unknown>) =>
    api.get<ImageRecord[]>('/triage/quarantine', { params }),
};

export const tigersApi = {
  list: (params?: Record<string, unknown>) =>
    api.get<PaginatedResponse<Tiger>>('/tigers', { params }),
  get: (code: string) => api.get<TigerProfile>(`/tigers/${code}`),
  getObservations: (code: string, params?: Record<string, unknown>) =>
    api.get<PaginatedResponse<Observation>>(`/tigers/${code}/observations`, { params }),
  getGallery: (code: string) => api.get<ImageRecord[]>(`/tigers/${code}/gallery`),
};

export const observationsApi = {
  list: (params?: Record<string, unknown>) =>
    api.get<PaginatedResponse<Observation>>('/observations', { params }),
  get: (id: string) => api.get<Observation>(`/observations/${id}`),
};

export const reviewsApi = {
  list: (params?: Record<string, unknown>) =>
    api.get<PaginatedResponse<ReviewQueueItem>>('/reviews', { params }),
  get: (id: string) => api.get<ReviewQueueItem>(`/reviews/${id}`),
  approve: (id: string, data: { tiger_id: string; reviewer: string; note?: string }) =>
    api.post(`/reviews/${id}/approve`, data),
  reject: (id: string, data: { reviewer: string; note?: string }) =>
    api.post(`/reviews/${id}/reject`, data),
  newTiger: (id: string, data: { reviewer: string; note?: string }) =>
    api.post(`/reviews/${id}/new-tiger`, data),
};

export const camerasApi = {
  list: (params?: Record<string, unknown>) =>
    api.get<PaginatedResponse<CameraStation>>('/cameras', { params }),
  get: (id: string) => api.get<CameraDetail>(`/cameras/${id}`),
  update: (
    id: string,
    data: Partial<Pick<CameraStation, 'name' | 'zone' | 'zone_code' | 'status' | 'latitude' | 'longitude'>> & {
      description?: string | null;
      altitude_m?: number | null;
    }
  ) => api.patch<CameraDetail>(`/cameras/${id}`, data),
  getObservations: (id: string, params?: Record<string, unknown>) =>
    api.get<PaginatedResponse<Observation>>(`/cameras/${id}/observations`, { params }),
};

export const mapApi = {
  overview: (params?: Record<string, unknown>) =>
    api.get<MapOverview>('/map/overview', { params }),
  zones: () => api.get<Zone[]>('/map/zones'),
  sightings: (params?: Record<string, unknown>) =>
    api.get<MapSighting[]>('/map/sightings', { params }),
  movement: (params?: Record<string, unknown>) =>
    api.get<MovementTrack[]>('/map/movement', { params }),
};

export const alertsApi = {
  list: (params?: Record<string, unknown>) =>
    api.get<PaginatedResponse<Alert>>('/alerts', { params }),
  summary: () => api.get<AlertSummary>('/alerts/summary'),
  evaluate: () => api.post('/alerts/evaluate'),
  updateStatus: (alertId: string, status: string, actor?: string) =>
    api.patch<Alert>(`/alerts/${alertId}`, { status, actor }),
};

export const analyticsApi = {
  overview: (days = 90) => api.get<AnalyticsOverview>('/analytics/overview', { params: { days } }),
};

export const demoApi = {
  status: () => api.get<DemoStatus>('/demo/status'),
  simulate: (data: { camera_id?: string; count?: number } = {}) =>
    api.post<{ events: SimulationEvent[]; disclaimer: string }>('/demo/simulate', data),
};
