export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  size: number;
  pages: number;
}

export type MarkerState = 'active' | 'offline' | 'warning' | 'maintenance' | 'recent_detection';
export type CameraStatus = 'active' | 'inactive' | 'maintenance';
export type CameraZone = 'core' | 'buffer' | 'village_adjacent';
export type ImageStatus = 'pending' | 'triaged' | 'quarantined' | 'deleted' | 'processed';
export type MatchType = 'auto_match' | 'human_verified' | 'new_individual' | 'demo';
export type TigerSex = 'male' | 'female' | 'unknown';
export type TigerStatusValue = 'active' | 'inactive' | 'deceased' | 'unknown';
export type AlertSeverity = 'critical' | 'high' | 'medium' | 'low' | 'info';
export type AlertStatus = 'open' | 'acknowledged' | 'resolved';
export type AlertType =
  | 'high_priority_detection'
  | 'camera_offline'
  | 'unusual_movement'
  | 'high_activity'
  | 'low_confidence';

export interface LabeledCount {
  label: string;
  count: number;
}

export interface CameraHealth {
  active: number;
  offline: number;
  maintenance: number;
  warning: number;
  total: number;
}

export interface RecentIdentification {
  observation_id: string;
  tiger_id: string | null;
  tiger_code: string | null;
  tiger_name: string | null;
  camera_id: string | null;
  camera_name: string | null;
  timestamp: string | null;
  identity_confidence: number | null;
  match_type: MatchType | null;
  image_id: string | null;
  image_url: string | null;
}

export interface AlertPreview {
  alert_id: string;
  alert_type: AlertType;
  severity: AlertSeverity;
  status: AlertStatus;
  title: string;
  message: string;
  camera_id: string | null;
  zone_code: string | null;
  created_at: string | null;
}

export interface DashboardStats {
  total_images: number;
  blank_images: number;
  subject_images: number;
  quarantined_images: number;
  total_tigers: number;
  active_tigers: number;
  total_observations: number;
  detections_last_7_days: number;
  total_cameras: number;
  active_cameras: number;
  pending_reviews: number;
  open_alerts: number;
  storage_saved_bytes: number;
  total_storage_bytes: number;
  mean_identity_confidence: number | null;
  demo_mode: boolean;
  data_source: string;
  camera_health: CameraHealth;
  recent_identifications: RecentIdentification[];
  recent_alerts: AlertPreview[];
  recent_images: Array<{
    image_id: string;
    camera_id: string | null;
    status: ImageStatus | null;
    blank_probability: number | null;
    timestamp: string | null;
    url: string;
  }>;
  images_by_camera: LabeledCount[];
  detections_by_zone: LabeledCount[];
  detection_trend: Array<{ date: string; detections: number; blanks: number }>;
  most_active_tigers: LabeledCount[];
}

export interface CameraStation {
  id: string;
  camera_id: string;
  name: string;
  zone: CameraZone | null;
  zone_code: string | null;
  latitude: number | null;
  longitude: number | null;
  status: CameraStatus | null;
  marker_state: MarkerState;
  battery_percent: number | null;
  last_active_at: string | null;
  last_detection_at: string | null;
  observation_count: number;
  image_count: number;
  open_alert_count: number;
}

export interface CameraRecentDetection {
  observation_id: string;
  timestamp: string | null;
  tiger_code: string | null;
  tiger_name: string | null;
  species: string | null;
  identity_confidence: number | null;
  detection_confidence: number | null;
  image_id: string | null;
}

export interface CameraGalleryItem {
  image_id: string;
  url: string | null;
  timestamp: string | null;
  status: ImageStatus | null;
  blank_probability: number | null;
  tiger_code?: string | null;
  identity_confidence?: number | null;
}

export interface CameraDetail extends CameraStation {
  altitude_m: number | null;
  description: string | null;
  installed_at: string | null;
  unique_tigers: number;
  recent_detections: CameraRecentDetection[];
  recent_images: CameraGalleryItem[];
  detection_timeline: Array<{ date: string; detections: number }>;
}

export interface Tiger {
  id: string;
  tiger_id: string;
  name: string | null;
  sex: TigerSex | null;
  status: TigerStatusValue | null;
  total_observations: number;
  first_seen: string | null;
  last_seen: string | null;
  mean_confidence: number | null;
  camera_count: number;
  is_demo: boolean;
}

export interface TigerCameraUsage {
  camera_id: string;
  camera_name: string | null;
  detections: number;
  latitude: number | null;
  longitude: number | null;
}

export interface TigerProfile extends Tiger {
  estimated_age_years: number | null;
  notes: string | null;
  created_at: string | null;
  zone_distribution: LabeledCount[];
  frequent_cameras: TigerCameraUsage[];
  recent_observations: Array<{
    observation_id: string;
    timestamp: string | null;
    camera_id: string | null;
    camera_name: string | null;
    zone: string | null;
    identity_confidence: number | null;
    latitude: number | null;
    longitude: number | null;
    image_id: string | null;
    image_url: string | null;
  }>;
  detections_by_month: LabeledCount[];
}

export interface Observation {
  id: string;
  observation_id: string;
  tiger_id: string | null;
  tiger_code: string | null;
  tiger_name: string | null;
  image_id: string | null;
  image_code: string | null;
  image_url: string | null;
  camera_id: string | null;
  camera_name: string | null;
  timestamp: string | null;
  latitude: number | null;
  longitude: number | null;
  zone: CameraZone | null;
  species: string | null;
  detection_type: string | null;
  detection_confidence: number | null;
  identity_confidence: number | null;
  match_type: MatchType | null;
  review_status: string | null;
  flank_side: string | null;
  model_version: string | null;
  is_demo: boolean;
}

export interface ImageRecord {
  id: string;
  image_id: string;
  original_filename: string | null;
  camera_id: string | null;
  camera_name: string | null;
  status: ImageStatus | null;
  blank_probability: number | null;
  triage_reason: string | null;
  timestamp: string | null;
  url: string | null;
  species: string | null;
  tiger_code: string | null;
  tiger_name: string | null;
  identity_confidence: number | null;
  is_demo: boolean;
}

export interface ReviewCandidate {
  tiger_id: string;
  tiger_code: string | null;
  tiger_name: string | null;
  score: number | null;
}

export interface ReviewQueueItem {
  id: string;
  review_id: string;
  observation_id: string | null;
  observation_code: string | null;
  camera_id: string | null;
  timestamp: string | null;
  image_id: string | null;
  image_url: string | null;
  status: string | null;
  candidates: ReviewCandidate[];
  review_note: string | null;
  reviewed_by: string | null;
  reviewed_at: string | null;
  created_at: string | null;
}

export interface Alert {
  id: string;
  alert_id: string;
  alert_type: AlertType;
  severity: AlertSeverity;
  status: AlertStatus;
  title: string;
  message: string;
  camera_id: string | null;
  tiger_id: string | null;
  zone_code: string | null;
  latitude: number | null;
  longitude: number | null;
  detail_json: Record<string, unknown> | null;
  is_demo: boolean;
  acknowledged_by: string | null;
  resolved_at: string | null;
  created_at: string | null;
}

export interface AlertSummary {
  open: number;
  acknowledged: number;
  resolved: number;
  critical: number;
  high: number;
  medium: number;
  low: number;
}

export interface GeoJsonGeometry {
  type: string;
  coordinates: number[][][];
}

export interface Zone {
  zone_code: string;
  name: string;
  zone_type: string;
  description: string | null;
  center_latitude: number | null;
  center_longitude: number | null;
  area_km2: number | null;
  style_color: string | null;
  geometry_json: GeoJsonGeometry | null;
  camera_count: number;
  observation_count: number;
  is_demo: boolean;
}

export interface Gate {
  code: string;
  name: string;
  latitude: number;
  longitude: number;
  gate_type: string;
}

export interface MapSighting {
  observation_id: string;
  tiger_code: string | null;
  tiger_name: string | null;
  camera_id: string | null;
  camera_name: string | null;
  timestamp: string | null;
  latitude: number | null;
  longitude: number | null;
  zone: string | null;
  species: string | null;
  detection_type: string | null;
  identity_confidence: number | null;
  detection_confidence: number | null;
  image_id: string | null;
  image_url: string | null;
  is_demo: boolean;
}

export interface MovementLeg {
  from_camera_id: string | null;
  from_camera_name: string | null;
  from_latitude: number | null;
  from_longitude: number | null;
  from_timestamp: string | null;
  to_camera_id: string | null;
  to_camera_name: string | null;
  to_latitude: number | null;
  to_longitude: number | null;
  to_timestamp: string | null;
  distance_km: number;
  hours_elapsed: number;
}

export interface MovementPoint {
  camera_id: string | null;
  camera_name: string | null;
  latitude: number | null;
  longitude: number | null;
  timestamp: string | null;
}

export interface MovementTrack {
  tiger_code: string;
  tiger_name: string | null;
  observations: MovementPoint[];
  legs: MovementLeg[];
  total_distance_km: number;
  sighting_count: number;
}

export interface MapOverview {
  center: [number, number];
  bounds: [[number, number], [number, number]];
  data_source: string;
  disclaimer: string;
  zones: Zone[];
  gates: Gate[];
  cameras: CameraStation[];
  sightings: MapSighting[];
  tracks: MovementTrack[];
}

export interface AnalyticsOverview {
  range_days: number;
  detections_over_time: Array<{ date: string; detections: number; tigers: number; blanks: number }>;
  detections_by_camera: LabeledCount[];
  detections_by_zone: LabeledCount[];
  detections_by_hour: LabeledCount[];
  detections_by_weekday: LabeledCount[];
  species_distribution: LabeledCount[];
  top_tigers: LabeledCount[];
  confidence_distribution: Array<{ range: string; count: number }>;
  movement_frequency: Array<{ from_camera: string; to_camera: string; transitions: number }>;
  camera_activity: LabeledCount[];
  mean_identity_confidence: number | null;
  is_demo_data: boolean;
}

export interface TriageReport {
  run_id: string;
  total_images: number;
  blank_count: number;
  subject_count: number;
  quarantined_count: number;
  storage_saved_bytes: number;
  blanks_by_camera: LabeledCount[];
  quality_distribution: Array<{ range: string; count: number }>;
}

export interface TriageRun {
  id: string;
  run_id: string;
  status: string;
  total_images: number;
  blank_count: number;
  subject_count: number;
  quarantined_count: number;
  storage_saved_bytes: number;
  triage_processing_seconds: number | null;
  started_at: string | null;
  completed_at: string | null;
}

export interface DemoStatus {
  demo_mode: boolean;
  ml_mode: string;
  model_version: string;
  is_demo_inference: boolean;
  disclaimer: string;
  simulation_disclaimer: string;
  geo_data_source: string;
  reid_available: boolean;
  reid_model_version: string | null;
  reid_is_demo: boolean;
  reid_validated: boolean | null;
  reid_known_identities: number | null;
  reid_error: string | null;
}

export interface SimulationEvent {
  image_id: string;
  status: string;
  is_blank: boolean;
  blank_probability: number | null;
  triage_reason: string | null;
  observation_id: string | null;
  tiger_code: string | null;
  identity_confidence: number | null;
  decision: string | null;
  species?: string | null;
  alerts_created: number;
  message: string;
  camera_id: string;
  camera_name: string;
  zone: string | null;
  latitude: number | null;
  longitude: number | null;
  captured_at: string;
  disclaimer: string;
  identity_error?: string | null;
  identity_quality_warnings?: string[];
  gallery_total?: number | null;
}
