/**
 * Leaflet div-icon builders for the monitoring layers. Kept separate from the
 * legacy map's `mapConfig` so the two map implementations don't fight over
 * marker styles.
 */

import L from 'leaflet';

import { MONITORING_COLORS } from './config';
import type { CameraTrap, TrackedTiger } from './types';

const CAMERA_STATUS_COLORS: Record<CameraTrap['status'], string> = {
  active: MONITORING_COLORS.camera,
  maintenance: '#eab308',
  offline: '#ef4444',
};

/** Camera trap: small square "lens" marker, coloured by status. */
export function cameraMarker(camera: CameraTrap, selected = false): L.DivIcon {
  const color = CAMERA_STATUS_COLORS[camera.status];
  const size = selected ? 16 : 12;
  return L.divIcon({
    className: 'pench-mon-marker',
    html: `<span style="display:block;width:${size}px;height:${size}px;border:2px solid #fff;border-radius:3px;background:${color};box-shadow:0 0 0 1px rgba(0,0,0,.35)${
      selected ? ',0 0 0 4px rgba(14,165,233,.35)' : ''
    }"></span>`,
    iconSize: [size, size],
    iconAnchor: [size / 2, size / 2],
    popupAnchor: [0, -size / 2],
  });
}

/** Tiger: teardrop pin, coloured by sex, with the short id label. */
export function tigerMarker(tiger: TrackedTiger, highlighted = false): L.DivIcon {
  const color = tiger.sex === 'female' ? MONITORING_COLORS.tigerFemale : MONITORING_COLORS.tigerMale;
  const scale = highlighted ? 1.25 : 1;
  const w = 22 * scale;
  const h = 30 * scale;
  const label = tiger.id.replace('T-', '');
  return L.divIcon({
    className: 'pench-mon-marker',
    html: `
      <div style="position:relative;width:${w}px;height:${h}px;transform:translateY(-2px)">
        <div style="position:absolute;left:50%;top:0;transform:translateX(-50%);width:${w}px;height:${w}px;background:${color};border:2px solid #fff;border-radius:50% 50% 50% 0;transform:translateX(-50%) rotate(-45deg);box-shadow:0 1px 3px rgba(0,0,0,.5)"></div>
        <span style="position:absolute;left:0;top:${w * 0.16}px;width:${w}px;text-align:center;font:700 ${9 * scale}px/1 system-ui;color:#fff">${label}</span>
      </div>`,
    iconSize: [w, h],
    iconAnchor: [w / 2, h - 2],
    popupAnchor: [0, -h + 6],
  });
}
