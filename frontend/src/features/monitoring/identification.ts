/**
 * Tiger identification abstraction.
 *
 * The UI and detection pipeline depend only on the `TigerIdentificationService`
 * interface, never on a concrete implementation. Today we ship a deterministic
 * mock; later a `RealTigerReIDService` (calling the backend Re-ID endpoint) can
 * be dropped in without changing the map or upload UI.
 *
 * IMPORTANT: the mock does NOT pretend to be an accurate model. It returns a
 * plausible identity + confidence purely so the demo pipeline is exercisable.
 */

import { hashString, mulberry32 } from './geo';
import type { DetectionSource } from './types';

export interface IdentificationResult {
  tigerId: string | null; // null → treated as a new / unidentified individual
  confidence: number;
  source: DetectionSource;
}

export interface TigerIdentificationService {
  readonly kind: 'mock' | 'real';
  identifyTiger(image: File, cameraId: string, knownTigerIds: string[]): Promise<IdentificationResult>;
}

/**
 * Deterministic mock: hashes the file name + camera so the same upload yields
 * the same identity, and biases towards tigers whose home range includes the
 * camera when that mapping is supplied.
 */
export class MockTigerIdentificationService implements TigerIdentificationService {
  readonly kind = 'mock' as const;

  constructor(private readonly cameraTigerHints: Map<string, string[]> = new Map()) {}

  async identifyTiger(
    image: File,
    cameraId: string,
    knownTigerIds: string[]
  ): Promise<IdentificationResult> {
    // Simulate a little processing latency.
    await new Promise((r) => setTimeout(r, 400));

    const seed = hashString(`${image.name}:${image.size}:${cameraId}`);
    const rand = mulberry32(seed);

    // Prefer tigers hinted for this camera; fall back to the full roster.
    const hinted = this.cameraTigerHints.get(cameraId) ?? [];
    const pool = hinted.length ? hinted : knownTigerIds;

    // ~10% of the time report an unidentified individual.
    if (pool.length === 0 || rand() < 0.1) {
      return { tigerId: null, confidence: Number((0.4 + rand() * 0.2).toFixed(3)), source: 'manual' };
    }

    const tigerId = pool[Math.floor(rand() * pool.length)];
    const confidence = Number((0.7 + rand() * 0.28).toFixed(3));
    return { tigerId, confidence, source: 'manual' };
  }
}
