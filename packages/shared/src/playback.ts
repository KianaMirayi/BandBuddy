import type { PracticeState } from './domain.js'

export type LoopState = Pick<PracticeState, 'loopStartMs' | 'loopEndMs' | 'loopEnabled'>
export type LoopPhase = 'off' | 'start' | 'active'
export const MIN_LOOP_DURATION_MS = 100

export function activeLoopRange(practice: LoopState): { startMs: number; endMs: number } | null {
  const { loopEnabled, loopStartMs, loopEndMs } = practice
  if (!loopEnabled || loopStartMs === null || loopEndMs === null
    || !Number.isFinite(loopStartMs) || !Number.isFinite(loopEndMs)
    || loopStartMs < 0 || loopEndMs - loopStartMs < MIN_LOOP_DURATION_MS) return null
  return { startMs: loopStartMs, endMs: loopEndMs }
}

export function loopPhase(practice: LoopState): LoopPhase {
  if (activeLoopRange(practice)) return 'active'
  return practice.loopStartMs === null ? 'off' : 'start'
}

export function nextLoopState(practice: LoopState, currentMs: number, durationMs: number): LoopState | null {
  if (loopPhase(practice) === 'active') return { loopStartMs: null, loopEndMs: null, loopEnabled: false }
  if (!Number.isFinite(currentMs) || !Number.isFinite(durationMs) || durationMs <= 0) return null
  const position = Math.max(0, Math.min(currentMs, durationMs))
  if (practice.loopStartMs === null) {
    if (durationMs - position < MIN_LOOP_DURATION_MS) return null
    return { loopStartMs: position, loopEndMs: null, loopEnabled: false }
  }
  if (position - practice.loopStartMs < MIN_LOOP_DURATION_MS) return null
  return { loopStartMs: practice.loopStartMs, loopEndMs: position, loopEnabled: true }
}

export function restartPositionMs(practice: LoopState): number {
  return activeLoopRange(practice)?.startMs ?? 0
}
