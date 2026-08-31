import { describe, expect, it } from 'vitest'
import { activeLoopRange, loopPhase, nextLoopState, restartPositionMs, type LoopState } from '@shared/playback.js'

const off: LoopState = { loopStartMs: null, loopEndMs: null, loopEnabled: false }

describe('three-stage A-B transport', () => {
  it('sets A, sets B and enables looping, then clears both markers', () => {
    const a = nextLoopState(off, 12_000, 60_000)!
    expect(a).toEqual({ loopStartMs: 12_000, loopEndMs: null, loopEnabled: false })
    expect(loopPhase(a)).toBe('start')
    const ab = nextLoopState(a, 24_000, 60_000)!
    expect(activeLoopRange(ab)).toEqual({ startMs: 12_000, endMs: 24_000 })
    expect(loopPhase(ab)).toBe('active')
    expect(nextLoopState(ab, 15_000, 60_000)).toEqual(off)
  })

  it('does not create a zero-length, reversed or out-of-bounds loop', () => {
    const a = nextLoopState(off, 5000, 10_000)!
    for (const position of [4000, 5000, 5099, NaN]) expect(nextLoopState(a, position, 10_000)).toBeNull()
    expect(nextLoopState(a, 20_000, 10_000)?.loopEndMs).toBe(10_000)
    expect(nextLoopState(off, 10_000, 10_000)).toBeNull()
    expect(nextLoopState(off, 0, 0)).toBeNull()
  })

  it('only restarts at A for an enabled and valid loop', () => {
    expect(restartPositionMs(off)).toBe(0)
    expect(restartPositionMs({ ...off, loopStartMs: 10_000 })).toBe(0)
    expect(restartPositionMs({ loopStartMs: 10_000, loopEndMs: 30_000, loopEnabled: false })).toBe(0)
    expect(restartPositionMs({ loopStartMs: 10_000, loopEndMs: 30_000, loopEnabled: true })).toBe(10_000)
    expect(restartPositionMs({ loopStartMs: 10_000, loopEndMs: 5000, loopEnabled: true })).toBe(0)
  })
})
