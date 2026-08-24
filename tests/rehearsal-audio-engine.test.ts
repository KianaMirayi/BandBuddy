import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { fixtureDetail, fixtureRehearsal, fixtureSongs } from '../src/renderer/src/fixtures.js'
import { buildRehearsalTimeline } from '../packages/shared/src/rehearsal.js'

const audioHarness = vi.hoisted(() => ({
  instances: [] as Array<{ play: ReturnType<typeof vi.fn>; load: ReturnType<typeof vi.fn> }>
}))

vi.mock('../src/renderer/src/audio-engine.js', () => {
  class FakeMultiTrackAudioEngine {
    onTime = vi.fn()
    onEnded = vi.fn()
    load = vi.fn().mockResolvedValue(undefined)
    applyPractice = vi.fn()
    seek = vi.fn()
    play = vi.fn().mockResolvedValue(true)
    pause = vi.fn()
    destroy = vi.fn()

    constructor() {
      audioHarness.instances.push(this as unknown as { play: ReturnType<typeof vi.fn>; load: ReturnType<typeof vi.fn> })
    }
  }

  return { MultiTrackAudioEngine: FakeMultiTrackAudioEngine }
})

import { RehearsalAudioEngine } from '../src/renderer/src/rehearsal-audio-engine.js'

class FakeGainNode {
  gain = { value: 1 }
  connect(): this { return this }
  disconnect(): void {}
}

class FakeAudioContext {
  state = 'running'
  currentTime = 0
  destination = {}
  createGain(): FakeGainNode { return new FakeGainNode() }
  close(): Promise<void> { return Promise.resolve() }
  resume(): Promise<void> { return Promise.resolve() }
}

function playbackConfiguration() {
  const song = fixtureDetail(fixtureSongs[0]!)
  song.stems = song.stems.map((stem) => ({ ...stem, mediaUrl: `bandbuddy-media://song/${stem.id}` }))
  return {
    timeline: buildRehearsalTimeline([fixtureRehearsal.items[0]!], [song]),
    songs: [song],
    recordingTracks: [],
    recordingTakes: [],
    outputDeviceId: '',
    latencyMode: 'interactive' as const
  }
}

describe('RehearsalAudioEngine playback startup', () => {
  beforeEach(() => {
    audioHarness.instances.length = 0
    vi.stubGlobal('AudioContext', FakeAudioContext)
    vi.stubGlobal('requestAnimationFrame', vi.fn(() => 1))
    vi.stubGlobal('cancelAnimationFrame', vi.fn())
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('restarts the active song when rehearsal playback resumes', async () => {
    const engine = new RehearsalAudioEngine()
    await engine.configure(playbackConfiguration())

    await engine.play()
    engine.pause()
    await engine.play()

    expect(audioHarness.instances[0]?.play).toHaveBeenCalledTimes(2)
    expect(engine.isPlaying).toBe(true)
    engine.destroy()
  })

  it('waits for an in-flight preparation before reporting playback as started', async () => {
    const engine = new RehearsalAudioEngine()
    await engine.configure(playbackConfiguration())
    const songEngine = audioHarness.instances[0]!
    let resolveFirstLoad: (() => void) | undefined
    songEngine.load.mockImplementationOnce(() => new Promise<void>((resolve) => { resolveFirstLoad = resolve }))

    const firstStart = engine.play()
    const latestStart = engine.play()
    resolveFirstLoad?.()
    await Promise.all([firstStart, latestStart])

    expect(songEngine.play).toHaveBeenCalledTimes(1)
    expect(engine.isPlaying).toBe(true)
    engine.destroy()
  })
})
