// @vitest-environment jsdom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { fixtureDetail, fixtureSongs } from '../src/renderer/src/fixtures.js'
import { MultiTrackAudioEngine } from '../src/renderer/src/audio-engine.js'

const stretchMock = vi.hoisted(() => {
  const node = {
    connect: vi.fn(),
    disconnect: vi.fn(),
    start: vi.fn(async (change: unknown) => change),
    schedule: vi.fn(async (change: unknown) => change),
    stop: vi.fn(async () => undefined),
    latency: vi.fn(async () => 0.125)
  }
  node.connect.mockReturnValue(node)
  return { node, create: vi.fn(async () => node) }
})

vi.mock('signalsmith-stretch', () => ({ default: stretchMock.create }))

class FakeAudioParam {
  value = 0
  cancelScheduledValues(): void {}
  setValueAtTime(value: number): void { this.value = value }
  linearRampToValueAtTime(value: number): void { this.value = value }
  exponentialRampToValueAtTime(value: number): void { this.value = value }
}

class FakeAudioNode {
  readonly connections: unknown[] = []
  connect(destination: unknown): unknown { this.connections.push(destination); return destination }
  disconnect(): void {}
}

class FakeGainNode extends FakeAudioNode { readonly gain = new FakeAudioParam() }
class FakeDelayNode extends FakeAudioNode { readonly delayTime = new FakeAudioParam() }
class FakeCompressorNode extends FakeAudioNode {
  readonly threshold = new FakeAudioParam()
  readonly knee = new FakeAudioParam()
  readonly ratio = new FakeAudioParam()
  readonly attack = new FakeAudioParam()
  readonly release = new FakeAudioParam()
}

class FakeAudioContext {
  static latest: FakeAudioContext | null = null
  readonly audioWorklet = {}
  readonly destination = new FakeAudioNode()
  readonly gains: FakeGainNode[] = []
  readonly delays: FakeDelayNode[] = []
  readonly sources: FakeAudioNode[] = []
  readonly currentTime = 1
  readonly state = 'running'

  constructor() { FakeAudioContext.latest = this }
  createGain(): FakeGainNode { const node = new FakeGainNode(); this.gains.push(node); return node }
  createDelay(): FakeDelayNode { const node = new FakeDelayNode(); this.delays.push(node); return node }
  createDynamicsCompressor(): FakeCompressorNode { return new FakeCompressorNode() }
  createMediaElementSource(): FakeAudioNode { const node = new FakeAudioNode(); this.sources.push(node); return node }
  close(): Promise<void> { return Promise.resolve() }
  resume(): Promise<void> { return Promise.resolve() }
}

class FakeAudioElement {
  preload = ''
  crossOrigin: string | null = null
  preservesPitch = true
  src = ''
  playbackRate = 1
  currentTime = 0
  paused = true
  ended = false
  play(): Promise<void> { this.paused = false; return Promise.resolve() }
  pause(): void { this.paused = true }
  removeAttribute(name: string): void { if (name === 'src') this.src = '' }
  load(): void {}
}

describe('Signalsmith realtime pitch graph', () => {
  beforeEach(() => {
    stretchMock.create.mockClear()
    stretchMock.node.connect.mockClear()
    stretchMock.node.disconnect.mockClear()
    stretchMock.node.start.mockClear()
    stretchMock.node.schedule.mockClear()
    stretchMock.node.latency.mockClear()
    vi.stubGlobal('AudioContext', FakeAudioContext)
    vi.stubGlobal('AudioWorkletNode', class {})
    vi.stubGlobal('Audio', FakeAudioElement)
  })

  afterEach(() => vi.unstubAllGlobals())

  it('routes harmonic stems through Signalsmith while drums use the latency-aligned bypass', async () => {
    const song = fixtureDetail(fixtureSongs[0]!)
    song.practice = { ...song.practice, pitchSemitones: 5 }
    const engine = new MultiTrackAudioEngine()

    await engine.load(song)

    const context = FakeAudioContext.latest!
    const harmonicBus = context.gains[1]!
    const vocalsGain = context.gains[4]!
    const drumsGain = context.gains[5]!
    const bypassDelay = context.delays[1]!
    expect(stretchMock.create).toHaveBeenCalledTimes(1)
    expect((stretchMock.create as typeof stretchMock.create & { moduleUrl?: string }).moduleUrl)
      .toMatch(/SignalsmithStretch\.mjs$/)
    expect((stretchMock.create as typeof stretchMock.create & { moduleUrl?: string }).moduleUrl)
      .not.toMatch(/^blob:/)
    expect(harmonicBus.connections).toContain(stretchMock.node)
    expect(vocalsGain.connections).toContain(harmonicBus)
    expect(drumsGain.connections).toContain(bypassDelay)
    expect(stretchMock.node.schedule).toHaveBeenLastCalledWith(expect.objectContaining({ semitones: 5 }))
    expect(bypassDelay.delayTime.value).toBe(0.125)
    expect(engine.outputLatencySeconds).toBe(0.125)

    engine.applyPractice({ ...song.practice, masterGainDb: -3 })
    expect(stretchMock.node.schedule).toHaveBeenCalledTimes(1)

    engine.applyPractice({ ...song.practice, pitchSemitones: 0 })
    await (engine as unknown as { pitchTransition: Promise<void> }).pitchTransition
    expect(stretchMock.node.schedule).toHaveBeenLastCalledWith(expect.objectContaining({ active: false }))
    expect(bypassDelay.delayTime.value).toBe(0)
    expect(engine.outputLatencySeconds).toBe(0)
    engine.destroy()
  })
})
