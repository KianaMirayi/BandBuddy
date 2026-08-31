// @vitest-environment jsdom

import { afterEach, describe, expect, it, vi } from 'vitest'
import { MultiTrackAudioEngine } from '../src/renderer/src/audio-engine.js'
import { fixtureDetail, fixtureSongs } from '../src/renderer/src/fixtures.js'

class TransportAudio {
  src = '/track.wav'
  paused = true
  ended = false
  playbackRate = 1
  private position = 0
  get currentTime(): number { return this.position }
  set currentTime(value: number) { this.position = value; this.ended = false }
  play = vi.fn(async () => { this.paused = false })
  pause(): void { this.paused = true }
}

afterEach(() => vi.restoreAllMocks())

async function setup(loopEnabled: boolean) {
  const song = fixtureDetail(fixtureSongs[0]!)
  song.durationMs = 20_000
  song.practice = { ...song.practice, loopEnabled, loopStartMs: 5000, loopEndMs: 20_000 }
  const elements = [new TransportAudio(), new TransportAudio()]
  const engine = new MultiTrackAudioEngine()
  Object.assign(engine, {
    song, practice: song.practice, context: { state: 'running' },
    tracks: new Map(['vocals', 'drums'].map((type, index) => [type, { element: elements[index]! }]))
  })
  let frame: FrameRequestCallback = () => undefined
  vi.spyOn(window, 'requestAnimationFrame').mockImplementation((callback) => { frame = callback; return 1 })
  vi.spyOn(window, 'cancelAnimationFrame').mockImplementation(() => undefined)
  const ended = vi.fn()
  engine.onEnded(ended)
  await engine.play()
  return { engine, elements, ended, tick: () => frame(0) }
}

describe('loop transport at end of media', () => {
  it('resumes every stem from A when B is the final sample', async () => {
    const { engine, elements, ended, tick } = await setup(true)
    for (const element of elements) { element.currentTime = 20; element.ended = true; element.paused = true }
    tick()
    await vi.waitFor(() => expect(elements.every((element) => !element.paused && element.currentTime === 5)).toBe(true))
    expect(ended).not.toHaveBeenCalled()
    expect(elements[0]!.play).toHaveBeenCalledTimes(2)
    engine.pause()
  })

  it('resets all media elements at a natural end so the next play starts at zero', async () => {
    const { engine, elements, ended, tick } = await setup(false)
    for (const element of elements) { element.currentTime = 20; element.ended = true; element.paused = true }
    tick()
    expect(ended).toHaveBeenCalledOnce()
    expect(elements.every((element) => element.currentTime === 0)).toBe(true)
    await engine.play()
    expect(elements.every((element) => !element.paused)).toBe(true)
    engine.pause()
  })

  it('keeps the newest restart playing when an older play request settles late', async () => {
    const { engine, elements } = await setup(false)
    engine.pause()
    const finishOlder: Array<() => void> = []
    for (const element of elements) {
      element.play.mockImplementationOnce(() => {
        element.paused = false
        return new Promise<void>((resolve) => finishOlder.push(resolve))
      })
    }
    const olderPlay = engine.play()
    await vi.waitFor(() => expect(finishOlder).toHaveLength(elements.length))
    engine.pause()
    engine.seek(0)
    await expect(engine.play()).resolves.toBe(true)
    for (const finish of finishOlder) finish()
    await expect(olderPlay).resolves.toBe(false)
    expect(elements.every((element) => !element.paused && element.currentTime === 0)).toBe(true)
    engine.pause()
  })
})
