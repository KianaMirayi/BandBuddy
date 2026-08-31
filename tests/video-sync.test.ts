// @vitest-environment jsdom

import { beforeEach, describe, expect, it, vi } from 'vitest'
import { VideoSynchronizer, type VideoPlaybackState } from '../src/renderer/src/video-sync.js'

class FakeVideo extends EventTarget {
  muted = false
  readyState = 1
  duration = 60
  currentTime = 0
  playbackRate = 1
  paused = true
  play = vi.fn(async () => { this.paused = false })
  pause = vi.fn(() => { this.paused = true })
  load = vi.fn()
  removeAttribute = vi.fn()
}

const state: VideoPlaybackState = { currentMs: 12_000, playing: false, playbackRate: 1, seekPositionMs: 12_000, outputLatencyMs: 0 }
let video: FakeVideo
let controller: VideoSynchronizer
let onError = vi.fn<() => void>()

beforeEach(() => {
  video = new FakeVideo()
  onError = vi.fn<() => void>()
  controller = new VideoSynchronizer(video as unknown as HTMLVideoElement, onError)
})

describe('video follows the stem transport', () => {
  it('follows start, pause, speed and seek without enabling original audio', async () => {
    controller.update(state)
    expect(video.currentTime).toBe(12)
    expect(video.play).not.toHaveBeenCalled()
    controller.update({ ...state, playing: true, playbackRate: 0.5 })
    await Promise.resolve()
    expect(video.play).toHaveBeenCalledOnce()
    expect(video.muted).toBe(true)
    expect(video.playbackRate).toBe(0.5)
    controller.update({ ...state, playing: true, currentMs: 32_000, seekPositionMs: 32_000, playbackRate: 4 })
    expect(video.currentTime).toBe(32)
    expect(video.playbackRate).toBe(4)
    controller.update({ ...state, currentMs: 32_500 })
    expect(video.paused).toBe(true)
    expect(video.currentTime).toBe(32.5)
  })

  it('corrects loop jumps immediately but lets frames advance naturally within tolerance', () => {
    controller.update({ ...state, playing: true })
    video.currentTime = 12.015
    controller.update({ ...state, playing: true, currentMs: 12_020 })
    expect(video.currentTime).toBe(12.015)
    controller.update({ ...state, playing: true, currentMs: 5000 })
    expect(video.currentTime).toBe(5)
    video.currentTime = 5.3
    controller.update({ ...state, playing: true, currentMs: 5050 })
    expect(video.currentTime).toBe(5.05)
  })

  it('uses the newest transport state when metadata arrives late', () => {
    video.readyState = 0
    controller.update({ ...state, playing: true })
    controller.update({ ...state, currentMs: 24_000, playbackRate: 0.8 })
    video.readyState = 1
    video.dispatchEvent(new Event('loadedmetadata'))
    expect(video.currentTime).toBe(24)
    expect(video.playbackRate).toBe(0.8)
    expect(video.play).not.toHaveBeenCalled()
  })

  it('accounts for the audible pitch-processing latency in source time', () => {
    controller.update({ ...state, playing: true, playbackRate: 0.5, outputLatencyMs: 120 })
    expect(video.currentTime).toBe(11.94)
    controller.update({ ...state, playing: false, playbackRate: 0.5, outputLatencyMs: 120 })
    expect(video.currentTime).toBe(12)
  })

  it('does not restart an ended video until the audio clock seeks back', async () => {
    video.duration = 10
    controller.update({ ...state, playing: true })
    expect(video.currentTime).toBe(10)
    expect(video.play).not.toHaveBeenCalled()
    controller.update({ ...state, currentMs: 5000, playing: true })
    await Promise.resolve()
    expect(video.play).toHaveBeenCalledOnce()
  })

  it('ignores an obsolete play rejection after pause or disposal', async () => {
    let reject!: (error: Error) => void
    video.play.mockImplementationOnce(() => new Promise<void>((_resolve, rejectPlay) => { reject = rejectPlay }))
    controller.update({ ...state, playing: true })
    controller.update(state)
    controller.dispose()
    reject(new Error('interrupted'))
    await Promise.resolve()
    expect(onError).not.toHaveBeenCalled()
    expect(video.removeAttribute).toHaveBeenCalledWith('src')
    video.dispatchEvent(new Event('canplay'))
    expect(video.play).toHaveBeenCalledOnce()
  })

  it('reports video failure once and can retry while leaving transport alone', async () => {
    video.play.mockRejectedValueOnce(new Error('decode failure'))
    controller.update({ ...state, playing: true })
    await Promise.resolve()
    expect(onError).toHaveBeenCalledOnce()
    controller.update({ ...state, playing: true, currentMs: 13000 })
    expect(video.play).toHaveBeenCalledOnce()
    controller.retry()
    video.dispatchEvent(new Event('loadedmetadata'))
    expect(video.play).toHaveBeenCalledTimes(2)
  })
})
