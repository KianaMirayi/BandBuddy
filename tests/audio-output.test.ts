import { describe, expect, it, vi } from 'vitest'
import {
  resolveOutputChannelPair,
  routableOutputChannelCount,
  setAudioContextOutputDevice,
  setAudioContextOutputDeviceOrDefault
} from '../src/renderer/src/audio-engine.js'

describe('Web Audio output routing', () => {
  it('uses even stereo-pair capacity up to the Web Audio merger limit', () => {
    expect(routableOutputChannelCount({ maxChannelCount: 12, channelCount: 2 })).toBe(12)
    expect(routableOutputChannelCount({ maxChannelCount: 7 })).toBe(6)
    expect(routableOutputChannelCount({ maxChannelCount: 64 })).toBe(32)
    expect(routableOutputChannelCount({})).toBe(2)
  })

  it('keeps valid stereo pairs and falls back unavailable routes to 1–2', () => {
    expect(resolveOutputChannelPair(5, 12)).toBe(5)
    expect(resolveOutputChannelPair(11, 12)).toBe(11)
    expect(resolveOutputChannelPair(12, 12)).toBe(1)
    expect(resolveOutputChannelPair(5, 4)).toBe(1)
  })

  it('sets the output on the shared AudioContext rather than an individual media element', async () => {
    const setSinkId = vi.fn().mockResolvedValue(undefined)

    await setAudioContextOutputDevice({ setSinkId }, 'usb-interface-output')

    expect(setSinkId).toHaveBeenCalledOnce()
    expect(setSinkId).toHaveBeenCalledWith('usb-interface-output')
  })

  it('uses the empty sink ID to return the entire mix to the system default output', async () => {
    const setSinkId = vi.fn().mockResolvedValue(undefined)

    await setAudioContextOutputDevice({ setSinkId }, '')

    expect(setSinkId).toHaveBeenCalledWith('')
  })

  it('does not silently fall back to the default output when explicit selection is unsupported', async () => {
    await expect(setAudioContextOutputDevice({}, 'usb-interface-output'))
      .rejects.toThrow('AUDIO_OUTPUT_DEVICE_SELECTION_UNSUPPORTED')
  })

  it('falls back to the system default while loading when a remembered output disappeared', async () => {
    const setSinkId = vi.fn()
      .mockRejectedValueOnce(new DOMException('Device not found', 'NotFoundError'))
      .mockResolvedValueOnce(undefined)

    await expect(setAudioContextOutputDeviceOrDefault({ setSinkId }, 'disconnected-output'))
      .resolves.toBe('')
    expect(setSinkId.mock.calls).toEqual([['disconnected-output'], ['']])
  })

  it('still reports an error when even the system default output cannot be selected', async () => {
    const setSinkId = vi.fn().mockRejectedValue(new DOMException('No output', 'NotFoundError'))

    await expect(setAudioContextOutputDeviceOrDefault({ setSinkId }, 'disconnected-output'))
      .rejects.toThrow('No output')
  })
})
