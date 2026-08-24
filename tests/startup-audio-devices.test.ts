import { describe, expect, it } from 'vitest'
import {
  createDefaultRecordingAudioSettings,
  type AppSettings,
  type RecordingDeviceInfo
} from '../packages/shared/src/domain.js'
import { reconcileAudioDeviceSettings } from '../src/renderer/src/startup-audio-devices.js'

function appSettings(): AppSettings {
  return {
    libraryRoot: 'music',
    runtimeRoot: 'runtime',
    modelRoot: 'models',
    debugMode: false,
    preferredDevice: 'auto',
    audioOutputDeviceId: '',
    latencyMode: 'balanced',
    recordingAudio: createDefaultRecordingAudioSettings(),
    keepSource: true,
    closeToTrayWhileWorking: true,
    network: {
      proxyMode: 'system',
      proxyUrl: '',
      pythonInstallMirror: '',
      pythonIndexUrl: '',
      pytorchIndexUrl: '',
      modelBaseUrl: ''
    }
  }
}

function device(patch: Partial<RecordingDeviceInfo> = {}): RecordingDeviceInfo {
  return {
    id: 'wasapi:realtek',
    backend: 'wasapi-shared',
    name: 'Realtek Audio',
    inputChannels: 2,
    outputChannels: 2,
    duplexChannels: 2,
    sampleRates: [44_100, 48_000],
    preferredSampleRate: 48_000,
    defaultInput: true,
    defaultOutput: true,
    ...patch
  }
}

describe('startup audio device reconciliation', () => {
  it('replaces disconnected playback and recording devices with current system defaults', () => {
    const settings = appSettings()
    settings.audioOutputDeviceId = 'disconnected-web-output'
    settings.recordingAudio = {
      ...settings.recordingAudio,
      backend: 'asio',
      inputDeviceId: 'asio:disconnected-input',
      outputDeviceId: 'asio:disconnected-output',
      inputChannelMode: 'stereo',
      inputChannels: [4, 5],
      sampleRate: 96_000,
      alignmentOffsetMs: 42,
      deviceAlignmentOffsets: { 'wasapi-shared|default|default': 7 }
    }

    const reconciled = reconcileAudioDeviceSettings(settings, {
      playbackOutputDeviceIds: new Set(['default', 'realtek-web-output']),
      recordingDevices: [device()],
      platform: 'Win32'
    })

    expect(reconciled.audioOutputDeviceId).toBe('')
    expect(reconciled.recordingAudio).toMatchObject({
      backend: 'auto',
      inputDeviceId: '',
      outputDeviceId: '',
      inputChannelMode: 'stereo',
      inputChannels: [0, 1],
      sampleRate: 0,
      alignmentOffsetMs: 7
    })
  })

  it('keeps explicit devices that are still available', () => {
    const settings = appSettings()
    settings.audioOutputDeviceId = 'usb-web-output'
    settings.recordingAudio = {
      ...settings.recordingAudio,
      backend: 'asio',
      inputDeviceId: 'asio:usb',
      outputDeviceId: 'asio:usb',
      inputChannelMode: 'stereo',
      inputChannels: [1, 2],
      sampleRate: 48_000
    }
    const usb = device({
      id: 'asio:usb',
      backend: 'asio',
      inputChannels: 4,
      outputChannels: 4
    })

    const reconciled = reconcileAudioDeviceSettings(settings, {
      playbackOutputDeviceIds: new Set(['usb-web-output']),
      recordingDevices: [usb],
      platform: 'Win32'
    })

    expect(reconciled).toBe(settings)
  })

  it('does not erase selections when either device scan itself failed', () => {
    const settings = appSettings()
    settings.audioOutputDeviceId = 'remembered-output'
    settings.recordingAudio = {
      ...settings.recordingAudio,
      backend: 'asio',
      inputDeviceId: 'remembered-input',
      outputDeviceId: 'remembered-recording-output'
    }

    const reconciled = reconcileAudioDeviceSettings(settings, {
      playbackOutputDeviceIds: null,
      recordingDevices: null,
      platform: 'Win32'
    })

    expect(reconciled).toBe(settings)
  })

  it('clears a remembered device that no longer supports its selected direction', () => {
    const settings = appSettings()
    settings.recordingAudio = {
      ...settings.recordingAudio,
      inputDeviceId: 'wasapi:output-only',
      outputDeviceId: 'wasapi:output-only'
    }
    const outputOnly = device({
      id: 'wasapi:output-only',
      inputChannels: 0,
      outputChannels: 2,
      defaultInput: false
    })

    const reconciled = reconcileAudioDeviceSettings(settings, {
      playbackOutputDeviceIds: new Set(),
      recordingDevices: [device(), outputOnly],
      platform: 'Win32'
    })

    expect(reconciled.recordingAudio.inputDeviceId).toBe('')
    expect(reconciled.recordingAudio.outputDeviceId).toBe('wasapi:output-only')
  })
})
