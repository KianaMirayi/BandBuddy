// @vitest-environment jsdom

import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { fixtureDetail, fixtureSongs } from '../src/renderer/src/fixtures.js'
import { PracticeRoom } from '../src/renderer/src/pages/PracticeRoom.js'

vi.mock('../src/renderer/src/components/Waveform.js', () => ({
  Waveform: () => <div className="waveform" />
}))

describe('PracticeRoom output routing', () => {
  it('offers every detected stereo pair and persists a per-track selection', () => {
    const song = fixtureDetail(fixtureSongs[0]!)
    const onTrack = vi.fn()
    render(<PracticeRoom
      song={song}
      practice={song.practice}
      currentMs={0}
      playing={false}
      selectedStem="vocals"
      availableOutputChannelPairs={6}
      recordingState={{
        target: 'song', phase: 'idle', sessionId: null, songId: null, recordingTrackId: null,
        sourcePositionMs: 0, countInRemaining: 0, sampleRate: 0, bufferFrames: 0,
        latencyMs: 0, xruns: 0, splitDevices: false, message: '', error: null
      }}
      recordingMeter={{ peak: [0, 0], rms: [0, 0], clipped: false, sourcePositionMs: 0, recording: false }}
      locked={false}
      onBack={() => undefined}
      onSeek={() => undefined}
      onTogglePlayback={() => undefined}
      onRestart={() => undefined}
      onCycleLoop={() => undefined}
      onPatch={() => undefined}
      onTrack={onTrack}
      onSelected={() => undefined}
      onExport={() => undefined}
      onAddRecordingTrack={() => undefined}
      onEdit={() => undefined}
      onMore={() => undefined}
      onRecord={() => undefined}
      onStopRecording={() => undefined}
      onCancelRecording={() => undefined}
      onSelectTake={() => undefined}
      onUpdateTake={() => undefined}
      onDeleteTake={() => undefined}
      onRecordingTrack={() => undefined}
      onUseTakePractice={() => undefined}
    />)

    const vocalsOutput = screen.getByRole('combobox', { name: '人声输出通道' })
    expect(vocalsOutput.querySelectorAll('option')).toHaveLength(6)
    expect(screen.getByText('多通道输出已启用 · 节拍器与录音预听固定到 1–2')).toBeTruthy()

    fireEvent.change(vocalsOutput, { target: { value: '5' } })
    expect(onTrack).toHaveBeenCalledWith('vocals', { outputChannelPair: 5 })
  })
})
