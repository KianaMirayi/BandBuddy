// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import type { ComponentProps } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { formatGainDb, parseGainDb } from '../src/renderer/src/components/LevelInput.js'
import { isSilenced, silenceToggle } from '../src/renderer/src/utils.js'
import { fixtureDetail, fixtureSongs } from '../src/renderer/src/fixtures.js'
import { PracticeRoom } from '../src/renderer/src/pages/PracticeRoom.js'

vi.mock('../src/renderer/src/components/Waveform.js', () => ({
  Waveform: () => <div className="waveform" />
}))

afterEach(cleanup)

function practiceRoomProps(song = fixtureDetail(fixtureSongs[0]!)): ComponentProps<typeof PracticeRoom> {
  return {
    song,
    practice: song.practice,
    currentMs: 0,
    playing: false,
    selectedStem: 'vocals',
    availableOutputChannelPairs: 6,
    recordingState: {
      target: 'song', phase: 'idle', sessionId: null, songId: null, recordingTrackId: null,
      sourcePositionMs: 0, countInRemaining: 0, sampleRate: 0, bufferFrames: 0,
      latencyMs: 0, xruns: 0, splitDevices: false, message: '', error: null
    },
    recordingMeter: { peak: [0, 0], rms: [0, 0], clipped: false, sourcePositionMs: 0, recording: false },
    locked: false,
    onBack: () => undefined,
    onSeek: () => undefined,
    onTogglePlayback: () => undefined,
    onRestart: () => undefined,
    onCycleLoop: () => undefined,
    onPatch: () => undefined,
    onGuitarSplit: () => undefined,
    onTrack: () => undefined,
    onSelected: () => undefined,
    onExport: () => undefined,
    onAddRecordingTrack: () => undefined,
    onEdit: () => undefined,
    onMore: () => undefined,
    onRecord: () => undefined,
    onStopRecording: () => undefined,
    onCancelRecording: () => undefined,
    onSelectTake: () => undefined,
    onUpdateTake: () => undefined,
    onDeleteTake: () => undefined,
    onRecordingTrack: () => undefined,
    onUseTakePractice: () => undefined
  }
}

describe('parseGainDb', () => {
  it('accepts plain, signed, decimal and suffixed values', () => {
    expect(parseGainDb('3')).toBe(3)
    expect(parseGainDb('+3')).toBe(3)
    expect(parseGainDb('-12.5')).toBe(-12.5)
    expect(parseGainDb('.5')).toBe(0.5)
    expect(parseGainDb('  -6 dB  ')).toBe(-6)
  })

  it('clamps to the range the slider and the IPC contract allow', () => {
    expect(parseGainDb('99')).toBe(6)
    expect(parseGainDb('-99')).toBe(-60)
  })

  it('rejects anything that is not a number', () => {
    for (const text of ['', '   ', 'abc', '-', '1.2.3', '0x10', '1e3', 'dB']) expect(parseGainDb(text)).toBeNull()
  })

  it('formats a stored value back into an editable string', () => {
    expect(formatGainDb(0)).toBe('0')
    expect(formatGainDb(2)).toBe('+2')
    expect(formatGainDb(-3.5)).toBe('-3.5')
  })
})

describe('track level input', () => {
  const readout = () => screen.getByRole('button', { name: '人声电平（双击编辑）' })
  const editor = () => screen.queryByRole('textbox', { name: '人声电平' }) as HTMLInputElement | null

  it('stays a plain readout until it is double-clicked', () => {
    render(<PracticeRoom {...practiceRoomProps()} />)
    expect(readout().textContent).toBe('0 dB (100%)')
    expect(editor()).toBeNull()
  })

  it('opens an editor pre-filled with the stored level, text selected', () => {
    render(<PracticeRoom {...practiceRoomProps()} />)
    fireEvent.doubleClick(readout())
    const field = editor()!
    expect(field.value).toBe('0')
    expect([field.selectionStart, field.selectionEnd]).toEqual([0, 1])
  })

  it('commits a typed level on Enter and closes the editor', () => {
    const onTrack = vi.fn()
    render(<PracticeRoom {...practiceRoomProps()} onTrack={onTrack} />)
    fireEvent.doubleClick(readout())
    fireEvent.change(editor()!, { target: { value: '-6' } })
    fireEvent.keyDown(editor()!, { key: 'Enter' })
    expect(onTrack).toHaveBeenCalledWith('vocals', { gainDb: -6 })
    expect(editor()).toBeNull()
  })

  it('clamps a typed level into the allowed range on blur', () => {
    const onTrack = vi.fn()
    render(<PracticeRoom {...practiceRoomProps()} onTrack={onTrack} />)
    fireEvent.doubleClick(readout())
    fireEvent.change(editor()!, { target: { value: '40' } })
    fireEvent.blur(editor()!)
    expect(onTrack).toHaveBeenCalledWith('vocals', { gainDb: 6 })
  })

  it('reverts unusable text without touching the store', () => {
    const onTrack = vi.fn()
    render(<PracticeRoom {...practiceRoomProps()} onTrack={onTrack} />)
    fireEvent.doubleClick(readout())
    fireEvent.change(editor()!, { target: { value: 'loud' } })
    fireEvent.blur(editor()!)
    expect(onTrack).not.toHaveBeenCalled()
    expect(readout().textContent).toBe('0 dB (100%)')
  })

  it('discards the edit on Escape', () => {
    const onTrack = vi.fn()
    render(<PracticeRoom {...practiceRoomProps()} onTrack={onTrack} />)
    fireEvent.doubleClick(readout())
    fireEvent.change(editor()!, { target: { value: '-30' } })
    fireEvent.keyDown(editor()!, { key: 'Escape' })
    expect(onTrack).not.toHaveBeenCalled()
    expect(editor()).toBeNull()
  })

  it('steps with the arrow keys while editing', () => {
    const onTrack = vi.fn()
    render(<PracticeRoom {...practiceRoomProps()} onTrack={onTrack} />)
    fireEvent.doubleClick(readout())
    fireEvent.keyDown(editor()!, { key: 'ArrowUp' })
    expect(onTrack).toHaveBeenLastCalledWith('vocals', { gainDb: 0.5 })
    fireEvent.keyDown(editor()!, { key: 'ArrowUp' })
    expect(onTrack).toHaveBeenLastCalledWith('vocals', { gainDb: 1 })
    fireEvent.change(editor()!, { target: { value: '-3' } })
    fireEvent.keyDown(editor()!, { key: 'ArrowDown' })
    expect(onTrack).toHaveBeenLastCalledWith('vocals', { gainDb: -3.5 })
  })

  it('also opens from the keyboard, but not from a single mouse click', () => {
    render(<PracticeRoom {...practiceRoomProps()} />)
    fireEvent.click(readout(), { detail: 1 })
    expect(editor()).toBeNull()
    fireEvent.click(readout(), { detail: 0 })
    expect(editor()).not.toBeNull()
  })

  it('locks the level cell together with the slider', () => {
    render(<PracticeRoom {...practiceRoomProps()} locked />)
    const slider = screen.getByRole('slider', { name: '人声电平滑块' }) as HTMLInputElement
    expect((readout() as HTMLButtonElement).disabled).toBe(true)
    expect(slider.disabled).toBe(true)
    expect([slider.min, slider.max, slider.step]).toEqual(['-60', '6', '0.5'])
    fireEvent.doubleClick(readout())
    expect(editor()).toBeNull()
  })
})

describe('silence floor', () => {
  it('counts the floor level as silenced', () => {
    expect(isSilenced({ muted: false, gainDb: 0 })).toBe(false)
    expect(isSilenced({ muted: false, gainDb: -59.5 })).toBe(false)
    expect(isSilenced({ muted: false, gainDb: -60 })).toBe(true)
    expect(isSilenced({ muted: true, gainDb: 0 })).toBe(true)
  })

  it('restores an audible level when un-silencing a floored track', () => {
    expect(silenceToggle({ muted: false, gainDb: 0 })).toEqual({ muted: true })
    expect(silenceToggle({ muted: true, gainDb: -3 })).toEqual({ muted: false })
    expect(silenceToggle({ muted: false, gainDb: -60 })).toEqual({ muted: false, gainDb: 0 })
  })
})

describe('mute button against the silence floor', () => {
  const flooredSong = () => {
    const song = fixtureDetail(fixtureSongs[0]!)
    return {
      ...song,
      practice: {
        ...song.practice,
        tracks: song.practice.tracks.map((track) => (track.stemType === 'vocals' ? { ...track, gainDb: -60 } : track))
      }
    }
  }
  const muteButton = () => [...document.querySelectorAll('.track-row')]
    .find((row) => (row.textContent || '').includes('Vocal'))!
    .querySelector('.ms-buttons button') as HTMLButtonElement

  it('lights up while the level sits on the floor', () => {
    render(<PracticeRoom {...practiceRoomProps(flooredSong())} />)
    expect(muteButton().className).toBe('active')
  })

  it('un-silences a floored track back to 0 dB instead of doing nothing', () => {
    const onTrack = vi.fn()
    render(<PracticeRoom {...practiceRoomProps(flooredSong())} onTrack={onTrack} />)
    fireEvent.click(muteButton())
    expect(onTrack).toHaveBeenCalledWith('vocals', { muted: false, gainDb: 0 })
  })

  it('keeps the plain mute behaviour for audible tracks', () => {
    const onTrack = vi.fn()
    render(<PracticeRoom {...practiceRoomProps()} onTrack={onTrack} />)
    expect(muteButton().className).toBe('')
    fireEvent.click(muteButton())
    expect(onTrack).toHaveBeenCalledWith('vocals', { muted: true })
  })
})
