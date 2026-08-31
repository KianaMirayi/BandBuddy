// @vitest-environment jsdom

import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { PlayerBar } from '../src/renderer/src/components/PlayerBar.js'
import { fixtureDetail, fixtureSongs } from '../src/renderer/src/fixtures.js'
import { usePlayerStore } from '../src/renderer/src/player-store.js'

const playerProps = {
  practiceMode: true,
  countInRemaining: 0,
  locked: false,
  onToggle: vi.fn(),
  onSeek: vi.fn(),
  onRestart: vi.fn(),
  onCycleLoop: vi.fn(),
  onPractice: vi.fn()
}

describe('player bar transposition', () => {
  beforeEach(() => {
    const song = fixtureDetail(fixtureSongs[1]!)
    song.practice = {
      ...song.practice,
      playbackRate: 0.8,
      positionMs: 35_000,
      loopStartMs: 30_000,
      loopEndMs: 50_000,
      loopEnabled: true
    }
    usePlayerStore.getState().loadSong(song)
  })

  afterEach(() => {
    cleanup()
    usePlayerStore.getState().unload()
    vi.clearAllMocks()
  })

  it('steps immediately from the footer, including rapid clicks, without changing tempo, position or the track mix', () => {
    usePlayerStore.getState().setSelectedStem('drums')
    usePlayerStore.getState().setPlaying(true)
    const before = usePlayerStore.getState().practice!
    render(<PlayerBar {...playerProps} />)

    expect(screen.queryByRole('dialog', { name: '升降调设置' })).toBeNull()
    const raise = screen.getByRole('button', { name: '升高半音' })
    act(() => {
      fireEvent.click(raise)
      fireEvent.click(raise)
      fireEvent.click(raise)
    })
    expect(usePlayerStore.getState().practice).toEqual({ ...before, pitchSemitones: 3 })
    expect(screen.getByRole('button', { name: '升降调：+3 半音' })).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: '降低半音' }))
    expect(usePlayerStore.getState().practice?.pitchSemitones).toBe(2)
    fireEvent.click(screen.getByRole('button', { name: '恢复原调' }))
    expect(usePlayerStore.getState().practice).toEqual(before)
    expect(usePlayerStore.getState()).toMatchObject({ currentMs: 35_000, playing: true, selectedStem: 'drums' })
    expect(playerProps.onToggle).not.toHaveBeenCalled()
    expect(playerProps.onSeek).not.toHaveBeenCalled()
  })

  it.each([
    ['升高半音', '降低半音', 12],
    ['降低半音', '升高半音', -12]
  ] as const)('stops %s at one octave and still allows stepping back', (direction, reverse, limit) => {
    render(<PlayerBar {...playerProps} />)
    const step = screen.getByRole('button', { name: direction }) as HTMLButtonElement
    act(() => {
      for (let index = 0; index < 25; index += 1) fireEvent.click(step)
    })
    expect(usePlayerStore.getState().practice?.pitchSemitones).toBe(limit)
    expect(step.disabled).toBe(true)

    fireEvent.click(screen.getByRole('button', { name: reverse }))
    expect(usePlayerStore.getState().practice?.pitchSemitones).toBe(limit - Math.sign(limit))
    expect(step.disabled).toBe(false)
    fireEvent.click(screen.getByRole('button', { name: '恢复原调' }))
    expect((screen.getByRole('button', { name: '恢复原调' }) as HTMLButtonElement).disabled).toBe(true)
  })

  it('keeps the slider and direct controls in sync and restores the saved pitch when changing songs', () => {
    const savedSong = fixtureDetail(fixtureSongs[0]!)
    savedSong.practice.pitchSemitones = -5
    render(<PlayerBar {...playerProps} />)
    fireEvent.click(screen.getByRole('button', { name: '升降调：原调' }))
    const slider = screen.getByRole('slider', { name: '升降调半音数' }) as HTMLInputElement
    expect([slider.min, slider.max, slider.step]).toEqual(['-12', '12', '1'])
    fireEvent.change(slider, { target: { value: '7' } })
    expect(screen.getByRole('button', { name: '升降调：+7 半音' })).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: '降低半音' }))
    expect(slider.value).toBe('6')

    act(() => usePlayerStore.getState().loadSong(savedSong))
    expect(screen.queryByRole('dialog', { name: '升降调设置' })).toBeNull()
    expect(screen.getByRole('button', { name: '升降调：−5 半音' })).toBeTruthy()
    expect(usePlayerStore.getState().practice?.pitchSemitones).toBe(-5)
  })

  it('locks every pitch control during recording, including an already open slider', () => {
    usePlayerStore.getState().patchPractice({ pitchSemitones: 4 })
    const { rerender } = render(<PlayerBar {...playerProps} />)
    fireEvent.click(screen.getByRole('button', { name: '升降调：+4 半音' }))
    rerender(<PlayerBar {...playerProps} locked />)

    for (const label of ['降低半音', '升高半音', '升降调：+4 半音', '恢复原调']) {
      const button = screen.getByRole('button', { name: label }) as HTMLButtonElement
      expect(button.disabled).toBe(true)
      fireEvent.click(button)
    }
    const slider = screen.getByRole('slider', { name: '升降调半音数' }) as HTMLInputElement
    expect(slider.disabled).toBe(true)
    fireEvent.change(slider, { target: { value: '-12' } })
    expect(usePlayerStore.getState().practice?.pitchSemitones).toBe(4)
  })

  it('handles pitch keyboard input without triggering playback shortcuts and closes with Escape', () => {
    const globalShortcut = vi.fn()
    render(<div onKeyDown={globalShortcut}><PlayerBar {...playerProps} /></div>)
    const toggle = screen.getByRole('button', { name: '升降调：原调' })
    fireEvent.click(toggle)
    const slider = screen.getByRole('slider', { name: '升降调半音数' })
    fireEvent.keyDown(slider, { key: 'ArrowRight' })
    expect(usePlayerStore.getState().practice?.pitchSemitones).toBe(1)
    fireEvent.keyDown(slider, { key: 'End' })
    expect(usePlayerStore.getState().practice?.pitchSemitones).toBe(12)
    fireEvent.keyDown(slider, { key: 'Home' })
    expect(usePlayerStore.getState().practice?.pitchSemitones).toBe(-12)
    fireEvent.keyDown(slider, { key: 'Escape' })
    expect(globalShortcut).not.toHaveBeenCalled()
    expect(screen.queryByRole('dialog', { name: '升降调设置' })).toBeNull()
    expect(document.activeElement).toBe(toggle)
    expect(usePlayerStore.getState().practice?.loopEnabled).toBe(true)
  })
})
