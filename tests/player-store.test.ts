import { describe, expect, it } from 'vitest'
import { createDefaultPracticeState } from '../packages/shared/src/domain.js'
import { fixtureDetail, fixtureSongs } from '../src/renderer/src/fixtures.js'
import { patchTrackStates, usePlayerStore } from '../src/renderer/src/player-store.js'

describe('practice track button rules', () => {
  it('keeps Solo exclusive and prevents a track from being both muted and soloed', () => {
    let tracks = createDefaultPracticeState('song').tracks
    tracks = patchTrackStates(tracks, 'drums', { solo: true })
    tracks = patchTrackStates(tracks, 'vocals', { solo: true })

    expect(tracks.filter((track) => track.solo).map((track) => track.stemType)).toEqual(['vocals'])
    tracks = patchTrackStates(tracks, 'vocals', { muted: true })
    expect(tracks.find((track) => track.stemType === 'vocals')).toMatchObject({ muted: true, solo: false })

    tracks = patchTrackStates(tracks, 'vocals', { solo: true })
    expect(tracks.find((track) => track.stemType === 'vocals')).toMatchObject({ muted: false, solo: true })
  })

  it('uses the editable song BPM as the metronome value when loading a song', () => {
    const song = fixtureDetail(fixtureSongs[1]!)
    song.practice.metronomeBpm = 120
    song.practice.metronomeOffsetMs = 0
    song.beatOffsetMs = -86
    expect(song.bpm).toBe(74)

    usePlayerStore.getState().loadSong(song)
    expect(usePlayerStore.getState().practice?.metronomeBpm).toBe(74)
    expect(usePlayerStore.getState().practice?.metronomeOffsetMs).toBe(-86)
    usePlayerStore.getState().unload()
  })

  it.each([
    [-24, -12], [24, 12], [-3.8, -4], [2.2, 2], [NaN, 0], [Infinity, 0]
  ])('normalizes a saved or edited pitch of %s before playback and autosave', (input, expected) => {
    const song = fixtureDetail(fixtureSongs[0]!)
    song.practice.pitchSemitones = input
    usePlayerStore.getState().loadSong(song)
    expect(usePlayerStore.getState().practice?.pitchSemitones).toBe(expected)
    usePlayerStore.getState().patchPractice({ pitchSemitones: 0 })
    usePlayerStore.getState().patchPractice({ pitchSemitones: input })
    expect(usePlayerStore.getState().practice?.pitchSemitones).toBe(expected)
    usePlayerStore.getState().unload()
  })
})
