import { describe, expect, it } from 'vitest'
import { detectMusicalKeyFromSamples } from '../src/main/key-detection.js'

const SAMPLE_RATE = 8192

function synthesizeProgression(chords: number[][], repeats = 4, chordSeconds = 2): Float32Array {
  const chordSamples = Math.round(SAMPLE_RATE * chordSeconds)
  const result = new Float32Array(chordSamples * chords.length * repeats)
  for (let repeat = 0; repeat < repeats; repeat += 1) {
    for (let chordIndex = 0; chordIndex < chords.length; chordIndex += 1) {
      const notes = chords[chordIndex]!
      const offset = (repeat * chords.length + chordIndex) * chordSamples
      for (let sample = 0; sample < chordSamples; sample += 1) {
        const time = sample / SAMPLE_RATE
        const fade = Math.min(1, sample / 160, (chordSamples - sample) / 160)
        let value = 0
        for (const midi of notes) {
          const frequency = 440 * 2 ** ((midi - 69) / 12)
          value += Math.sin(2 * Math.PI * frequency * time)
          value += Math.sin(4 * Math.PI * frequency * time) * 0.18
        }
        result[offset + sample] = value / notes.length * 0.45 * fade
      }
    }
  }
  return result
}

function concatenate(...parts: Float32Array[]): Float32Array {
  const result = new Float32Array(parts.reduce((sum, part) => sum + part.length, 0))
  let offset = 0
  for (const part of parts) { result.set(part, offset); offset += part.length }
  return result
}

describe('musical key detection', () => {
  it('identifies a C major chord progression', () => {
    const samples = synthesizeProgression([[48, 52, 55], [53, 57, 60], [55, 59, 62], [48, 52, 55]])
    const result = detectMusicalKeyFromSamples(samples, SAMPLE_RATE, ['bass', 'guitar'])
    expect(result?.label).toBe('C major')
    expect(result?.candidates).toHaveLength(3)
    expect(result?.confidence).toBeGreaterThan(0.25)
  })

  it('distinguishes A minor from its relative major', () => {
    const samples = synthesizeProgression([[45, 48, 52], [50, 53, 57], [52, 56, 59], [45, 48, 52]])
    const result = detectMusicalKeyFromSamples(samples, SAMPLE_RATE, ['piano'])
    expect(result?.label).toBe('A minor')
  })

  it('marks a sustained local-key change as a possible modulation', () => {
    const cMajor = synthesizeProgression([[48, 52, 55], [53, 57, 60], [55, 59, 62], [48, 52, 55]], 3)
    const gMajor = synthesizeProgression([[55, 59, 62], [60, 64, 67], [62, 66, 69], [55, 59, 62]], 3)
    const result = detectMusicalKeyFromSamples(concatenate(cMajor, gMajor), SAMPLE_RATE, ['other'])
    expect(result?.segments.some((segment) => segment.label === 'C major')).toBe(true)
    expect(result?.segments.some((segment) => segment.label === 'G major')).toBe(true)
    expect(result?.segments.some((segment) => segment.possibleModulation)).toBe(true)
  })

  it('rejects silence', () => {
    expect(detectMusicalKeyFromSamples(new Float32Array(SAMPLE_RATE * 10), SAMPLE_RATE)).toBeNull()
  })
})
