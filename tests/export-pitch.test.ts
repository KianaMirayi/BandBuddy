import { describe, expect, it } from 'vitest'
import { exportedStemPitchSemitones } from '../src/main/exporter.js'

describe('export pitch policy', () => {
  it('applies the current pitch to every non-drum stem', () => {
    const request = { applyPitchShift: true, pitchSemitones: -5 }
    expect(exportedStemPitchSemitones(request, 'vocals')).toBe(-5)
    expect(exportedStemPitchSemitones(request, 'bass')).toBe(-5)
    expect(exportedStemPitchSemitones(request, 'drums')).toBe(0)
  })

  it('keeps all stems at the original key when no shift is active', () => {
    expect(exportedStemPitchSemitones({ applyPitchShift: false, pitchSemitones: 4 }, 'piano')).toBe(0)
  })
})
