import {
  MUSICAL_KEY_TONICS,
  formatMusicalKey,
  type MusicalKeyAnalysis,
  type MusicalKeyCandidate,
  type MusicalKeyMode,
  type MusicalKeySegment,
  type StemType
} from '@shared/domain.js'

// Krumhansl–Kessler pitch-class profiles, ordered C through B.
const MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
const MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
const MODES: MusicalKeyMode[] = ['major', 'minor']
const MIN_ANALYSIS_SECONDS = 6
const SEGMENT_SECONDS = 20
const CHANGE_PENALTY = 0.075
const LOW_CONFIDENCE_THRESHOLD = 0.7

interface ChromaFrame {
  startMs: number
  endMs: number
  weight: number
  chroma: Float64Array
}

interface ScoredKey extends MusicalKeyCandidate {
  index: number
  score: number
}

interface KeyWindow {
  startMs: number
  endMs: number
  ranked: ScoredKey[]
  scores: number[]
}

function nextPowerOfTwo(value: number): number {
  let result = 1
  while (result < value) result *= 2
  return result
}

function fft(real: Float64Array, imaginary: Float64Array): void {
  const size = real.length
  for (let index = 1, reversed = 0; index < size; index += 1) {
    let bit = size >> 1
    for (; reversed & bit; bit >>= 1) reversed ^= bit
    reversed ^= bit
    if (index >= reversed) continue
    const realValue = real[index]!
    const imaginaryValue = imaginary[index]!
    real[index] = real[reversed]!
    imaginary[index] = imaginary[reversed]!
    real[reversed] = realValue
    imaginary[reversed] = imaginaryValue
  }
  for (let length = 2; length <= size; length *= 2) {
    const angle = -2 * Math.PI / length
    const stepReal = Math.cos(angle)
    const stepImaginary = Math.sin(angle)
    for (let offset = 0; offset < size; offset += length) {
      let twiddleReal = 1
      let twiddleImaginary = 0
      const half = length / 2
      for (let item = 0; item < half; item += 1) {
        const even = offset + item
        const odd = even + half
        const oddReal = real[odd]! * twiddleReal - imaginary[odd]! * twiddleImaginary
        const oddImaginary = real[odd]! * twiddleImaginary + imaginary[odd]! * twiddleReal
        real[odd] = real[even]! - oddReal
        imaginary[odd] = imaginary[even]! - oddImaginary
        real[even] = real[even]! + oddReal
        imaginary[even] = imaginary[even]! + oddImaginary
        const nextReal = twiddleReal * stepReal - twiddleImaginary * stepImaginary
        twiddleImaginary = twiddleReal * stepImaginary + twiddleImaginary * stepReal
        twiddleReal = nextReal
      }
    }
  }
}

function buildChromaFrames(samples: Float32Array, sampleRate: number): ChromaFrame[] {
  const frameSize = Math.min(8192, Math.max(4096, nextPowerOfTwo(sampleRate * 0.45)))
  const hopSize = frameSize / 2
  const window = Float64Array.from({ length: frameSize }, (_, index) => 0.5 - 0.5 * Math.cos(2 * Math.PI * index / (frameSize - 1)))
  const real = new Float64Array(frameSize)
  const imaginary = new Float64Array(frameSize)
  const magnitudes = new Float64Array(frameSize / 2)
  const minimumBin = Math.max(2, Math.ceil(55 * frameSize / sampleRate))
  const maximumBin = Math.min(frameSize / 2 - 2, Math.floor(2800 * frameSize / sampleRate))
  const frames: ChromaFrame[] = []

  for (let start = 0; start + frameSize <= samples.length; start += hopSize) {
    let mean = 0
    let squareSum = 0
    for (let index = 0; index < frameSize; index += 1) mean += samples[start + index]!
    mean /= frameSize
    for (let index = 0; index < frameSize; index += 1) {
      const sample = samples[start + index]! - mean
      squareSum += sample * sample
      real[index] = sample * window[index]!
      imaginary[index] = 0
    }
    const rms = Math.sqrt(squareSum / frameSize)
    if (rms < 0.00008) continue
    fft(real, imaginary)
    for (let bin = minimumBin - 1; bin <= maximumBin + 1; bin += 1) {
      magnitudes[bin] = Math.hypot(real[bin]!, imaginary[bin]!)
    }

    const chroma = new Float64Array(12)
    let spectralEnergy = 0
    for (let bin = minimumBin; bin <= maximumBin; bin += 1) {
      const magnitude = magnitudes[bin]!
      if (magnitude <= magnitudes[bin - 1]! || magnitude < magnitudes[bin + 1]!) continue
      const localFloor = (magnitudes[bin - 2]! + magnitudes[bin + 2]!) * 0.25
      const prominence = Math.max(0, magnitude - localFloor)
      if (prominence <= 0) continue
      const frequency = bin * sampleRate / frameSize
      const midi = 69 + 12 * Math.log2(frequency / 440)
      const nearest = Math.round(midi)
      const detune = midi - nearest
      const tuningWeight = Math.exp(-(detune * detune) / (2 * 0.32 * 0.32))
      const pitchClass = ((nearest % 12) + 12) % 12
      const contribution = Math.log1p(prominence) * tuningWeight / Math.sqrt(frequency / 55)
      chroma[pitchClass] = chroma[pitchClass]! + contribution
      spectralEnergy += contribution
    }
    if (spectralEnergy < 0.02) continue
    for (let pitchClass = 0; pitchClass < 12; pitchClass += 1) chroma[pitchClass] = chroma[pitchClass]! / spectralEnergy
    frames.push({
      startMs: start / sampleRate * 1000,
      endMs: (start + frameSize) / sampleRate * 1000,
      weight: Math.min(1, Math.sqrt(rms * 8)),
      chroma
    })
  }
  return frames
}

function aggregateChroma(frames: ChromaFrame[]): Float64Array {
  const result = new Float64Array(12)
  let totalWeight = 0
  for (const frame of frames) {
    totalWeight += frame.weight
    for (let pitchClass = 0; pitchClass < 12; pitchClass += 1) {
      result[pitchClass] = result[pitchClass]! + frame.chroma[pitchClass]! * frame.weight
    }
  }
  if (totalWeight > 0) {
    for (let pitchClass = 0; pitchClass < 12; pitchClass += 1) result[pitchClass] = result[pitchClass]! / totalWeight
  }
  return result
}

function correlation(chroma: Float64Array, tonic: number, profile: number[]): number {
  let chromaMean = 0
  let profileMean = 0
  for (let index = 0; index < 12; index += 1) {
    chromaMean += chroma[index]!
    profileMean += profile[index]!
  }
  chromaMean /= 12
  profileMean /= 12
  let numerator = 0
  let chromaVariance = 0
  let profileVariance = 0
  for (let pitchClass = 0; pitchClass < 12; pitchClass += 1) {
    const chromaValue = chroma[pitchClass]! - chromaMean
    const profileValue = profile[(pitchClass - tonic + 12) % 12]! - profileMean
    numerator += chromaValue * profileValue
    chromaVariance += chromaValue * chromaValue
    profileVariance += profileValue * profileValue
  }
  const denominator = Math.sqrt(chromaVariance * profileVariance)
  return denominator > 1e-12 ? numerator / denominator : 0
}

function scoreKeys(chroma: Float64Array): ScoredKey[] {
  const scored: ScoredKey[] = []
  for (let modeIndex = 0; modeIndex < MODES.length; modeIndex += 1) {
    const mode = MODES[modeIndex]!
    const profile = mode === 'major' ? MAJOR_PROFILE : MINOR_PROFILE
    for (let tonic = 0; tonic < 12; tonic += 1) {
      scored.push({
        index: modeIndex * 12 + tonic,
        tonic: MUSICAL_KEY_TONICS[tonic]!,
        mode,
        label: formatMusicalKey(MUSICAL_KEY_TONICS[tonic]!, mode),
        confidence: 0,
        score: correlation(chroma, tonic, profile)
      })
    }
  }
  const bestScore = Math.max(...scored.map((candidate) => candidate.score))
  const temperature = 0.055
  const weights = scored.map((candidate) => Math.exp((candidate.score - bestScore) / temperature))
  const weightSum = weights.reduce((sum, weight) => sum + weight, 0)
  for (let index = 0; index < scored.length; index += 1) {
    scored[index]!.confidence = Math.round(weights[index]! / weightSum * 10_000) / 10_000
  }
  return scored.sort((left, right) => right.score - left.score)
}

function buildWindows(frames: ChromaFrame[], durationMs: number): KeyWindow[] {
  const windows: KeyWindow[] = []
  const windowMs = SEGMENT_SECONDS * 1000
  for (let startMs = 0; startMs < durationMs; startMs += windowMs) {
    const endMs = Math.min(durationMs, startMs + windowMs)
    const localFrames = frames.filter((frame) => frame.startMs < endMs && frame.endMs > startMs)
    if (localFrames.length < 4) continue
    const ranked = scoreKeys(aggregateChroma(localFrames))
    const scores = Array.from({ length: 24 }, () => -1)
    for (const candidate of ranked) scores[candidate.index] = candidate.score
    windows.push({ startMs, endMs, ranked, scores })
  }
  return windows
}

function smoothWindowKeys(windows: KeyWindow[]): number[] {
  if (!windows.length) return []
  const states = 24
  const history: number[][] = []
  let previous = [...windows[0]!.scores]
  history.push(Array.from({ length: states }, (_, state) => state))
  for (let windowIndex = 1; windowIndex < windows.length; windowIndex += 1) {
    const next = Array.from({ length: states }, () => Number.NEGATIVE_INFINITY)
    const back = Array.from({ length: states }, () => 0)
    for (let state = 0; state < states; state += 1) {
      let bestPrevious = 0
      let bestValue = Number.NEGATIVE_INFINITY
      for (let prior = 0; prior < states; prior += 1) {
        const transition = prior === state ? 0 : CHANGE_PENALTY
        const value = previous[prior]! - transition
        if (value > bestValue) { bestValue = value; bestPrevious = prior }
      }
      next[state] = bestValue + windows[windowIndex]!.scores[state]!
      back[state] = bestPrevious
    }
    previous = next
    history.push(back)
  }
  let state = previous.indexOf(Math.max(...previous))
  const path = Array.from({ length: windows.length }, () => 0)
  for (let windowIndex = windows.length - 1; windowIndex >= 0; windowIndex -= 1) {
    path[windowIndex] = state
    state = history[windowIndex]![state]!
  }
  return path
}

function buildSegments(windows: KeyWindow[], path: number[], overallLabel: string): MusicalKeySegment[] {
  const merged: MusicalKeySegment[] = []
  for (let index = 0; index < windows.length; index += 1) {
    const window = windows[index]!
    const selected = window.ranked.find((candidate) => candidate.index === path[index]) ?? window.ranked[0]!
    const previous = merged.at(-1)
    if (previous?.label === selected.label) {
      const previousDuration = previous.endMs - previous.startMs
      const windowDuration = window.endMs - window.startMs
      previous.confidence = Math.round((previous.confidence * previousDuration + selected.confidence * windowDuration) / (previousDuration + windowDuration) * 10_000) / 10_000
      previous.endMs = window.endMs
      continue
    }
    merged.push({
      tonic: selected.tonic,
      mode: selected.mode,
      label: selected.label,
      confidence: selected.confidence,
      startMs: window.startMs,
      endMs: window.endMs,
      possibleModulation: false
    })
  }
  for (const segment of merged) {
    segment.possibleModulation = segment.label !== overallLabel
      && segment.endMs - segment.startMs >= 15_000
      && segment.confidence >= 0.18
  }
  return merged
}

export function detectMusicalKeyFromSamples(
  samples: Float32Array,
  sampleRate: number,
  analyzedStems: StemType[] = []
): MusicalKeyAnalysis | null {
  if (!Number.isFinite(sampleRate) || sampleRate <= 0 || samples.length < sampleRate * MIN_ANALYSIS_SECONDS) return null
  const frames = buildChromaFrames(samples, sampleRate)
  if (frames.length < 8) return null
  const ranked = scoreKeys(aggregateChroma(frames))
  const primary = ranked[0]!
  if (primary.score < 0.08) return null
  const durationMs = Math.round(samples.length / sampleRate * 1000)
  const windows = buildWindows(frames, durationMs)
  const segments = buildSegments(windows, smoothWindowKeys(windows), primary.label)
  return {
    tonic: primary.tonic,
    mode: primary.mode,
    label: primary.label,
    confidence: primary.confidence,
    lowConfidence: primary.confidence < LOW_CONFIDENCE_THRESHOLD,
    candidates: ranked.slice(0, 3).map(({ tonic, mode, label, confidence }) => ({ tonic, mode, label, confidence })),
    segments,
    analyzedStems,
    analyzedDurationMs: durationMs,
    analyzedAt: new Date().toISOString()
  }
}
