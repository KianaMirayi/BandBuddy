declare module 'signalsmith-stretch' {
  export interface SignalsmithStretchSchedule {
    output?: number
    outputTime?: number
    active?: boolean
    semitones?: number
    tonalityHz?: number
    formantSemitones?: number
    formantCompensation?: boolean
    formantBaseHz?: number
  }

  export interface SignalsmithStretchNode extends AudioWorkletNode {
    schedule(change: SignalsmithStretchSchedule): Promise<SignalsmithStretchSchedule>
    start(change?: SignalsmithStretchSchedule): Promise<SignalsmithStretchSchedule>
    stop(when?: number): Promise<SignalsmithStretchSchedule>
    latency(): Promise<number>
  }

  export default function createSignalsmithStretch(
    context: AudioContext,
    options?: AudioWorkletNodeOptions
  ): Promise<SignalsmithStretchNode>
}
