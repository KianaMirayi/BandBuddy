import { Repeat2 } from 'lucide-react'
import { loopPhase, type LoopState } from '@shared/playback.js'
import { formatTime } from '../utils.js'

export function LoopButton({ practice, disabled, onClick }: {
  practice: LoopState
  disabled?: boolean
  onClick(): void
}): React.JSX.Element {
  const phase = loopPhase(practice)
  const label = phase === 'active' ? '取消 A-B 循环' : phase === 'start' ? '设置 B 点并开始循环' : '设置 A 点'
  const hint = phase === 'active'
    ? `${formatTime(practice.loopStartMs!)} – ${formatTime(practice.loopEndMs!)} · 点击取消`
    : phase === 'start' ? `A ${formatTime(practice.loopStartMs!)} · 再次点击设置 B 点` : '点击设置 A 点，再次点击设置 B 点'
  return <button
    type="button"
    className={`loop-button loop-${phase}`}
    aria-label={label}
    aria-pressed={phase === 'active'}
    data-loop-state={phase}
    title={hint}
    disabled={disabled}
    onClick={onClick}
  >
    <span className="loop-symbol" key={phase} aria-hidden="true"><Repeat2 size={18} /></span>
    <span aria-hidden="true">{phase === 'start' ? 'A' : 'A–B'}</span>
    <span className="loop-state-dot" aria-hidden="true" />
    <span className="sr-only" role="status">{phase === 'active' ? 'A-B 循环中' : phase === 'start' ? 'A 点已设置，等待 B 点' : 'A-B 循环关闭'}</span>
  </button>
}
