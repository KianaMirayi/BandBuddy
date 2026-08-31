import { Film, Maximize, Minimize, Pause, Play, RotateCcw, SkipBack } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { PLAYBACK_RATE_MAX, PLAYBACK_RATE_MIN, type PracticeState } from '@shared/domain.js'
import { activeLoopRange } from '@shared/playback.js'
import { VideoSynchronizer } from '../video-sync.js'
import { formatTime } from '../utils.js'
import { LoopButton } from './LoopButton.js'

export function VideoPlayer({
  src, title, currentMs, durationMs, playing, practice, outputLatencyMs, locked,
  onToggle, onSeek, onRestart, onCycleLoop, onRateChange
}: {
  src: string
  title: string
  currentMs: number
  durationMs: number
  playing: boolean
  practice: PracticeState
  outputLatencyMs: number
  locked: boolean
  onToggle(): void
  onSeek(milliseconds: number): void
  onRestart(): void
  onCycleLoop(): void
  onRateChange(rate: number): void
}): React.JSX.Element {
  const container = useRef<HTMLDivElement>(null)
  const video = useRef<HTMLVideoElement>(null)
  const synchronizer = useRef<VideoSynchronizer | null>(null)
  const [fullscreen, setFullscreen] = useState(false)
  const [videoError, setVideoError] = useState(false)
  const [fullscreenError, setFullscreenError] = useState(false)

  useEffect(() => {
    const element = video.current
    if (!element) return
    setVideoError(false)
    const controller = new VideoSynchronizer(element, () => setVideoError(true))
    element.src = src
    synchronizer.current = controller
    return () => {
      controller.dispose()
      synchronizer.current = null
    }
  }, [src])

  useEffect(() => {
    synchronizer.current?.update({
      currentMs, playing, playbackRate: practice.playbackRate,
      seekPositionMs: practice.positionMs, outputLatencyMs
    })
  }, [src, currentMs, playing, practice.playbackRate, practice.positionMs, outputLatencyMs])

  useEffect(() => {
    const changed = (): void => setFullscreen(document.fullscreenElement === container.current)
    document.addEventListener('fullscreenchange', changed)
    return () => document.removeEventListener('fullscreenchange', changed)
  }, [])

  const toggleFullscreen = async (): Promise<void> => {
    setFullscreenError(false)
    try {
      if (document.fullscreenElement === container.current) await document.exitFullscreen()
      else await container.current?.requestFullscreen()
    } catch {
      setFullscreenError(true)
    }
  }

  return <div ref={container} className={`video-player ${fullscreen ? 'is-fullscreen' : ''}`}>
    <div className="video-heading"><span><Film size={16} />视频练习</span><small>画面随分轨同步 · 原视频静音</small></div>
    <div className="video-stage" onDoubleClick={() => void toggleFullscreen()}>
      <video ref={video} src={src} muted playsInline preload="auto" disablePictureInPicture aria-label={`${title}视频`} />
      {videoError && <div className="video-error" role="alert">
        <b>视频画面暂时无法播放</b><span>分轨音频仍可正常练习，可重试或重新分轨。</span>
        <button onClick={() => { setVideoError(false); synchronizer.current?.retry() }}><RotateCcw size={15} />重新加载视频</button>
      </div>}
    </div>
    <div className="video-caption">
      <b title={title}>{title}</b><span>{practice.playbackRate.toFixed(2)}×</span>
      <button className="video-fullscreen-toggle" aria-label={fullscreen ? '退出视频全屏' : '全屏播放视频'} title="双击画面也可切换全屏" onClick={() => void toggleFullscreen()}>
        {fullscreen ? <Minimize size={18} /> : <Maximize size={18} />}
      </button>
    </div>
    {fullscreenError && <p className="video-fullscreen-error" role="alert">无法进入全屏，请重试。</p>}
    {fullscreen && <div className="video-fullscreen-controls">
      <button disabled={locked} aria-label={activeLoopRange(practice) ? '跳回 A 点并播放' : '跳回开头并播放'} onClick={onRestart}><SkipBack size={20} /></button>
      <button disabled={locked} aria-label={playing ? '暂停视频练习' : '播放视频练习'} onClick={onToggle}>{playing ? <Pause size={22} /> : <Play size={22} />}</button>
      <span className="video-time">{formatTime(currentMs)} / {formatTime(durationMs)}</span>
      <input aria-label="视频播放进度" disabled={locked} type="range" min="0" max={durationMs} step="100" value={Math.max(0, Math.min(durationMs, currentMs))} onChange={(event) => onSeek(Number(event.target.value))} />
      <LoopButton practice={practice} disabled={locked} onClick={onCycleLoop} />
      <label className="video-speed">速度<input aria-label="视频播放速度" disabled={locked} type="range" min={PLAYBACK_RATE_MIN} max={PLAYBACK_RATE_MAX} step="0.01" value={practice.playbackRate} onChange={(event) => onRateChange(Number(event.target.value))} /><span>{practice.playbackRate.toFixed(2)}×</span></label>
    </div>}
  </div>
}
