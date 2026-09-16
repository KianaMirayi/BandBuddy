import { readFileSync, writeFileSync } from 'node:fs'
import type { BrowserWindow } from 'electron'

export interface WindowSize {
  width: number
  height: number
  maximized: boolean
}

export interface WindowSizeLimits {
  width: number
  height: number
}

const SAVE_DELAY_MS = 400

/** Coerces a stored value into a usable size, or null when it cannot be trusted. */
export function normalizeWindowSize(raw: unknown, limits: WindowSizeLimits): WindowSize | null {
  if (!raw || typeof raw !== 'object') return null
  const candidate = raw as Partial<Record<keyof WindowSize, unknown>>
  const width = Math.round(Number(candidate.width))
  const height = Math.round(Number(candidate.height))
  if (!Number.isFinite(width) || !Number.isFinite(height)) return null
  return {
    width: Math.max(limits.width, width),
    height: Math.max(limits.height, height),
    maximized: candidate.maximized === true
  }
}

/** Remembers the window size and maximized state in a small file next to the other local state. */
export class WindowState {
  private timer: NodeJS.Timeout | null = null
  private pending: WindowSize | null = null

  constructor(
    private readonly filePath: string,
    private readonly limits: WindowSizeLimits
  ) {}

  /** Last saved size, or the fallback when nothing usable is stored. */
  restore(fallback: WindowSize): WindowSize {
    try {
      return normalizeWindowSize(JSON.parse(readFileSync(this.filePath, 'utf8')), this.limits) ?? fallback
    } catch {
      return fallback
    }
  }

  /** Saves on every size change; writes are debounced because resizing is continuous. */
  track(window: BrowserWindow): void {
    const schedule = (): void => {
      // getBounds() reports the maximized size, which must never become the restored size.
      const bounds = window.isMaximized() ? window.getNormalBounds() : window.getBounds()
      this.pending = { width: bounds.width, height: bounds.height, maximized: window.isMaximized() }
      if (this.timer) return
      this.timer = setTimeout(() => this.flush(), SAVE_DELAY_MS)
    }
    window.on('resize', schedule)
    window.on('maximize', schedule)
    window.on('unmaximize', schedule)
  }

  /** Writes any pending change immediately. */
  flush(): void {
    if (this.timer) {
      clearTimeout(this.timer)
      this.timer = null
    }
    const size = this.pending
    this.pending = null
    if (!size) return
    try {
      writeFileSync(this.filePath, JSON.stringify(size), 'utf8')
    } catch {
      // A failed write must never break window handling or shutdown.
    }
  }
}
