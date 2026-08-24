import { appendFileSync, mkdirSync } from 'node:fs'
import path from 'node:path'

const SENSITIVE = /(https?:\/\/)([^\s/:@]+):([^\s/@]+)@/gi
type LogLevel = 'debug' | 'info' | 'warn' | 'error'

export class Logger {
  private readonly applicationLogPath: string
  readonly debugLogPath: string

  constructor(private readonly logsRoot: string, private debugMode = false) {
    mkdirSync(logsRoot, { recursive: true })
    this.applicationLogPath = path.join(logsRoot, 'bandbuddy.log')
    this.debugLogPath = path.join(logsRoot, 'debug.log')
    if (debugMode) this.capture('debug', 'debug logging session started', { pid: process.pid })
  }

  private redact(value: unknown): string {
    let text: string
    if (typeof value === 'string') text = value
    else if (value instanceof Error) text = `${value.name}: ${value.message}${value.stack ? `\n${value.stack}` : ''}`
    else {
      try {
        text = JSON.stringify(value, (_key, nested) => nested instanceof Error
          ? { name: nested.name, message: nested.message, stack: nested.stack }
          : nested) ?? String(value)
      }
      catch { text = String(value) }
    }
    return text.replace(SENSITIVE, '$1***:***@').replace(/(token|password|authorization)["'=:\s]+[^\s",}]+/gi, '$1=***')
  }

  private format(level: LogLevel, message: string, detail?: unknown): string {
    return JSON.stringify({
      at: new Date().toISOString(),
      level,
      message: this.redact(message),
      ...(detail === undefined ? {} : { detail: this.redact(detail) })
    })
  }

  private append(filePath: string, line: string): void {
    appendFileSync(filePath, `${line}\n`, 'utf8')
  }

  write(level: Exclude<LogLevel, 'debug'>, message: string, detail?: unknown): void {
    const line = this.format(level, message, detail)
    this.append(this.applicationLogPath, line)
    if (this.debugMode) this.append(this.debugLogPath, line)
  }

  capture(level: LogLevel, message: string, detail?: unknown): void {
    if (!this.debugMode) return
    this.append(this.debugLogPath, this.format(level, message, detail))
  }

  setDebugMode(enabled: boolean): void {
    if (enabled === this.debugMode) return
    if (enabled) {
      this.debugMode = true
      this.capture('info', 'debug mode enabled', { pid: process.pid })
    } else {
      this.capture('info', 'debug mode disabled')
      this.debugMode = false
    }
  }

  ensureDebugLog(): string {
    mkdirSync(this.logsRoot, { recursive: true })
    appendFileSync(this.debugLogPath, '', 'utf8')
    return this.debugLogPath
  }

  info(message: string, detail?: unknown): void { this.write('info', message, detail) }
  warn(message: string, detail?: unknown): void { this.write('warn', message, detail) }
  error(message: string, detail?: unknown): void { this.write('error', message, detail) }
}
