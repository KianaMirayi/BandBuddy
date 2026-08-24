import { existsSync, mkdtempSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import { Logger } from '../src/main/logger.js'

const temporaryRoots: string[] = []

afterEach(() => {
  for (const root of temporaryRoots.splice(0)) rmSync(root, { recursive: true, force: true })
})

describe('debug logger', () => {
  it('captures detailed logs only while debug mode is enabled', () => {
    const root = mkdtempSync(path.join(tmpdir(), 'bandbuddy-logger-'))
    temporaryRoots.push(root)
    const logger = new Logger(root)

    logger.info('before debug')
    expect(existsSync(logger.debugLogPath)).toBe(false)

    logger.setDebugMode(true)
    logger.info('during debug')
    logger.capture('error', 'renderer console', { error: new Error('render failed'), token: 'secret-value' })
    logger.setDebugMode(false)
    logger.warn('after debug')

    const debugLog = readFileSync(logger.debugLogPath, 'utf8')
    expect(debugLog).toContain('debug mode enabled')
    expect(debugLog).toContain('during debug')
    expect(debugLog).toContain('renderer console')
    expect(debugLog).toContain('render failed')
    expect(debugLog).toContain('debug mode disabled')
    expect(debugLog).not.toContain('before debug')
    expect(debugLog).not.toContain('after debug')
    expect(debugLog).not.toContain('secret-value')

    const applicationLog = readFileSync(path.join(root, 'bandbuddy.log'), 'utf8')
    expect(applicationLog).toContain('before debug')
    expect(applicationLog).toContain('during debug')
    expect(applicationLog).toContain('after debug')
    expect(applicationLog).not.toContain('renderer console')
  })

  it('creates an empty debug log on demand so the app can open it', () => {
    const root = mkdtempSync(path.join(tmpdir(), 'bandbuddy-logger-'))
    temporaryRoots.push(root)
    const logger = new Logger(root)

    expect(logger.ensureDebugLog()).toBe(path.join(root, 'debug.log'))
    expect(readFileSync(logger.debugLogPath, 'utf8')).toBe('')
  })
})
