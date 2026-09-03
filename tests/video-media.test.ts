import { DatabaseSync } from 'node:sqlite'
import { existsSync, mkdirSync } from 'node:fs'
import { mkdtemp, readFile, rm } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest'
import { GUITAR_SPLIT_STEMS, LEGACY_STEM_ORDER } from '@shared/domain.js'
import { SOURCE_MEDIA_EXTENSIONS, VIDEO_EXTENSIONS, AUDIO_EXTENSIONS, isVideoSource } from '@shared/media-formats.js'
import { BandBuddyDatabase, DATABASE_MIGRATIONS } from '../src/main/database.js'
import { ImportService } from '../src/main/imports.js'
import { JobScheduler } from '../src/main/jobs.js'
import { MediaService } from '../src/main/media.js'
import { AppPaths } from '../src/main/paths.js'
import { runProcess } from '../src/main/process.js'

const electron = vi.hoisted(() => ({
  protocol: { handle: vi.fn() },
  dialog: { showOpenDialog: vi.fn() },
  Notification: { isSupported: () => false }
}))
vi.mock('electron', () => electron)
vi.mock('better-sqlite3', async () => {
  const { DatabaseSync } = await import('node:sqlite')
  // Match the driver's transaction API while exercising the real database
  // constructor, migrations, row mappings and services below.
  class TestDatabase extends DatabaseSync {
    pragma(statement: string): void { this.exec(`PRAGMA ${statement}`) }
    transaction(fn: (...args: unknown[]) => unknown) {
      const run = (...args: unknown[]) => {
        this.exec('BEGIN')
        try { const result = fn(...args); this.exec('COMMIT'); return result }
        catch (error) { this.exec('ROLLBACK'); throw error }
      }
      return Object.assign(run, { deferred: run, immediate: run, exclusive: run })
    }
  }
  return { default: TestDatabase }
})

describe('video input formats', () => {
  it('accepts common video sources without accepting videos as existing stems', () => {
    for (const extension of VIDEO_EXTENSIONS) {
      expect(SOURCE_MEDIA_EXTENSIONS.has(extension)).toBe(true)
      expect(AUDIO_EXTENSIONS.has(extension)).toBe(false)
      expect(isVideoSource(`C:/乐队 🎸/现场 ${extension.toUpperCase()}`)).toBe(true)
    }
    expect(isVideoSource('song.m4a')).toBe(false)
  })
})

const hasTools = existsSync(path.join(process.cwd(), 'resources', 'bin', process.platform === 'win32' ? 'ffmpeg.exe' : 'ffmpeg'))
describe.skipIf(!hasTools)('real local video preprocessing and library lifecycle', () => {
  let root: string
  let media: MediaService
  let paths: AppPaths
  let database: BandBuddyDatabase
  let source: string
  const logger = { error: vi.fn(), warn: vi.fn(), info: vi.fn() }

  beforeAll(async () => {
    root = await mkdtemp(path.join(os.tmpdir(), 'bandbuddy-video-'))
    paths = Object.assign(Object.create(AppPaths.prototype) as AppPaths, {
      defaultLibraryRoot: path.join(root, 'library'), pythonRoot: path.join(root, 'runtime'), modelRoot: path.join(root, 'models'),
      databasePath: path.join(root, 'library.db'), backupRoot: path.join(root, 'backups'),
      packagedResource: () => path.join(root, 'missing-tools')
    })
    mkdirSync(paths.backupRoot, { recursive: true })
    database = new BandBuddyDatabase(paths)
    media = new MediaService(paths, database, logger as never)
    expect(media.toolsReady()).toBe(true)
    source = path.join(root, '现场 测试 🎸.avi')
    const generated = await runProcess(media.tool('ffmpeg')!, [
      '-y', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=160x90:rate=10:duration=1.2',
      '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=44100:duration=1.2',
      '-c:v', 'mpeg4', '-c:a', 'pcm_s16le', '-shortest', source
    ])
    expect(generated.code, generated.stderr).toBe(0)
  }, 15_000)

  afterAll(async () => {
    database?.close()
    if (root) await rm(root, { recursive: true, force: true })
  })

  it('queues a copied video, feeds extracted WAV to separation, and serves seekable muted video', async () => {
    const workerInputs: string[] = []
    let releaseGuitar = (): void => undefined
    const guitarGate = new Promise<void>((resolve) => { releaseGuitar = resolve })
    const runtime = {
      onChange: vi.fn(),
      getInfo: () => ({ status: 'ready', selectedDevice: 'cpu' }),
      runWorker: vi.fn(async (args: string[]) => {
        const command = args[0]
        const workerInput = args[args.indexOf('--input') + 1]!
        workerInputs.push(workerInput)
        const probe = await media.probe(workerInput)
        expect(probe.video).toBeNull()
        expect(probe).toMatchObject({ sampleRate: 44100, channels: 2 })
        if (command === 'separate-guitar') await guitarGate
        const stems = command === 'separate-demucs' ? LEGACY_STEM_ORDER : GUITAR_SPLIT_STEMS
        return {
          code: 0,
          result: {
            files: Object.fromEntries(stems.map((stem) => [stem, workerInput])),
            stats: Object.fromEntries(stems.map((stem) => [stem, { peak: stem === 'drums' ? 1.36 : 0.8 }]))
          }
        }
      })
    }
    const normalizeSpy = vi.spyOn(media, 'normalize')
    const guitarCompleted = vi.fn()
    const scheduler = new JobScheduler(paths, database, runtime as never, media, logger as never, () => undefined, () => undefined, guitarCompleted)
    const imports = new ImportService(paths, database, media, runtime as never, logger as never, () => undefined, () => undefined)
    const imported = await imports.importSource({ filePath: source })
    expect(imported.songId).toBeTruthy()
    expect(database.getJob(imported.jobId!)?.payload).toMatchObject({ storageFormat: 'mp3_320' })
    // A task owns the setting snapshot captured when it was created.
    database.saveSettings({ ...database.getSettings(), highQualityStems: true })
    expect(database.getJob(imported.jobId!)?.payload).toMatchObject({ storageFormat: 'mp3_320' })
    const stored = database.getSongRow(imported.songId!)!
    expect(stored.source_rel_path).toMatch(/original\.avi$/)
    const originalCopy = paths.resolveLibraryPath(database.getSettings().libraryRoot, stored.source_rel_path!)
    expect(await readFile(originalCopy)).toEqual(await readFile(source))
    const duplicate = await imports.importSource({ filePath: source })
    expect(duplicate.duplicate?.id).toBe(imported.songId)
    scheduler.kick()
    await vi.waitFor(() => expect(['completed', 'failed']).toContain(database.getJob(imported.jobId!)?.status), { timeout: 15_000, interval: 40 })
    expect(database.getJob(imported.jobId!)?.status, database.getJob(imported.jobId!)?.errorMessage ?? '').toBe('completed')
    const partial = database.getSong(imported.songId!)!
    expect(partial.status).toBe('ready')
    expect(partial.stems.map((stem) => stem.type).sort()).toEqual([...LEGACY_STEM_ORDER].sort())
    expect(partial.guitarSplitStatus).toBe('pending')
    expect(normalizeSpy).toHaveBeenCalledTimes(6)
    const guitarJob = database.listJobs().find((job) => job.songId === imported.songId && job.type === 'guitarSplit')
    expect(guitarJob).toBeTruthy()
    releaseGuitar()
    await vi.waitFor(() => expect(database.getJob(guitarJob!.id)?.status).toBe('completed'), { timeout: 15_000, interval: 40 })
    expect(normalizeSpy).toHaveBeenCalledTimes(9)
    for (const call of normalizeSpy.mock.calls) expect(call[5]).toBeCloseTo(0.999 / 1.36, 10)
    expect(workerInputs).toHaveLength(2)
    expect(workerInputs.every((input) => /video-audio\.wav$/.test(input))).toBe(true)
    const song = database.getSong(imported.songId!)!
    expect(song.stems).toHaveLength(9)
    expect(song.guitarSplitStatus).toBe('ready')
    expect(guitarCompleted).toHaveBeenCalledWith(song.id)
    expect(song.stems.every((stem) => stem.peaksUrl !== null)).toBe(true)
    for (const stem of song.stems) {
      const asset = database.getStemAsset(song.id, stem.id)
      expect(asset?.peaksRelPath).toBeTruthy()
      expect(existsSync(paths.resolveLibraryPath(database.getSettings().libraryRoot, asset!.peaksRelPath!))).toBe(true)
    }
    expect(song.practice.guitarSplitEnabled).toBe(false)
    expect(Math.max(...song.stems.map((stem) => stem.durationMs)) - Math.min(...song.stems.map((stem) => stem.durationMs))).toBeLessThan(30)
    const mp3Files = database.getActiveStemFiles(song.id)
    expect(mp3Files).toHaveLength(9)
    expect(mp3Files.every((file) => file.relPath.endsWith('.mp3'))).toBe(true)
    const mp3Inspect = await runProcess(media.tool('ffprobe')!, [
      '-v', 'error', '-select_streams', 'a:0', '-show_entries', 'stream=codec_name,sample_rate,channels,bit_rate', '-of', 'json',
      paths.resolveLibraryPath(database.getSettings().libraryRoot, mp3Files[0]!.relPath)
    ])
    expect(mp3Inspect.code, mp3Inspect.stderr).toBe(0)
    expect(JSON.parse(mp3Inspect.stdout)).toMatchObject({
      streams: [{ codec_name: 'mp3', sample_rate: '44100', channels: 2, bit_rate: '320000' }]
    })
    expect(imports.requestGuitarSplit(song.id)).toBeNull()
    expect(database.getSong(song.id)?.practice.guitarSplitEnabled).toBe(true)
    expect(song.videoUrl).toBe(`bandbuddy-media://song/${song.id}/video`)
    const playbackPath = paths.resolveLibraryPath(database.getSettings().libraryRoot, database.getVideoRelative(song.id)!)
    const inspected = await runProcess(media.tool('ffprobe')!, ['-v', 'error', '-show_streams', '-of', 'json', playbackPath])
    const streams = JSON.parse(inspected.stdout).streams as Array<{ codec_type: string; codec_name: string }>
    expect(streams).toHaveLength(1)
    expect(streams[0]).toMatchObject({ codec_type: 'video', codec_name: 'vp9' })
    media.registerProtocol()
    const handler = electron.protocol.handle.mock.calls.at(-1)![1] as (request: Request) => Promise<Response>
    const response = await handler(new Request(song.videoUrl!, { headers: { Range: 'bytes=0-63' } }))
    expect(response.status).toBe(206)
    expect(response.headers.get('Content-Type')).toBe('video/webm')
    expect((await response.arrayBuffer()).byteLength).toBe(64)
    expect((await handler(new Request(`${song.videoUrl}/extra`))).status).toBe(404)
    expect((await handler(new Request('bandbuddy-media://song/not-a-song/video')))).toHaveProperty('status', 404)
    const previousVideo = database.getVideoRelative(song.id)
    const prepareSpy = vi.spyOn(media, 'prepareVideo')
    const retryId = imports.reSeparate(song.id)
    expect(database.getJob(retryId)?.payload).toMatchObject({ storageFormat: 'flac24' })
    database.saveSettings({ ...database.getSettings(), highQualityStems: false })
    expect(database.getJob(retryId)?.payload).toMatchObject({ storageFormat: 'flac24' })
    scheduler.kick()
    await vi.waitFor(() => expect(database.getJob(retryId)?.status).toBe('completed'), { timeout: 15_000, interval: 40 })
    const retryGuitarJob = database.listJobs().find((job) => job.songId === song.id && job.type === 'guitarSplit' && job.id !== guitarJob!.id)
    expect(retryGuitarJob).toBeTruthy()
    await vi.waitFor(() => expect(database.getJob(retryGuitarJob!.id)?.status).toBe('completed'), { timeout: 15_000, interval: 40 })
    expect(prepareSpy).not.toHaveBeenCalled()
    expect(database.getVideoRelative(song.id)).toBe(previousVideo)
    const flacFiles = database.getActiveStemFiles(song.id)
    expect(flacFiles).toHaveLength(9)
    expect(flacFiles.every((file) => file.relPath.endsWith('.flac'))).toBe(true)
    const flacInspect = await runProcess(media.tool('ffprobe')!, [
      '-v', 'error', '-select_streams', 'a:0', '-show_entries', 'stream=codec_name,sample_rate,channels,bits_per_raw_sample', '-of', 'json',
      paths.resolveLibraryPath(database.getSettings().libraryRoot, flacFiles[0]!.relPath)
    ])
    expect(flacInspect.code, flacInspect.stderr).toBe(0)
    expect(JSON.parse(flacInspect.stdout)).toMatchObject({
      streams: [{ codec_name: 'flac', sample_rate: '44100', channels: 2, bits_per_raw_sample: '24' }]
    })
    prepareSpy.mockRestore()
  }, 30_000)

  it('remuxes common H.264 MP4 without re-encoding or retaining its original audio', async () => {
    const input = path.join(root, 'common-h264.mp4')
    const generated = await runProcess(media.tool('ffmpeg')!, [
      '-y', '-v', 'error', '-i', source, '-c:v', 'libopenh264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', input
    ])
    expect(generated.code, generated.stderr).toBe(0)
    const playback = await media.prepareVideo(input, root, path.join(root, 'copied-h264'), new AbortController().signal)
    expect(playback).toMatch(/\.mp4$/)
    const inspected = await runProcess(media.tool('ffprobe')!, ['-v', 'error', '-show_streams', '-of', 'json', playback])
    const streams = JSON.parse(inspected.stdout).streams
    expect(streams).toHaveLength(1)
    expect(streams[0]).toMatchObject({ codec_type: 'video', codec_name: 'h264', pix_fmt: 'yuv420p' })
    const packets = await Promise.all([input, playback].map((file) => runProcess(media.tool('ffprobe')!, [
      '-v', 'error', '-select_streams', 'v:0', '-show_packets', '-show_data_hash', 'sha256',
      '-show_entries', 'packet=data_hash', '-of', 'json', file
    ])))
    for (const packet of packets) expect(packet.code, packet.stderr).toBe(0)
    expect(JSON.parse(packets[1]!.stdout).packets).toEqual(JSON.parse(packets[0]!.stdout).packets)
    const audio = path.join(root, 'h264-audio.wav')
    await media.extractVideoAudio(input, path.join(root, 'h264-audio.part.wav'), audio, new AbortController().signal)
    expect((await media.probe(audio)).durationMs).toBeCloseTo((await media.probe(input)).durationMs, -1)
  })

  it('preserves an audio track delayed relative to the picture', async () => {
    const input = path.join(root, 'delayed-audio.mkv')
    const generated = await runProcess(media.tool('ffmpeg')!, [
      '-y', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=96x64:rate=10:duration=1',
      '-itsoffset', '0.3', '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=44100:duration=0.7',
      '-c:v', 'libvpx', '-deadline', 'realtime', '-c:a', 'pcm_s16le', input
    ])
    expect(generated.code, generated.stderr).toBe(0)
    const output = path.join(root, 'delayed-audio.wav')
    await media.extractVideoAudio(input, path.join(root, 'delayed.part.wav'), output, new AbortController().signal)
    const silence = await media.leadingSilenceMs(output)
    expect(silence).toBeGreaterThanOrEqual(290)
    expect(silence).toBeLessThanOrEqual(320)
    expect((await media.probe(output)).durationMs).toBeGreaterThanOrEqual(990)
    const playback = await media.prepareVideo(input, root, path.join(root, 'copied-vp8'), new AbortController().signal)
    const inspected = await runProcess(media.tool('ffprobe')!, ['-v', 'error', '-show_streams', '-of', 'json', playback])
    expect(JSON.parse(inspected.stdout).streams[0].codec_name).toBe('vp8')
  })

  it('reprocesses a legacy six-stem song for guitar mode and preserves it on failure', async () => {
    database.saveSettings({ ...database.getSettings(), highQualityStems: false })
    const songId = '80000000-0000-4000-8000-000000000000'
    database.createSong({
      title: '旧六轨', artist: '', sourceRelPath: `${songId}/source/original.mp3`, sourceHash: 'legacy',
      sourceFormat: 'mp3', durationMs: 1_000, sampleRate: 44_100, channels: 2, artworkRelPath: null,
      status: 'ready', phase: null
    }, songId)
    const oldJob = database.createJob('separate', songId, 'queued', '旧任务', {})
    database.activateSeparation(songId, oldJob, 'bandbuddy-stems:v1.3.0', 'cpu', LEGACY_STEM_ORDER.map((type, index) => ({
      id: `${index + 1}0000000-0000-4000-8000-000000000000`, type,
      relPath: `${songId}/versions/old/${type}.flac`, peaksRelPath: null,
      durationMs: 1_000, sampleRate: 44_100, channels: 2
    })))

    const runtime = {
      onChange: vi.fn(),
      getInfo: () => ({ status: 'ready', selectedDevice: 'cpu' }),
      runWorker: vi.fn(async () => ({ code: 1, result: {}, error: 'synthetic worker failure' }))
    }
    const imports = new ImportService(paths, database, media, runtime as never, logger as never, () => undefined, () => undefined)
    const jobId = imports.requestGuitarSplit(songId)
    expect(jobId).toBeTruthy()
    expect(database.getJob(jobId!)?.type).toBe('guitarSplit')
    expect(database.getJob(jobId!)?.payload).toMatchObject({
      storageFormat: 'mp3_320', baseSeparationId: expect.any(String),
      expectedActiveSeparationId: expect.any(String), enableGuitarSplitOnSuccess: true
    })
    const scheduler = new JobScheduler(paths, database, runtime as never, media, logger as never, () => undefined, () => undefined)
    scheduler.kick()
    await vi.waitFor(() => expect(database.getJob(jobId!)?.status).toBe('failed'), { timeout: 5_000, interval: 25 })
    const preserved = database.getSong(songId)!
    expect(preserved.status).toBe('ready')
    expect(preserved.phase).toBe('吉他细分未完成，基础分轨可继续')
    expect(preserved.guitarSplitStatus).toBe('failed')
    expect(preserved.stems.map((stem) => stem.type).sort()).toEqual([...LEGACY_STEM_ORDER].sort())
    expect(preserved.practice.guitarSplitEnabled).toBe(false)
  })

  it('does not offer guitar reprocessing when a historical song has no source audio', () => {
    const songId = '90000000-0000-4000-8000-000000000000'
    database.createSong({
      title: '仅历史分轨', artist: '', sourceRelPath: null, sourceHash: null, sourceFormat: null,
      durationMs: 1_000, sampleRate: 44_100, channels: 2, artworkRelPath: null,
      status: 'ready', phase: null
    }, songId)
    const runtime = { getInfo: () => ({ status: 'ready' }) }
    const imports = new ImportService(paths, database, media, runtime as never, logger as never, () => undefined, () => undefined)
    expect(() => imports.requestGuitarSplit(songId)).toThrow('ORIGINAL_SOURCE_NOT_AVAILABLE')
  })

  it('rejects silent videos and cancels extraction without publishing partial output', async () => {
    const silent = path.join(root, 'silent.webm')
    const generated = await runProcess(media.tool('ffmpeg')!, [
      '-y', '-v', 'error', '-i', source, '-an', '-c:v', 'libvpx', '-deadline', 'realtime', silent
    ])
    expect(generated.code, generated.stderr).toBe(0)
    await expect(media.probe(silent)).rejects.toThrow('NO_AUDIO_STREAM')
    const controller = new AbortController()
    controller.abort()
    const temporary = path.join(root, 'cancelled.part.wav')
    const final = path.join(root, 'cancelled.wav')
    await expect(media.extractVideoAudio(source, temporary, final, controller.signal)).rejects.toThrow('JOB_CANCELLED')
    expect(existsSync(temporary)).toBe(false)
    expect(existsSync(final)).toBe(false)
  })

  it('upgrades a previous library without changing its songs or saved practice state', () => {
    const legacyPath = path.join(root, 'legacy.db')
    const old = new DatabaseSync(legacyPath)
    for (const migration of DATABASE_MIGRATIONS.slice(0, -1)) old.exec(migration)
    old.exec('CREATE TABLE schema_version (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)')
    old.prepare('INSERT INTO schema_version VALUES (?, ?)').run(DATABASE_MIGRATIONS.length - 1, '2026-08-01')
    old.prepare('INSERT INTO songs(id, title, created_at, updated_at) VALUES (?, ?, ?, ?)')
      .run('old-song', '已有音频', '2026-08-01', '2026-08-01')
    old.prepare('INSERT INTO practice_states(song_id, state_json, updated_at) VALUES (?, ?, ?)')
      .run('old-song', JSON.stringify({
        positionMs: 12000, playbackRate: 0.8, loopEnabled: true, loopStartMs: 5000, loopEndMs: 15000,
        selectedStem: 'guitar', tracks: [{ stemType: 'guitar', gainDb: -5, muted: false, solo: true, outputChannelPair: 3 }]
      }), '2026-08-01')
    old.prepare("INSERT INTO settings(key, value_json, updated_at) VALUES ('app', ?, ?)")
      .run(JSON.stringify({ debugMode: true }), '2026-08-01')
    old.close()
    const upgraded = new BandBuddyDatabase({ ...paths, databasePath: legacyPath } as AppPaths)
    try {
      const song = upgraded.getSong('old-song')!
      expect(song.title).toBe('已有音频')
      expect(song.videoUrl).toBeNull()
      expect(song.practice).toMatchObject({ positionMs: 12000, playbackRate: 0.8, loopEnabled: true, loopStartMs: 5000, loopEndMs: 15000 })
      expect(song.practice.guitarSplitEnabled).toBe(false)
      expect(song.practice.tracks).toHaveLength(9)
      expect(song.practice.tracks.find((track) => track.stemType === 'guitar')).toMatchObject({ gainDb: -5, solo: true, outputChannelPair: 3 })
      expect(song.practice.tracks.find((track) => track.stemType === 'lead_guitar')).toMatchObject({ gainDb: 0, muted: false, solo: false })
      expect(upgraded.getSettings()).toMatchObject({ debugMode: true, highQualityStems: false })
    } finally { upgraded.close() }
  })
})
