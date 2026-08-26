#!/usr/bin/env node
import { mkdir, readdir, rm } from 'node:fs/promises'
import path from 'node:path'
import { spawn } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import { decodeNcmFile } from '../src/main/ncm.ts'

function parseArgs(argv) {
  const values = {
    input: path.resolve('测试用例'),
    output: path.resolve('.codex-tmp/msr-mvp/cases'),
    start: '60',
    duration: '20'
  }
  for (let index = 0; index < argv.length; index += 2) {
    const key = argv[index]?.replace(/^--/, '')
    const value = argv[index + 1]
    if (!key || !(key in values) || value === undefined) throw new Error(`INVALID_ARGUMENT:${argv[index] ?? ''}`)
    values[key] = value
  }
  return values
}

function run(executable, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(executable, args, { stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true })
    let stderr = ''
    child.stderr.on('data', (chunk) => { stderr += chunk.toString('utf8') })
    child.on('error', reject)
    child.on('close', (code) => code === 0 ? resolve() : reject(new Error(`FFMPEG_FAILED:${code}:${stderr.slice(-1600)}`)))
  })
}

async function main() {
  const options = parseArgs(process.argv.slice(2))
  const workspace = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
  const ffmpeg = path.join(workspace, 'resources', 'bin', process.platform === 'win32' ? 'ffmpeg.exe' : 'ffmpeg')
  await mkdir(options.output, { recursive: true })
  const entries = (await readdir(options.input, { withFileTypes: true }))
    .filter((entry) => entry.isFile() && ['.ncm', '.mp3', '.wav', '.flac', '.m4a', '.aac'].includes(path.extname(entry.name).toLowerCase()))
    .sort((left, right) => left.name.localeCompare(right.name, 'zh-CN'))
  const results = []
  for (const entry of entries) {
    const source = path.join(options.input, entry.name)
    const decoded = path.join(options.output, `.${path.parse(entry.name).name}.decoded`)
    const input = path.extname(entry.name).toLowerCase() === '.ncm' ? decoded : source
    const destination = path.join(options.output, `${path.parse(entry.name).name}.wav`)
    try {
      if (input === decoded) await decodeNcmFile(source, decoded)
      await run(ffmpeg, [
        '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
        '-ss', String(options.start), '-t', String(options.duration), '-i', input,
        '-vn', '-ar', '44100', '-ac', '2', '-c:a', 'pcm_s24le', destination
      ])
      results.push({ source: entry.name, output: destination })
      console.log(JSON.stringify({ type: 'progress', source: entry.name, output: destination }))
    } finally {
      if (input === decoded) await rm(decoded, { force: true })
    }
  }
  console.log(JSON.stringify({ type: 'result', files: results }))
}

main().catch((error) => {
  console.error(JSON.stringify({ type: 'error', message: error instanceof Error ? error.message : String(error) }))
  process.exitCode = 1
})

