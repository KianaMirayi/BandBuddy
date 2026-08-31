export const AUDIO_EXTENSIONS = new Set(['.mp3', '.wav', '.flac', '.m4a', '.aac'])
export const SOURCE_AUDIO_EXTENSIONS = new Set([...AUDIO_EXTENSIONS, '.ncm'])
export const VIDEO_EXTENSIONS = new Set(['.mp4', '.m4v', '.mov', '.mkv', '.webm', '.avi'])
export const SOURCE_MEDIA_EXTENSIONS = new Set([...SOURCE_AUDIO_EXTENSIONS, ...VIDEO_EXTENSIONS])

export function isVideoSource(filePath: string): boolean {
  return VIDEO_EXTENSIONS.has(filePath.slice(filePath.lastIndexOf('.')).toLowerCase())
}
