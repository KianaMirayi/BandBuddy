// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from '../src/renderer/src/App.js'
import { ExportDialog, ImportDialog, MetadataDialog, SettingsDrawer, SongActionsDialog } from '../src/renderer/src/components/Dialogs.js'
import { fixtureDetail, fixtureSongs } from '../src/renderer/src/fixtures.js'
import { installFixtureBridge } from '../src/renderer/src/mock-bridge.js'

describe('library dialogs', () => {
  beforeEach(() => {
    Object.defineProperty(window, 'bandbuddy', { configurable: true, writable: true, value: undefined })
    installFixtureBridge()
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('starts each import session with fresh metadata', async () => {
    const choices = [
      { path: 'C:/Music/歌曲 A.mp3', name: '歌曲 A.mp3', inferredTitle: '歌曲 A' },
      { path: 'C:/Music/歌曲 B.mp3', name: '歌曲 B.mp3', inferredTitle: '歌曲 B' }
    ]
    vi.spyOn(window.bandbuddy.library, 'chooseSource').mockImplementation(async () => choices.shift() ?? null)
    const props = {
      onOpenChange: vi.fn(), onImported: vi.fn(), onOpenDuplicate: vi.fn(), onNeedsRuntime: vi.fn()
    }
    const { rerender } = render(<ImportDialog open {...props} />)

    fireEvent.click(screen.getByText('选择音频或视频文件'))
    await waitFor(() => expect((screen.getByLabelText('歌曲标题') as HTMLInputElement).value).toBe('歌曲 A'))
    fireEvent.change(screen.getByLabelText('艺术家'), { target: { value: '艺术家 A' } })

    rerender(<ImportDialog open={false} {...props} />)
    rerender(<ImportDialog open {...props} />)
    await waitFor(() => expect((screen.getByLabelText('歌曲标题') as HTMLInputElement).value).toBe(''))
    expect((screen.getByLabelText('艺术家') as HTMLInputElement).value).toBe('')

    fireEvent.click(screen.getByText('选择音频或视频文件'))
    await waitFor(() => expect((screen.getByLabelText('歌曲标题') as HTMLInputElement).value).toBe('歌曲 B'))
    expect(screen.queryByText(/导入已有分轨/)).toBeNull()
  })

  it('offers metadata editing from the song actions menu', () => {
    const onOpenChange = vi.fn()
    const onEditMetadata = vi.fn()
    render(<SongActionsDialog
      open
      onOpenChange={onOpenChange}
      song={fixtureSongs[0]!}
      onOpen={() => undefined}
      onEditMetadata={onEditMetadata}
      onImportLyrics={() => undefined}
      onReveal={() => undefined}
      onReseparate={() => undefined}
      onDelete={() => undefined}
    />)

    fireEvent.click(screen.getByRole('button', { name: /编辑歌曲信息/ }))
    expect(onOpenChange).toHaveBeenCalledWith(false)
    expect(onEditMetadata).toHaveBeenCalledTimes(1)
  })

  it('opens the selected library song metadata from its three-dot menu', async () => {
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(<QueryClientProvider client={queryClient}><App /></QueryClientProvider>)

    await screen.findByRole('heading', { name: '曲库' })
    const menus = screen.getAllByRole('button', { name: '歌曲菜单' })
    fireEvent.click(menus[0]!)

    const editButton = await screen.findByRole('button', { name: /编辑歌曲信息/ })
    fireEvent.click(editButton)
    await waitFor(() => expect((screen.getByLabelText('歌曲标题') as HTMLInputElement).value).toBe(fixtureSongs[0]!.title))
  })

  it('shows low-confidence key candidates and saves a manual correction', async () => {
    const song = fixtureDetail(fixtureSongs[0]!)
    const update = vi.spyOn(window.bandbuddy.library, 'update')
    render(<MetadataDialog open onOpenChange={() => undefined} song={song} onSaved={() => undefined} />)

    fireEvent.click(screen.getByRole('button', { name: '识别歌曲调' }))
    await screen.findByText('置信度较低，前三候选')
    expect(screen.getByText(/E minor/)).toBeTruthy()
    expect(screen.getByText('可能转调')).toBeTruthy()

    fireEvent.change(screen.getByLabelText('主音'), { target: { value: 'A' } })
    fireEvent.change(screen.getByLabelText('调式'), { target: { value: 'minor' } })
    fireEvent.click(screen.getByRole('button', { name: '保存' }))
    await waitFor(() => expect(update).toHaveBeenCalledWith(expect.objectContaining({
      id: song.id,
      patch: expect.objectContaining({ musicalKey: 'A minor', musicalKeySource: 'manual' })
    })))
  })

  it('always carries the current pitch into stem exports', async () => {
    const song = fixtureDetail(fixtureSongs[0]!)
    const practice = { ...song.practice, pitchSemitones: 4 }
    vi.spyOn(window.bandbuddy.export, 'choosePath').mockResolvedValue('C:/Exports')
    const start = vi.spyOn(window.bandbuddy.export, 'start')
    render(<ExportDialog open onOpenChange={() => undefined} song={song} practice={practice} onBeforeStart={async () => undefined} />)

    fireEvent.click(screen.getByRole('button', { name: /分别导出音轨/ }))
    expect(screen.getByText('导出当前 +4 半音')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: /选择位置并导出/ }))
    await waitFor(() => expect(start).toHaveBeenCalledWith(expect.objectContaining({
      kind: 'stems', applyPitchShift: true, pitchSemitones: 4
    })))
  })

  it('defaults stem export to the visible guitar mode and can include hidden alternatives', async () => {
    const song = fixtureDetail(fixtureSongs[0]!)
    vi.spyOn(window.bandbuddy.export, 'choosePath').mockResolvedValue('C:/Exports')
    const start = vi.spyOn(window.bandbuddy.export, 'start')
    render(<ExportDialog open onOpenChange={() => undefined} song={song} practice={song.practice} onBeforeStart={async () => undefined} />)

    fireEvent.click(screen.getByRole('button', { name: /分别导出音轨/ }))
    expect(screen.getByText(/包含隐藏吉他备选轨/).textContent).toContain('Acoustic / Lead / Rhythm')
    fireEvent.click(screen.getByRole('checkbox', { name: /包含隐藏吉他备选轨/ }))
    fireEvent.click(screen.getByRole('button', { name: /选择位置并导出/ }))
    await waitFor(() => expect(start).toHaveBeenCalledWith(expect.objectContaining({
      kind: 'stems',
      stemTypes: expect.arrayContaining(['guitar', 'acoustic_guitar', 'lead_guitar', 'rhythm_guitar'])
    })))
    expect(start.mock.calls[0]![0].stemTypes).toHaveLength(9)
  })

  it('saves the high-quality switch as a future-task setting', async () => {
    const [settings, runtime] = await Promise.all([
      window.bandbuddy.settings.get(),
      window.bandbuddy.runtime.get()
    ])
    const update = vi.spyOn(window.bandbuddy.settings, 'update').mockImplementation(async (next) => next)
    render(<SettingsDrawer
      open
      onOpenChange={() => undefined}
      runtime={runtime}
      settings={{ ...settings, highQualityStems: false }}
      onSaved={() => undefined}
      onRefresh={() => undefined}
    />)

    expect(screen.getByText('新分轨保存为 320 kbps MP3，节省空间')).toBeTruthy()
    expect(screen.getByText('仅影响后续分轨；已有歌曲需重新分轨才会改变格式')).toBeTruthy()
    fireEvent.click(screen.getByRole('checkbox', { name: '高音质分轨' }))
    expect(screen.getByText('新分轨保存为 24-bit FLAC，占用空间较大')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: '保存设置' }))
    await waitFor(() => expect(update).toHaveBeenCalledWith(expect.objectContaining({ highQualityStems: true })))
  })

  it('saves the draggable guitar quality tier for future tasks', async () => {
    const [settings, runtime] = await Promise.all([
      window.bandbuddy.settings.get(),
      window.bandbuddy.runtime.get()
    ])
    const update = vi.spyOn(window.bandbuddy.settings, 'update').mockImplementation(async (next) => next)
    render(<SettingsDrawer
      open
      onOpenChange={() => undefined}
      runtime={runtime}
      settings={{ ...settings, guitarSeparationQuality: 'balanced' }}
      onSaved={() => undefined}
      onRefresh={() => undefined}
    />)

    const slider = screen.getByRole('slider', { name: '吉他分轨档位' })
    expect(slider.getAttribute('aria-valuetext')).toBe('平衡')
    fireEvent.change(slider, { target: { value: '0' } })
    expect(screen.getByText(/快速\/预览质量/)).toBeTruthy()
    expect(slider.getAttribute('aria-valuetext')).toBe('极速')
    fireEvent.click(screen.getByRole('button', { name: '保存设置' }))
    await waitFor(() => expect(update).toHaveBeenCalledWith(expect.objectContaining({ guitarSeparationQuality: 'fast' })))
  })

  it('enables debug mode immediately and reveals the debug log from settings', async () => {
    const [settings, runtime] = await Promise.all([
      window.bandbuddy.settings.get(),
      window.bandbuddy.runtime.get()
    ])
    const update = vi.spyOn(window.bandbuddy.settings, 'update')
    const setDebugMode = vi.spyOn(window.bandbuddy.settings, 'setDebugMode').mockImplementation(async (enabled) => ({ ...settings, debugMode: enabled }))
    const revealDebugLog = vi.spyOn(window.bandbuddy.settings, 'revealDebugLog')
    const onSaved = vi.fn()

    render(<SettingsDrawer
      open
      onOpenChange={() => undefined}
      runtime={runtime}
      settings={settings}
      onSaved={onSaved}
      onRefresh={() => undefined}
    />)

    fireEvent.click(screen.getByRole('checkbox', { name: 'Debug 模式' }))
    await waitFor(() => expect(setDebugMode).toHaveBeenCalledWith(true))
    expect(onSaved).toHaveBeenCalledWith(expect.objectContaining({ debugMode: true }))
    expect(update).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: '打开日志位置' }))
    await waitFor(() => expect(revealDebugLog).toHaveBeenCalledTimes(1))
  })
})
