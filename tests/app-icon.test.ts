import { describe, expect, it } from 'vitest'
import { APP_ICON_DATA_URL } from '../src/main/app-icon.js'

describe('embedded application icon', () => {
  it('is bundled as a PNG data URL instead of a runtime file path', () => {
    expect(APP_ICON_DATA_URL).toMatch(/^data:image\/png;base64,/)
    expect(APP_ICON_DATA_URL.length).toBeGreaterThan(1_000)
  })
})
