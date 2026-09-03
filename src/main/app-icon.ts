import appIconDataUrl from '../../build/icon.png?inline'

// `?inline` makes Vite place the PNG data directly in the main-process bundle.
// The installed app therefore does not need a loose icon.png at runtime.
export const APP_ICON_DATA_URL = appIconDataUrl
