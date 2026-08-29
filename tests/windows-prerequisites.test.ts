import { describe, expect, it } from 'vitest'
import {
  compareRuntimeVersions,
  isTrustedMicrosoftSignature,
  isWindowsNativeRuntimeError,
  parseAuthenticodeInfo,
  parseVcRuntimeRegistry
} from '../src/main/windows-prerequisites.js'

describe('Windows native prerequisites', () => {
  it('detects a supported x64 Visual C++ runtime from registry output', () => {
    const output = `
HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\VisualStudio\\14.0\\VC\\Runtimes\\x64
    Version    REG_SZ    v14.44.35211.0
    Installed    REG_DWORD    0x1
`
    expect(parseVcRuntimeRegistry(output)).toEqual({
      installed: true,
      version: 'v14.44.35211.0',
      supported: true
    })
  })

  it('rejects missing, disabled, old, and malformed runtime versions', () => {
    expect(parseVcRuntimeRegistry('ERROR: The system was unable to find the registry key')).toEqual({
      installed: false, version: null, supported: false
    })
    expect(parseVcRuntimeRegistry('Version REG_SZ v14.50.1.0\nInstalled REG_DWORD 0x0').supported).toBe(false)
    expect(parseVcRuntimeRegistry('Version REG_SZ v14.40.33810.0\nInstalled REG_DWORD 0x1').supported).toBe(false)
    expect(Number.isNaN(compareRuntimeVersions('invalid', '14.44'))).toBe(true)
  })

  it('accepts only a valid Microsoft Authenticode signature', () => {
    const valid = parseAuthenticodeInfo(JSON.stringify({
      status: 'Valid',
      subject: 'CN=Microsoft Corporation, O=Microsoft Corporation, L=Redmond, S=Washington, C=US'
    }))
    expect(isTrustedMicrosoftSignature(valid)).toBe(true)
    expect(isTrustedMicrosoftSignature({ status: 'HashMismatch', subject: valid!.subject })).toBe(false)
    expect(isTrustedMicrosoftSignature({ status: 'Valid', subject: 'CN=Example Corporation' })).toBe(false)
    expect(parseAuthenticodeInfo('not json')).toBeNull()
  })

  it('recognizes PyTorch and MSVC DLL initialization failures', () => {
    expect(isWindowsNativeRuntimeError('OSError: [WinError 1114] Error loading c10.dll')).toBe(true)
    expect(isWindowsNativeRuntimeError('VCRUNTIME140_1.dll was not found')).toBe(true)
    expect(isWindowsNativeRuntimeError('MODEL_DOWNLOAD_FAILED')).toBe(false)
  })
})
