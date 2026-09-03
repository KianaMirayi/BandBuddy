import { describe, expect, it } from 'vitest'
import {
  PYTHON_RUNTIME_REQUIREMENTS,
  PYTHON_RUNTIME_VERSIONS,
  pythonRuntimeRequirements,
  pythonRuntimeVersions
} from '../src/main/runtime-dependencies.js'

describe('managed Python runtime dependencies', () => {
  it('uses the current dependency set on Windows Python 3.12', () => {
    const windows = pythonRuntimeVersions('win32', 'x64')
    expect(windows).toEqual({
      torch: '2.11.0',
      torchaudio: '2.11.0',
      demucs: '4.1.0'
    })
    expect(pythonRuntimeRequirements(windows)).toEqual([
      'torch==2.11.0',
      'torchaudio==2.11.0',
      'demucs==4.1.0',
      'numpy==2.5.2',
      'scipy==1.17.0',
      'soundfile==0.14.0',
      'librosa==1.0.0',
      'PyYAML==6.0.3',
      'einops==0.8.1',
      'beartype==0.18.5',
      'rotary-embedding-torch==0.3.5',
      'packaging==26.2'
    ])
  })

  it('keeps the supported Apple Silicon runtime on the same fixed pair', () => {
    const intel = pythonRuntimeVersions('darwin', 'x64')
    const appleSilicon = pythonRuntimeVersions('darwin', 'arm64')
    expect(pythonRuntimeRequirements(intel).slice(0, 2)).toEqual(['torch==2.11.0', 'torchaudio==2.11.0'])
    expect(pythonRuntimeRequirements(appleSilicon).slice(0, 2)).toEqual(['torch==2.11.0', 'torchaudio==2.11.0'])
  })

  it('exports the dependency set for the running process', () => {
    expect(PYTHON_RUNTIME_VERSIONS).toEqual(pythonRuntimeVersions())
    expect(PYTHON_RUNTIME_REQUIREMENTS).toEqual(pythonRuntimeRequirements(PYTHON_RUNTIME_VERSIONS))
  })
})
