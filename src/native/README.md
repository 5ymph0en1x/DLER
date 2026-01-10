# Native yEnc Decoder - Build Instructions

The native yEnc decoder uses AVX2 SIMD instructions to achieve 3+ GB/s decoding throughput, approximately 15x faster than the pure Python implementation.

## Requirements

- **Windows 10/11** (64-bit)
- **Visual Studio 2019+** with "Desktop development with C++" workload
- **Python 3.11+** with development headers

## Build Steps

### 1. Ensure Visual Studio is installed

Download from [Visual Studio](https://visualstudio.microsoft.com/downloads/) and install the "Desktop development with C++" workload.

### 2. Open Developer Command Prompt

Open "x64 Native Tools Command Prompt for VS 2022" (or your VS version).

### 3. Navigate to this directory

```cmd
cd path\to\DLER\src\native
```

### 4. Build the extension

```cmd
python setup.py build_ext --inplace
```

This will create `yenc_turbo.cpXXX-win_amd64.pyd` where XXX is your Python version.

### 5. Verify the build

```python
python -c "import yenc_turbo; print('Native decoder loaded!')"
```

## Troubleshooting

### "Unable to find vcvarsall.bat"

Visual Studio C++ tools are not installed. Install the "Desktop development with C++" workload.

### "Python.h not found"

Python development headers are missing. Reinstall Python with "Download debug binaries" option, or use:
```cmd
pip install --upgrade setuptools
```

### Performance verification

Run a quick benchmark:
```python
import yenc_turbo
import time

data = b'=' * (100 * 1024 * 1024)  # 100 MB
start = time.perf_counter()
yenc_turbo.decode(data)
elapsed = time.perf_counter() - start
print(f"Speed: {100 / elapsed:.1f} MB/s")
```

Expected output: 2000+ MB/s on modern CPUs with AVX2.

## Architecture Notes

The decoder uses:
- **AVX2** 256-bit SIMD for parallel byte processing
- **Lookup tables** for escape sequence handling
- **Streaming** design for minimal memory allocation

Without AVX2 support, DLER falls back to `src/core/turbo_yenc.py` which provides ~200 MB/s throughput.

## Files

- `yenc_turbo.cpp` - C++ source with AVX2 intrinsics
- `setup.py` - Python build configuration
