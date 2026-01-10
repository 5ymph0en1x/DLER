# DLER - Ultra-Fast Usenet Downloader

<div align="center">
    <img src="logo.png" width="300" alt="DLER">
</div>

High-performance NZB downloader with AVX2 acceleration, designed for maximum speed and reliability.

![DLER Screenshot](screenshots/gui.jpg)

## Features

- **Multi-threaded NNTP connections** - 50+ simultaneous connections for maximum bandwidth utilization
- **AVX2-optimized yEnc decoding** - Native C++ decoder achieving 3+ GB/s throughput
- **Automatic PAR2 verification & repair** - Integrated MultiPar for data integrity
- **Smart archive extraction** - Handles nested archives (ZIP containing RAR parts)
- **Modern Tkinter GUI** - Clean, responsive interface with real-time progress
- **Portable** - Bundled 7-Zip & PAR2 tools, no external dependencies

## Requirements

- **OS:** Windows 10/11 (64-bit)
- **Python:** 3.11+ (3.14 recommended for best performance)
- **CPU:** x86-64 with AVX2 support (Intel Haswell+ / AMD Excavator+)

## Quick Start

### From Source

1. Clone the repository:
   ```bash
   git clone https://github.com/5ymph0en1x/DLER.git
   cd DLER
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Build the native yEnc decoder for maximum performance:
   ```bash
   cd src/native
   python setup.py build_ext --inplace
   cd ../..
   ```

4. Run the application:
   ```bash
   python tk_main.py
   ```

### Configuration

On first launch, configure your Usenet provider:

1. Click the **Settings** button
2. Enter your NNTP server details:
   - Server address
   - Port (usually 563 for SSL)
   - Username & Password
   - Number of connections (start with 20, increase if stable)
3. Set your download directory
4. Click **Save**

## Building Standalone Executable

Build a portable Windows executable with PyInstaller:

```bash
# Install PyInstaller
pip install pyinstaller

# Build the executable
pyinstaller dler_tk.spec
```

The executable will be created in `dist/DLER.exe` (~120 MB).

## Architecture

```
DLER/
├── tk_main.py              # Application entry point
├── src/
│   ├── core/
│   │   ├── fast_nntp.py        # High-performance NNTP client
│   │   ├── turbo_engine_v2.py  # Download orchestration engine
│   │   ├── turbo_yenc.py       # Python yEnc decoder (fallback)
│   │   ├── post_processor.py   # PAR2 verification & extraction
│   │   └── nzb_parser.py       # NZB file parser
│   ├── gui/
│   │   └── tkinter_app.py      # Tkinter GUI
│   ├── utils/
│   │   └── config.py           # Configuration management
│   └── native/
│       ├── yenc_turbo.cpp      # AVX2 yEnc decoder (C++)
│       └── setup.py            # Build script
└── tools/
    ├── 7z.exe                  # 7-Zip (extraction)
    ├── 7z.dll
    └── par2j64.exe             # MultiPar (PAR2 verification)
```

## Performance

DLER is optimized for maximum throughput:

| Component | Performance |
|-----------|-------------|
| yEnc Decoding (AVX2) | 3+ GB/s |
| yEnc Decoding (Python) | ~200 MB/s |
| NNTP Connections | 50+ simultaneous |
| Memory Usage | ~500 MB typical |

The native AVX2 decoder provides **15x faster** yEnc decoding compared to pure Python.

## Native Extension (Optional)

For maximum performance, compile the AVX2 yEnc decoder:

```bash
cd src/native
python setup.py build_ext --inplace
```

Requirements:
- Visual Studio 2019+ with C++ workload
- Python development headers

See [src/native/README.md](src/native/README.md) for detailed build instructions.

Without the native extension, DLER falls back to a pure Python decoder which is still fast enough for most connections.

## Bundled Tools

DLER includes these tools for a portable, zero-dependency experience:

- **7-Zip** (LGPL) - Archive extraction
- **MultiPar/par2j64** - PAR2 verification and repair

## Troubleshooting

### Download stuck at 99%
This can happen with incomplete NZB files. DLER uses a 95% completion threshold to mark downloads as successful.

### Slow speeds
- Increase the number of connections in Settings
- Ensure your ISP doesn't throttle Usenet traffic
- Try a different server if available

### PAR2 repair fails
- Ensure enough PAR2 files were downloaded
- Check available disk space
- Some releases have insufficient parity data

## License

MIT License - See [LICENSE](LICENSE) for details.

## Credits

- **Author:** Symphoenix
- **yEnc specification:** Jeremy Nixon
- **7-Zip:** Igor Pavlov
- **MultiPar:** Yutaka Sawada

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

*DLER - Download at the speed of light*
