# -*- coding: utf-8 -*-
# DLER Version Info for PyInstaller
# This file generates Windows version information for the EXE

VSVersionInfo(
    ffi=FixedFileInfo(
        # Version numbers (Major, Minor, Patch, Build)
        filevers=(1, 0, 3, 0),
        prodvers=(1, 0, 3, 0),
        # Bitmask: 0x3F = all flags valid
        mask=0x3f,
        # Flags: 0 = release build
        flags=0x0,
        # OS: VOS_NT_WINDOWS32 = Windows NT
        OS=0x40004,
        # File type: VFT_APP = Application
        fileType=0x1,
        # Subtype (not used for apps)
        subtype=0x0,
        # Creation date (not used)
        date=(0, 0)
    ),
    kids=[
        StringFileInfo(
            [
                StringTable(
                    '040904B0',  # Lang: US English, Charset: Unicode
                    [
                        StringStruct('CompanyName', 'Symphoenix'),
                        StringStruct('FileDescription', 'DLER - Ultra-Fast Usenet Downloader'),
                        StringStruct('FileVersion', '1.0.3.0'),
                        StringStruct('InternalName', 'DLER'),
                        StringStruct('LegalCopyright', '2026 Symphoenix. All rights reserved.'),
                        StringStruct('OriginalFilename', 'DLER.exe'),
                        StringStruct('ProductName', 'DLER'),
                        StringStruct('ProductVersion', '1.0.3'),
                        StringStruct('Comments', 'High-performance NZB downloader with AVX2 acceleration'),
                    ]
                )
            ]
        ),
        VarFileInfo([VarStruct('Translation', [0x0409, 1200])])  # US English, Unicode
    ]
)
