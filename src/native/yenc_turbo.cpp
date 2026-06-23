/*
 * yenc_turbo.cpp - Ultra-fast yEnc decoder with AVX2 SIMD
 * ========================================================
 *
 * Optimized for 10 Gbps throughput on modern CPUs.
 * Processes 32 bytes at a time with AVX2 vectorization.
 *
 * Performance:
 *   - Scalar: ~500 MB/s
 *   - AVX2:   ~4-5 GB/s (8x faster)
 *
 * Compile with MSVC (Windows):
 *   cl /O2 /GL /arch:AVX2 /LD /EHsc yenc_turbo.cpp /I<python_include> /link /LIBPATH:<python_lib> python3.lib
 *
 * Compile with GCC/Clang (Linux):
 *   g++ -O3 -march=native -mavx2 -shared -fPIC -o yenc_turbo.so yenc_turbo.cpp $(python3-config --includes --ldflags)
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cstdint>
#include <cstring>
#include <immintrin.h>  // AVX2 intrinsics

// Pre-computed decode tables
static uint8_t DECODE_TABLE[256];
static uint8_t ESCAPE_TABLE[256];
static uint32_t CRC32_TABLE[16][256];  // slice-by-16 (IEEE poly, reflected)
static bool tables_initialized = false;

// Runtime AVX2 detection
static bool has_avx2 = false;

// =============================================================================
// CPU Feature Detection
// =============================================================================
#ifdef _MSC_VER
#include <intrin.h>
static void detect_avx2() {
    int cpuInfo[4];
    __cpuid(cpuInfo, 0);
    int nIds = cpuInfo[0];
    if (nIds >= 7) {
        __cpuidex(cpuInfo, 7, 0);
        has_avx2 = (cpuInfo[1] & (1 << 5)) != 0;  // AVX2 bit in EBX
    }
}
#else
static void detect_avx2() {
    __builtin_cpu_init();
    has_avx2 = __builtin_cpu_supports("avx2");
}
#endif

// =============================================================================
// Initialize Lookup Tables
// =============================================================================
static void init_tables() {
    if (tables_initialized) return;

    // yEnc decode table: (byte - 42) & 0xFF
    for (int i = 0; i < 256; i++) {
        DECODE_TABLE[i] = (uint8_t)((i - 42) & 0xFF);
    }

    // Escape table: (byte - 64 - 42) & 0xFF
    for (int i = 0; i < 256; i++) {
        ESCAPE_TABLE[i] = (uint8_t)((i - 64 - 42) & 0xFF);
    }

    // CRC32 slice-by-16 tables (IEEE polynomial 0xEDB88320, reflected).
    // table[0] is the classic byte table; table[n] folds n+1 bytes ahead.
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t crc = i;
        for (int j = 0; j < 8; j++) {
            crc = (crc & 1) ? ((crc >> 1) ^ 0xEDB88320) : (crc >> 1);
        }
        CRC32_TABLE[0][i] = crc;
    }
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t crc = CRC32_TABLE[0][i];
        for (int n = 1; n < 16; n++) {
            crc = (crc >> 8) ^ CRC32_TABLE[0][crc & 0xFF];
            CRC32_TABLE[n][i] = crc;
        }
    }

    detect_avx2();
    tables_initialized = true;
}

// =============================================================================
// Fast CRC32 (slice-by-16) — ~5-7 GB/s, well above a 10 Gbps (1.25 GB/s) link.
// Produces the standard IEEE CRC32 (identical to zlib.crc32).
// =============================================================================
static uint32_t crc32_compute(const uint8_t* p, Py_ssize_t len) {
    uint32_t crc = 0xFFFFFFFF;

    while (len >= 16) {
        uint32_t w0 = (uint32_t)p[0]  | ((uint32_t)p[1]  << 8) | ((uint32_t)p[2]  << 16) | ((uint32_t)p[3]  << 24);
        uint32_t w1 = (uint32_t)p[4]  | ((uint32_t)p[5]  << 8) | ((uint32_t)p[6]  << 16) | ((uint32_t)p[7]  << 24);
        uint32_t w2 = (uint32_t)p[8]  | ((uint32_t)p[9]  << 8) | ((uint32_t)p[10] << 16) | ((uint32_t)p[11] << 24);
        uint32_t w3 = (uint32_t)p[12] | ((uint32_t)p[13] << 8) | ((uint32_t)p[14] << 16) | ((uint32_t)p[15] << 24);
        w0 ^= crc;
        crc = CRC32_TABLE[15][ w0        & 0xFF] ^
              CRC32_TABLE[14][(w0 >> 8)  & 0xFF] ^
              CRC32_TABLE[13][(w0 >> 16) & 0xFF] ^
              CRC32_TABLE[12][(w0 >> 24) & 0xFF] ^
              CRC32_TABLE[11][ w1        & 0xFF] ^
              CRC32_TABLE[10][(w1 >> 8)  & 0xFF] ^
              CRC32_TABLE[ 9][(w1 >> 16) & 0xFF] ^
              CRC32_TABLE[ 8][(w1 >> 24) & 0xFF] ^
              CRC32_TABLE[ 7][ w2        & 0xFF] ^
              CRC32_TABLE[ 6][(w2 >> 8)  & 0xFF] ^
              CRC32_TABLE[ 5][(w2 >> 16) & 0xFF] ^
              CRC32_TABLE[ 4][(w2 >> 24) & 0xFF] ^
              CRC32_TABLE[ 3][ w3        & 0xFF] ^
              CRC32_TABLE[ 2][(w3 >> 8)  & 0xFF] ^
              CRC32_TABLE[ 1][(w3 >> 16) & 0xFF] ^
              CRC32_TABLE[ 0][(w3 >> 24) & 0xFF];
        p += 16;
        len -= 16;
    }
    while (len-- > 0) {
        crc = CRC32_TABLE[0][(crc ^ *p++) & 0xFF] ^ (crc >> 8);
    }
    return crc ^ 0xFFFFFFFF;
}

// =============================================================================
// Scalar yEnc Decoder (fallback)
// =============================================================================
static Py_ssize_t decode_yenc_scalar(
    const uint8_t* src,
    Py_ssize_t src_len,
    uint8_t* output
) {
    Py_ssize_t out_len = 0;

    for (Py_ssize_t i = 0; i < src_len; i++) {
        uint8_t byte = src[i];

        if (byte == '=') {
            // Escape sequence
            i++;
            if (i < src_len) {
                output[out_len++] = ESCAPE_TABLE[src[i]];
            }
        } else if (byte != '\r' && byte != '\n') {
            // Regular data byte
            output[out_len++] = DECODE_TABLE[byte];
        }
        // Skip CR/LF
    }

    return out_len;
}

// Index of the first set bit. Caller guarantees x != 0.
#ifdef _MSC_VER
static inline unsigned int ctz32(unsigned int x) {
    unsigned long idx;
    _BitScanForward(&idx, x);
    return (unsigned int)idx;
}
#else
static inline unsigned int ctz32(unsigned int x) {
    return (unsigned int)__builtin_ctz(x);
}
#endif

// =============================================================================
// AVX2 SIMD yEnc Decoder - 32 bytes per iteration
// =============================================================================
static Py_ssize_t decode_yenc_avx2(
    const uint8_t* src,
    Py_ssize_t src_len,
    uint8_t* output
) {
    Py_ssize_t out_len = 0;
    Py_ssize_t i = 0;

    // AVX2 constants
    const __m256i v_42 = _mm256_set1_epi8(42);
    const __m256i v_cr = _mm256_set1_epi8('\r');
    const __m256i v_lf = _mm256_set1_epi8('\n');
    const __m256i v_eq = _mm256_set1_epi8('=');

    // Process 32 bytes at a time. On a window containing special bytes, bulk-
    // decode the clean prefix up to the FIRST special byte with one SIMD op,
    // then handle just that byte — instead of crawling byte-by-byte through the
    // clean data that precedes every line ending / escape.
    while (i + 32 <= src_len) {
        __m256i data = _mm256_loadu_si256((const __m256i*)(src + i));

        // Check for special characters
        __m256i is_cr = _mm256_cmpeq_epi8(data, v_cr);
        __m256i is_lf = _mm256_cmpeq_epi8(data, v_lf);
        __m256i is_eq = _mm256_cmpeq_epi8(data, v_eq);
        __m256i is_special = _mm256_or_si256(_mm256_or_si256(is_cr, is_lf), is_eq);
        unsigned int mask = (unsigned int)_mm256_movemask_epi8(is_special);

        if (mask == 0) {
            // FAST PATH: no special bytes - decode the whole 32-byte block.
            __m256i decoded = _mm256_sub_epi8(data, v_42);
            _mm256_storeu_si256((__m256i*)(output + out_len), decoded);
            out_len += 32;
            i += 32;
        } else {
            // Bulk-decode the clean prefix [0, first) with one SIMD op. The
            // 32-byte store is in-bounds (out_len + 32 <= i + 32 <= src_len);
            // bytes past out_len+first are overwritten by later stores.
            unsigned int first = ctz32(mask);
            if (first > 0) {
                __m256i decoded = _mm256_sub_epi8(data, v_42);
                _mm256_storeu_si256((__m256i*)(output + out_len), decoded);
                out_len += first;
                i += first;
            }
            // i now points exactly at a special byte (= , \r or \n).
            uint8_t byte = src[i];
            if (byte == '=') {
                i++;
                if (i < src_len) {
                    output[out_len++] = ESCAPE_TABLE[src[i]];
                    i++;
                }
            } else {
                // '\r' or '\n' - skip
                i++;
            }
        }
    }

    // Handle remaining bytes with scalar
    while (i < src_len) {
        uint8_t byte = src[i];

        if (byte == '=') {
            i++;
            if (i < src_len) {
                output[out_len++] = ESCAPE_TABLE[src[i]];
                i++;
            }
        } else if (byte != '\r' && byte != '\n') {
            output[out_len++] = DECODE_TABLE[byte];
            i++;
        } else {
            i++;
        }
    }

    return out_len;
}

// =============================================================================
// Python Interface: decode()
// =============================================================================
static PyObject* py_decode(PyObject* self, PyObject* args) {
    Py_buffer input_buffer;

    if (!PyArg_ParseTuple(args, "y*", &input_buffer)) {
        return NULL;
    }

    const uint8_t* src = (const uint8_t*)input_buffer.buf;
    Py_ssize_t src_len = input_buffer.len;

    // Allocate the result bytes up front (decoded <= encoded) and decode IN PLACE
    // into its buffer — no PyMem_Malloc, no extra copy. Resized down at the end.
    PyObject* result = PyBytes_FromStringAndSize(NULL, src_len);
    if (!result) {
        PyBuffer_Release(&input_buffer);
        return NULL;
    }
    uint8_t* output = (uint8_t*)PyBytes_AS_STRING(result);

    Py_ssize_t out_len = 0;

    // Release GIL for CPU-intensive work
    Py_BEGIN_ALLOW_THREADS

    // Auto-dispatch: AVX2 for large buffers, scalar for small
    if (has_avx2 && src_len >= 64) {
        out_len = decode_yenc_avx2(src, src_len, output);
    } else {
        out_len = decode_yenc_scalar(src, src_len, output);
    }

    Py_END_ALLOW_THREADS

    PyBuffer_Release(&input_buffer);

    if (_PyBytes_Resize(&result, out_len) != 0) {
        return NULL;  // result already cleared by _PyBytes_Resize on failure
    }
    return result;
}

// =============================================================================
// Python Interface: decode_with_crc()
// =============================================================================
static PyObject* py_decode_with_crc(PyObject* self, PyObject* args) {
    Py_buffer input_buffer;

    if (!PyArg_ParseTuple(args, "y*", &input_buffer)) {
        return NULL;
    }

    const uint8_t* src = (const uint8_t*)input_buffer.buf;
    Py_ssize_t src_len = input_buffer.len;

    // In-place decode into the result bytes (no malloc / no extra copy).
    PyObject* bytes_result = PyBytes_FromStringAndSize(NULL, src_len);
    if (!bytes_result) {
        PyBuffer_Release(&input_buffer);
        return NULL;
    }
    uint8_t* output = (uint8_t*)PyBytes_AS_STRING(bytes_result);

    Py_ssize_t out_len = 0;
    uint32_t crc = 0;

    Py_BEGIN_ALLOW_THREADS

    // Decode
    if (has_avx2 && src_len >= 64) {
        out_len = decode_yenc_avx2(src, src_len, output);
    } else {
        out_len = decode_yenc_scalar(src, src_len, output);
    }

    // Fast slice-by-16 CRC32 over the decoded data (== zlib.crc32)
    crc = crc32_compute(output, out_len);

    Py_END_ALLOW_THREADS

    PyBuffer_Release(&input_buffer);

    if (_PyBytes_Resize(&bytes_result, out_len) != 0) {
        return NULL;
    }

    PyObject* crc_obj = PyLong_FromUnsignedLong(crc);
    if (!crc_obj) {
        Py_DECREF(bytes_result);
        return NULL;
    }
    PyObject* result = PyTuple_Pack(2, bytes_result, crc_obj);
    Py_DECREF(bytes_result);
    Py_DECREF(crc_obj);

    return result;
}

// =============================================================================
// Python Interface: unstuff_body()
// Copy buf[start:end] into a new bytes with NNTP dot-unstuffing applied
// (a line beginning with '.' was sent doubled). Runs with the GIL RELEASED so
// the per-body copy of every connection runs in parallel instead of serialising
// on the GIL. Equivalent to Python:
//     b = bytes(buf[start:end]); b = b.replace(b"\r\n..", b"\r\n.")
//     if b[:2] == b"..": b = b[1:]
// =============================================================================
static PyObject* py_unstuff_body(PyObject* self, PyObject* args) {
    Py_buffer in;
    Py_ssize_t start, end;
    if (!PyArg_ParseTuple(args, "y*nn", &in, &start, &end)) {
        return NULL;
    }
    if (start < 0 || end < start || end > in.len) {
        PyBuffer_Release(&in);
        PyErr_SetString(PyExc_ValueError, "unstuff_body: invalid start/end");
        return NULL;
    }

    const uint8_t* src = (const uint8_t*)in.buf + start;
    Py_ssize_t n = end - start;

    PyObject* result = PyBytes_FromStringAndSize(NULL, n);  // upper bound
    if (!result) {
        PyBuffer_Release(&in);
        return NULL;
    }
    uint8_t* out = (uint8_t*)PyBytes_AS_STRING(result);
    Py_ssize_t out_len = 0;

    Py_BEGIN_ALLOW_THREADS
    int at_line_start = 1;  // first body line is a line start
    for (Py_ssize_t i = 0; i < n; i++) {
        uint8_t c = src[i];
        if (at_line_start && c == '.' && i + 1 < n && src[i + 1] == '.') {
            // Stuffed dot: drop ONE '.', the next '.' is emitted as data.
            at_line_start = 0;
            continue;
        }
        out[out_len++] = c;
        at_line_start = (c == '\n');
    }
    Py_END_ALLOW_THREADS

    PyBuffer_Release(&in);

    if (_PyBytes_Resize(&result, out_len) != 0) {
        return NULL;
    }
    return result;
}

// =============================================================================
// Python Interface: decode_batch()
// =============================================================================
static PyObject* py_decode_batch(PyObject* self, PyObject* args) {
    PyObject* segments_list;

    if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &segments_list)) {
        return NULL;
    }

    Py_ssize_t count = PyList_Size(segments_list);
    PyObject* results = PyList_New(count);
    if (!results) return NULL;

    for (Py_ssize_t idx = 0; idx < count; idx++) {
        PyObject* item = PyList_GetItem(segments_list, idx);

        if (!PyBytes_Check(item)) {
            Py_INCREF(Py_None);
            PyList_SetItem(results, idx, Py_None);
            continue;
        }

        const uint8_t* src = (const uint8_t*)PyBytes_AsString(item);
        Py_ssize_t src_len = PyBytes_Size(item);

        // In-place decode into the per-item result bytes (no malloc / no copy).
        PyObject* decoded = PyBytes_FromStringAndSize(NULL, src_len);
        if (!decoded) {
            Py_DECREF(results);
            return NULL;
        }
        uint8_t* output = (uint8_t*)PyBytes_AS_STRING(decoded);

        Py_ssize_t out_len = 0;

        Py_BEGIN_ALLOW_THREADS

        if (has_avx2 && src_len >= 64) {
            out_len = decode_yenc_avx2(src, src_len, output);
        } else {
            out_len = decode_yenc_scalar(src, src_len, output);
        }

        Py_END_ALLOW_THREADS

        if (_PyBytes_Resize(&decoded, out_len) != 0) {
            Py_DECREF(results);
            return NULL;
        }

        PyList_SetItem(results, idx, decoded);
    }

    return results;
}

// =============================================================================
// Python Interface: has_avx2()
// =============================================================================
static PyObject* py_has_avx2(PyObject* self, PyObject* args) {
    return PyBool_FromLong(has_avx2 ? 1 : 0);
}

// =============================================================================
// Module Definition
// =============================================================================
static PyMethodDef YencTurboMethods[] = {
    {"decode", py_decode, METH_VARARGS,
     "Decode yEnc data with AVX2 SIMD acceleration. Returns decoded bytes."},
    {"decode_with_crc", py_decode_with_crc, METH_VARARGS,
     "Decode yEnc data and compute CRC32. Returns (bytes, crc32)."},
    {"unstuff_body", py_unstuff_body, METH_VARARGS,
     "unstuff_body(buf, start, end) -> bytes: copy buf[start:end] with NNTP "
     "dot-unstuffing, GIL released."},
    {"decode_batch", py_decode_batch, METH_VARARGS,
     "Decode multiple yEnc segments with AVX2 SIMD. Returns list of decoded bytes."},
    {"has_avx2", py_has_avx2, METH_NOARGS,
     "Check if AVX2 SIMD is available on this CPU."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef yenc_turbo_module = {
    PyModuleDef_HEAD_INIT,
    "yenc_turbo",
    "Ultra-fast yEnc decoder with AVX2 SIMD and GIL release for true parallelism.\n\n"
    "Functions:\n"
    "  decode(data) -> bytes: Decode yEnc data\n"
    "  decode_with_crc(data) -> (bytes, crc32): Decode with CRC verification\n"
    "  decode_batch(list) -> list: Batch decode multiple segments\n"
    "  has_avx2() -> bool: Check for AVX2 support\n",
    -1,
    YencTurboMethods
};

PyMODINIT_FUNC PyInit_yenc_turbo(void) {
    init_tables();
    return PyModule_Create(&yenc_turbo_module);
}
