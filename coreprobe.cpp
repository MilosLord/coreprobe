/*
 * coreprobe - Per-Core FPU/SIMD Correctness Diagnostic
 * https://github.com/MilosLord/coreprobe
 *
 * Detects faulty floating-point execution units by stress-testing quaternion
 * normalization across SCALAR, SSE3, AVX2, FMA3, and cross-lane AVX2 (XLANE)
 * on each logical processor independently. Catches silicon defects that
 * memtest86+, Prime95, and WHEA miss.
 *
 * This is an arithmetic CORRECTNESS test, not a throughput stress test.
 * -O0 and volatile barriers ensure every FP op executes through hardware.
 *
 * License: MIT
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#define COREPROBE_VERSION "1.0.2"
#define MAX_THREADS       4096

#include <bit>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <csignal>
#include <cstring>
#include <vector>

static volatile sig_atomic_t g_stop = 0;
static void signal_handler(int) { g_stop = 1; }

#if !defined(__x86_64__) && !defined(_M_X64) && !defined(__i386__) && !defined(_M_IX86)
#error "coreprobe requires an x86 or x86-64 target (SSE/AVX/FMA intrinsics)."
#endif
#include <immintrin.h>

// Optimization guards - prevent optimization even under -O2/-O3/PGO

#ifdef _MSC_VER
#define COREPROBE_NO_OPTIMIZE __pragma(optimize("", off))
#define COREPROBE_RESTORE_OPT __pragma(optimize("", on))
#elif defined(__clang__)
#define COREPROBE_NO_OPTIMIZE _Pragma("clang optimize off")
#define COREPROBE_RESTORE_OPT _Pragma("clang optimize on")
#elif defined(__GNUC__)
#define COREPROBE_NO_OPTIMIZE _Pragma("GCC push_options") _Pragma("GCC optimize(\"O0\")")
#define COREPROBE_RESTORE_OPT _Pragma("GCC pop_options")
#else
#define COREPROBE_NO_OPTIMIZE
#define COREPROBE_RESTORE_OPT
#endif

// Per-function ISA target attributes (GCC/Clang; MSVC uses global /arch:)
#if defined(__GNUC__) || defined(__clang__)
#define TARGET_SSE3  __attribute__((target("sse3")))
#define TARGET_AVX2  __attribute__((target("avx2")))
#define TARGET_FMA   __attribute__((target("avx,fma")))
#define TARGET_XSAVE __attribute__((target("xsave")))
#else
#define TARGET_SSE3
#define TARGET_AVX2
#define TARGET_FMA
#define TARGET_XSAVE
#endif

// ============================================================================
// Platform abstraction
// ============================================================================

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <intrin.h>
#include <Windows.h>

static HANDLE hConsole;

static BOOL WINAPI console_ctrl_handler(DWORD) { g_stop = 1; return TRUE; }

static void platform_init()
{
    hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hConsole == INVALID_HANDLE_VALUE || hConsole == nullptr) return;
    DWORD mode = 0;
    if (GetConsoleMode(hConsole, &mode)) { SetConsoleMode(hConsole, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING); }
    SetConsoleCtrlHandler(console_ctrl_handler, TRUE);
    signal(SIGINT, signal_handler);
}

static int platform_num_threads()
{
    int total = static_cast<int>(GetActiveProcessorCount(ALL_PROCESSOR_GROUPS));
    if (total > 0) return total;
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    int n = static_cast<int>(si.dwNumberOfProcessors);
    return n > 0 ? n : 0;
}

static bool platform_set_affinity(int thread_id)
{
    GROUP_AFFINITY ga = {};
    ga.Group          = static_cast<WORD>(thread_id / 64);
    ga.Mask           = static_cast<KAFFINITY>(1) << (thread_id % 64);
    if (!SetThreadGroupAffinity(GetCurrentThread(), &ga, nullptr)) return false;
    Sleep(0);
    PROCESSOR_NUMBER pn = {};
    GetCurrentProcessorNumberEx(&pn);
    int actual = static_cast<int>(pn.Group) * 64 + static_cast<int>(pn.Number);
    if (actual != thread_id) return false;
    return true;
}

static bool platform_set_high_priority() { return SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS) != 0; }

static double now_sec()
{
    static LARGE_INTEGER freq = {};
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return static_cast<double>(t.QuadPart) / static_cast<double>(freq.QuadPart);
}

static void cpuid(uint32_t leaf, uint32_t subleaf, uint32_t out[4])
{
    int regs[4];
    __cpuidex(regs, static_cast<int>(leaf), static_cast<int>(subleaf));
    out[0] = static_cast<uint32_t>(regs[0]);
    out[1] = static_cast<uint32_t>(regs[1]);
    out[2] = static_cast<uint32_t>(regs[2]);
    out[3] = static_cast<uint32_t>(regs[3]);
}

#else // Linux / POSIX
#include <cpuid.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <unistd.h>

static void platform_init()
{
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
}

static int platform_num_threads()
{
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? static_cast<int>(n) : 0;
}

static bool platform_set_affinity(int thread_id)
{
    int num_cpus = static_cast<int>(sysconf(_SC_NPROCESSORS_CONF));
    if (num_cpus < thread_id + 1) num_cpus = thread_id + 1;
    size_t     size   = CPU_ALLOC_SIZE(num_cpus);
    cpu_set_t* cpuset = CPU_ALLOC(num_cpus);
    if (!cpuset) return false;
    CPU_ZERO_S(size, cpuset);
    CPU_SET_S(thread_id, size, cpuset);
    int ret = pthread_setaffinity_np(pthread_self(), size, cpuset);
    CPU_FREE(cpuset);
    if (ret != 0) return false;
    sched_yield();
    return true;
}

static bool platform_set_high_priority()
{
    errno = 0;
    nice(-20);
    return errno == 0;
}

static double now_sec()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) / 1e9;
}

static void cpuid(uint32_t leaf, uint32_t subleaf, uint32_t out[4])
{
    __cpuid_count(leaf, subleaf, out[0], out[1], out[2], out[3]);
}
#endif

// ============================================================================
// CPU topology detection
// ============================================================================

struct TopologyInfo
{
    std::vector<int> physical_core;
    std::vector<int> package_id;
    int              core_count;
    bool             valid;
};

static TopologyInfo detect_topology(int max_threads)
{
    TopologyInfo topo;
    int          n = max_threads < MAX_THREADS ? max_threads : MAX_THREADS;

    topo.physical_core.resize(static_cast<size_t>(n));
    topo.package_id.resize(static_cast<size_t>(n), 0);
    for (int i = 0; i < n; i++)
        topo.physical_core[static_cast<size_t>(i)] = i;
    topo.core_count = n;
    topo.valid      = false;

#ifdef _WIN32
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (GetLastError() == ERROR_INSUFFICIENT_BUFFER && len > 0)
    {
        std::vector<uint8_t> buf(len);
        if (GetLogicalProcessorInformationEx(
                RelationProcessorCore,
                reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data()),
                &len))
        {
            int   core_idx = 0;
            DWORD offset   = 0;
            while (offset < len)
            {
                auto* info =
                    reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data() + offset);
                if (info->Size == 0 || offset + info->Size > len) break;
                if (info->Relationship == RelationProcessorCore)
                {
                    for (WORD g = 0; g < info->Processor.GroupCount; g++)
                    {
                        WORD      grp  = info->Processor.GroupMask[g].Group;
                        KAFFINITY mask = info->Processor.GroupMask[g].Mask;
                        for (int bit = 0; bit < 64; bit++)
                        {
                            if (mask & (static_cast<KAFFINITY>(1) << bit))
                            {
                                int tid = static_cast<int>(grp) * 64 + bit;
                                if (tid >= 0 && tid < static_cast<int>(topo.physical_core.size()))
                                    topo.physical_core[static_cast<size_t>(tid)] = core_idx;
                            }
                        }
                    }
                    core_idx++;
                }
                offset += info->Size;
            }
            topo.core_count = core_idx;
            topo.valid      = true;
        }
    }

    if (topo.valid)
    {
        DWORD pkg_len = 0;
        GetLogicalProcessorInformationEx(RelationProcessorPackage, nullptr, &pkg_len);
        if (GetLastError() == ERROR_INSUFFICIENT_BUFFER && pkg_len > 0)
        {
            std::vector<uint8_t> pbuf(pkg_len);
            if (GetLogicalProcessorInformationEx(
                    RelationProcessorPackage,
                    reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(pbuf.data()),
                    &pkg_len))
            {
                int   pkg_idx = 0;
                DWORD poff    = 0;
                while (poff < pkg_len)
                {
                    auto* pi =
                        reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(pbuf.data() + poff);
                    if (pi->Size == 0 || poff + pi->Size > pkg_len) break;
                    if (pi->Relationship == RelationProcessorPackage)
                    {
                        for (WORD g = 0; g < pi->Processor.GroupCount; g++)
                        {
                            WORD      grp  = pi->Processor.GroupMask[g].Group;
                            KAFFINITY mask = pi->Processor.GroupMask[g].Mask;
                            for (int bit = 0; bit < 64; bit++)
                            {
                                if (mask & (static_cast<KAFFINITY>(1) << bit))
                                {
                                    int tid = static_cast<int>(grp) * 64 + bit;
                                    if (tid >= 0 && tid < static_cast<int>(topo.package_id.size()))
                                        topo.package_id[static_cast<size_t>(tid)] = pkg_idx;
                                }
                            }
                        }
                        pkg_idx++;
                    }
                    poff += pi->Size;
                }
            }
        }
    }
#else
    struct PkgCore
    {
        int pkg;
        int core;
    };
    std::vector<PkgCore> raw(n);
    bool                 ok = true;
    for (int i = 0; i < n && ok; i++)
    {
        char path[256];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/core_id", i);
        FILE* f = fopen(path, "r");
        if (f)
        {
            if (fscanf(f, "%d", &raw[i].core) != 1) ok = false;
            fclose(f);
        }
        else { ok = false; }

        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", i);
        f = fopen(path, "r");
        if (f)
        {
            if (fscanf(f, "%d", &raw[i].pkg) != 1) ok = false;
            fclose(f);
        }
        else { ok = false; }
        topo.package_id[i] = raw[i].pkg;
    }
    if (ok)
    {
        std::vector<PkgCore> unique(n);
        int                  num_unique = 0;
        for (int i = 0; i < n; i++)
        {
            int idx = -1;
            for (int u = 0; u < num_unique; u++)
            {
                if (unique[u].pkg == raw[i].pkg && unique[u].core == raw[i].core)
                {
                    idx = u;
                    break;
                }
            }
            if (idx < 0)
            {
                idx                  = num_unique;
                unique[num_unique++] = raw[i];
            }
            topo.physical_core[i] = idx;
        }
        topo.core_count = num_unique;
        topo.valid      = true;
    }
#endif

    return topo;
}

// ============================================================================
// ANSI colors
// ============================================================================

#define COL_RESET   "\033[0m"
#define COL_RED     "\033[1;31m"
#define COL_GREEN   "\033[1;32m"
#define COL_YELLOW  "\033[1;33m"
#define COL_CYAN    "\033[1;36m"
#define COL_MAGENTA "\033[1;35m"
#define COL_GRAY    "\033[0;37m"

// ============================================================================
// CPUID detection
// ============================================================================

struct CPUFeatures
{
    bool has_sse;
    bool has_sse3;
    bool has_avx2;
    bool has_fma3;
    bool os_avx_enabled;
    char brand[49];
    char vendor[13];
};

// Check if OS enabled AVX state via XGETBV (required for AVX/FMA instructions)
TARGET_XSAVE
static uint64_t xgetbv(uint32_t xcr)
{
#ifdef _WIN32
    return static_cast<uint64_t>(_xgetbv(xcr));
#else
    uint32_t lo, hi;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(xcr));
    return (static_cast<uint64_t>(hi) << 32) | lo;
#endif
}

static CPUFeatures detect_cpu()
{
    CPUFeatures f = {};
    uint32_t    r[4];

    cpuid(0, 0, r);
    uint32_t max_basic = r[0];
    memcpy(f.vendor + 0, &r[1], 4);
    memcpy(f.vendor + 4, &r[3], 4);
    memcpy(f.vendor + 8, &r[2], 4);
    f.vendor[12] = 0;

    cpuid(0x80000000, 0, r);
    uint32_t max_ext = r[0];
    if (max_ext >= 0x80000004)
    {
        cpuid(0x80000002, 0, r);
        memcpy(f.brand + 0, r, 16);
        cpuid(0x80000003, 0, r);
        memcpy(f.brand + 16, r, 16);
        cpuid(0x80000004, 0, r);
        memcpy(f.brand + 32, r, 16);
        f.brand[48] = 0;
        char* p     = f.brand;
        while (*p == ' ')
            p++;
        if (p != f.brand) memmove(f.brand, p, strlen(p) + 1);
    }

    bool has_osxsave = false;
    bool cpu_fma3 = false, cpu_avx = false;
    if (max_basic >= 1)
    {
        cpuid(1, 0, r);
        f.has_sse   = (r[3] & (1 << 25)) != 0;
        f.has_sse3  = (r[2] & (1 << 0)) != 0;
        has_osxsave = (r[2] & (1 << 27)) != 0;
        cpu_fma3    = (r[2] & (1 << 12)) != 0;
        cpu_avx     = (r[2] & (1 << 28)) != 0;
    }

    bool cpu_avx2 = false;
    if (max_basic >= 7)
    {
        cpuid(7, 0, r);
        cpu_avx2 = (r[1] & (1 << 5)) != 0;
    }

    f.os_avx_enabled = false;
    if (has_osxsave && cpu_avx)
    {
        uint64_t xcr0    = xgetbv(0);
        f.os_avx_enabled = ((xcr0 & 0x6) == 0x6);
    }

    f.has_avx2 = cpu_avx2 && f.os_avx_enabled;
    f.has_fma3 = cpu_fma3 && f.os_avx_enabled;

    return f;
}

// ============================================================================
// PRNG -xoshiro128**
// ============================================================================

struct PRNG
{
    uint32_t s[4];

    void seed(uint32_t v)
    {
        for (int i = 0; i < 4; i++)
        {
            v          += 0x9E3779B9u;
            uint32_t z  = v;
            z           = (z ^ (z >> 16)) * 0x85EBCA6Bu;
            z           = (z ^ (z >> 13)) * 0xC2B2AE35u;
            s[i]        = z ^ (z >> 16);
        }
    }

    uint32_t next()
    {
        uint32_t r  = ((s[1] * 5) << 7 | (s[1] * 5) >> 25) * 9;
        uint32_t t  = s[1] << 9;
        s[2]       ^= s[0];
        s[3]       ^= s[1];
        s[1]       ^= s[2];
        s[0]       ^= s[3];
        s[2]       ^= t;
        s[3]        = (s[3] << 11) | (s[3] >> 21);
        return r;
    }

    float randf() { return static_cast<float>(static_cast<int32_t>(next())) / static_cast<float>(INT32_MAX); }
};

// ============================================================================
// Test infrastructure
// ============================================================================

static const double   TOLERANCE       = 0.001;
static const double   WARN_THRESHOLD  = 0.0001;
static const int      RERUN_COUNT     = 3;
static const double   RERUN_DURATION  = 1.0;
static const uint64_t ITER_CHECK_FREQ = 0xFFFFF;

enum TestType
{
    T_SCALAR = 0,
    T_SSE3,
    T_AVX2,
    T_FMA3,
    T_XLANE,
    NUM_TESTS
};
static const char* tname[NUM_TESTS] = {"SCALAR", "SSE3", "AVX2", "FMA3", "XLANE"};

struct TestResult
{
    bool     passed;
    bool     skipped;
    bool     confirmed;
    int      rerun_fails;
    uint64_t iterations;
    double   worst_dev;
    float    worst_q[4];
    uint64_t worst_iter;
    uint32_t fail_seed;
};

// ============================================================================
// Test functions (optimization-protected)
// ============================================================================
COREPROBE_NO_OPTIMIZE

// SCALAR -single-operation FP pipeline

static TestResult run_scalar(PRNG& rng, double deadline)
{
    TestResult r = {};
    r.passed     = true;
    for (uint64_t i = 1;; i++)
    {
        volatile float qx = rng.randf(), qy = rng.randf();
        volatile float qz = rng.randf(), qw = rng.randf();
        volatile float x = qx, y = qy, z = qz, w = qw;
        volatile float ls = x * x + y * y + z * z + w * w;
        if (ls < 1e-10f) continue;
        volatile float il = 1.0f / sqrtf(static_cast<float>(ls));
        volatile float nx = x * il, ny = y * il, nz = z * il, nw = w * il;
        volatile float ck  = nx * nx + ny * ny + nz * nz + nw * nw;
        double         dev = fabs(static_cast<double>(static_cast<float>(ck)) - 1.0);
        if (std::isfinite(dev) && dev > r.worst_dev) r.worst_dev = dev;
        if (!std::isfinite(dev) || dev > TOLERANCE)
        {
            r.passed     = false;
            r.worst_dev  = dev;
            r.worst_iter = i;
            r.worst_q[0] = nx;
            r.worst_q[1] = ny;
            r.worst_q[2] = nz;
            r.worst_q[3] = nw;
            r.iterations = i;
            return r;
        }
        r.iterations = i;
        if ((i & ITER_CHECK_FREQ) == 0 && (g_stop || now_sec() >= deadline)) break;
    }
    return r;
}

// SSE3 helper - dot product for verification
TARGET_SSE3
static float sse3_dot4(__m128 v)
{
    __m128 sq2 = _mm_mul_ps(v, v);
    __m128 h1 = _mm_hadd_ps(sq2, sq2), h2 = _mm_hadd_ps(h1, h1);
    float  c_raw;
    _mm_store_ss(&c_raw, h2);
    volatile float c = c_raw;
    return c;
}

// SSE3 -128-bit SIMD

TARGET_SSE3
static TestResult run_sse(PRNG& rng, double deadline)
{
    TestResult r = {};
    r.passed     = true;
    for (uint64_t i = 1;; i++)
    {
        volatile float qx = rng.randf(), qy = rng.randf(), qz = rng.randf(), qw = rng.randf();
        __m128 q = _mm_set_ps(
            static_cast<float>(qw), static_cast<float>(qz), static_cast<float>(qy), static_cast<float>(qx));
        __m128 sq = _mm_mul_ps(q, q);
        __m128 s1 = _mm_hadd_ps(sq, sq), s2 = _mm_hadd_ps(s1, s1);
        float  ls_raw;
        _mm_store_ss(&ls_raw, s2);
        volatile float ls = ls_raw;
        if (static_cast<float>(ls) < 1e-10f) continue;

        __m128 lv  = _mm_shuffle_ps(s2, s2, 0);
        __m128 inv = _mm_rsqrt_ps(lv);
        __m128 nr  = _mm_mul_ps(_mm_mul_ps(_mm_set1_ps(0.5f), lv), _mm_mul_ps(inv, inv));
        inv        = _mm_mul_ps(inv, _mm_sub_ps(_mm_set1_ps(1.5f), nr));
        __m128 na  = _mm_mul_ps(q, inv);
        __m128 nb  = _mm_div_ps(q, _mm_sqrt_ps(lv));

        double da = fabs(static_cast<double>(sse3_dot4(na)) - 1.0);
        double db = fabs(static_cast<double>(sse3_dot4(nb)) - 1.0);
        double worst = da > db ? da : db;
        if (std::isfinite(worst) && worst > r.worst_dev) r.worst_dev = worst;
        if (!std::isfinite(da) || !std::isfinite(db) || worst > TOLERANCE)
        {
            r.passed     = false;
            r.worst_dev  = worst;
            r.worst_iter = i;
            float tmp[4];
            _mm_storeu_ps(tmp, da > db ? na : nb);
            r.worst_q[0] = tmp[0];
            r.worst_q[1] = tmp[1];
            r.worst_q[2] = tmp[2];
            r.worst_q[3] = tmp[3];
            r.iterations = i;
            return r;
        }
        r.iterations = i;
        if ((i & ITER_CHECK_FREQ) == 0 && (g_stop || now_sec() >= deadline)) break;
    }
    return r;
}

// AVX2 -256-bit SIMD (two quaternions simultaneously)

TARGET_AVX2
static TestResult run_avx2(PRNG& rng, double deadline)
{
    TestResult r = {};
    r.passed     = true;
    for (uint64_t i = 1;; i++)
    {
        volatile float a = rng.randf(), b = rng.randf(), c = rng.randf(), d = rng.randf();
        volatile float e = rng.randf(), f = rng.randf(), g = rng.randf(), h = rng.randf();
        __m256 q = _mm256_set_ps(
            static_cast<float>(h), static_cast<float>(g), static_cast<float>(f), static_cast<float>(e),
            static_cast<float>(d), static_cast<float>(c), static_cast<float>(b), static_cast<float>(a));
        __m256 sq = _mm256_mul_ps(q, q);
        __m256 h1 = _mm256_hadd_ps(sq, sq), h2 = _mm256_hadd_ps(h1, h1);
        float  l1_raw, l2_raw;
        _mm_store_ss(&l1_raw, _mm256_castps256_ps128(h2));
        _mm_store_ss(&l2_raw, _mm256_extractf128_ps(h2, 1));
        volatile float l1 = l1_raw, l2 = l2_raw;
        if (static_cast<float>(l1) < 1e-10f || static_cast<float>(l2) < 1e-10f) continue;

        __m256 norm = _mm256_div_ps(q, _mm256_sqrt_ps(h2));
        __m256 ns   = _mm256_mul_ps(norm, norm);
        __m256 nh1 = _mm256_hadd_ps(ns, ns), nh2 = _mm256_hadd_ps(nh1, nh1);
        float  c1_raw, c2_raw;
        _mm_store_ss(&c1_raw, _mm256_castps256_ps128(nh2));
        _mm_store_ss(&c2_raw, _mm256_extractf128_ps(nh2, 1));
        volatile float c1 = c1_raw, c2 = c2_raw;
        double d1 = fabs(static_cast<double>(static_cast<float>(c1)) - 1.0);
        double d2 = fabs(static_cast<double>(static_cast<float>(c2)) - 1.0);
        {
            double w = d1 > d2 ? d1 : d2;
            if (std::isfinite(w) && w > r.worst_dev) r.worst_dev = w;
        }

        if (!std::isfinite(d1) || !std::isfinite(d2) || d1 > TOLERANCE || d2 > TOLERANCE)
        {
            r.passed     = false;
            r.worst_dev  = d1 > d2 ? d1 : d2;
            r.worst_iter = i;
            float tmp[8];
            _mm256_storeu_ps(tmp, norm);
            int w        = d1 > d2 ? 0 : 4;
            r.worst_q[0] = tmp[w];
            r.worst_q[1] = tmp[w + 1];
            r.worst_q[2] = tmp[w + 2];
            r.worst_q[3] = tmp[w + 3];
            r.iterations = i;
            _mm256_zeroupper();
            return r;
        }
        r.iterations = i;
        if ((i & ITER_CHECK_FREQ) == 0 && (g_stop || now_sec() >= deadline)) break;
    }
    _mm256_zeroupper();
    return r;
}

// FMA3 helper - dot product using fused multiply-add
TARGET_FMA
static float fma3_check(__m128 n)
{
    float v_raw[4];
    _mm_storeu_ps(v_raw, n);
    volatile float v0 = v_raw[0], v1 = v_raw[1], v2 = v_raw[2], v3 = v_raw[3];
    __m128 a2 = _mm_set1_ps(static_cast<float>(v0)), b2 = _mm_set1_ps(static_cast<float>(v1));
    __m128 c2 = _mm_set1_ps(static_cast<float>(v2)), d2 = _mm_set1_ps(static_cast<float>(v3));
    __m128 r2 = _mm_mul_ss(a2, a2);
    r2        = _mm_fmadd_ss(b2, b2, r2);
    r2        = _mm_fmadd_ss(c2, c2, r2);
    r2        = _mm_fmadd_ss(d2, d2, r2);
    float o_raw;
    _mm_store_ss(&o_raw, r2);
    volatile float o = o_raw;
    return o;
}

// FMA3 -fused multiply-add pipeline

TARGET_FMA
static TestResult run_fma3(PRNG& rng, double deadline)
{
    TestResult r = {};
    r.passed     = true;
    for (uint64_t i = 1;; i++)
    {
        volatile float qx = rng.randf(), qy = rng.randf(), qz = rng.randf(), qw = rng.randf();
        __m128 q = _mm_set_ps(
            static_cast<float>(qw), static_cast<float>(qz), static_cast<float>(qy), static_cast<float>(qx));
        __m128 xx  = _mm_set1_ps(static_cast<float>(qx)), yy = _mm_set1_ps(static_cast<float>(qy));
        __m128 zz  = _mm_set1_ps(static_cast<float>(qz)), ww = _mm_set1_ps(static_cast<float>(qw));
        __m128 dot = _mm_mul_ss(xx, xx);
        dot        = _mm_fmadd_ss(yy, yy, dot);
        dot        = _mm_fmadd_ss(zz, zz, dot);
        dot        = _mm_fmadd_ss(ww, ww, dot);
        float ls_raw;
        _mm_store_ss(&ls_raw, dot);
        volatile float ls = ls_raw;
        if (static_cast<float>(ls) < 1e-10f) continue;

        __m128 inv = _mm_rsqrt_ss(dot);
        __m128 nrf = _mm_fnmadd_ss(_mm_mul_ss(dot, _mm_set_ss(0.5f)), _mm_mul_ss(inv, inv), _mm_set_ss(1.5f));
        inv        = _mm_mul_ss(inv, nrf);
        __m128 na  = _mm_mul_ps(q, _mm_shuffle_ps(inv, inv, 0));
        __m128 len = _mm_sqrt_ss(dot);
        __m128 nb  = _mm_div_ps(q, _mm_shuffle_ps(len, len, 0));

        double da = fabs(static_cast<double>(fma3_check(na)) - 1.0);
        double db = fabs(static_cast<double>(fma3_check(nb)) - 1.0);
        double worst = da > db ? da : db;
        if (std::isfinite(worst) && worst > r.worst_dev) r.worst_dev = worst;
        if (!std::isfinite(da) || !std::isfinite(db) || worst > TOLERANCE)
        {
            r.passed     = false;
            r.worst_dev  = worst;
            r.worst_iter = i;
            float tmp[4];
            _mm_storeu_ps(tmp, da > db ? na : nb);
            r.worst_q[0] = tmp[0];
            r.worst_q[1] = tmp[1];
            r.worst_q[2] = tmp[2];
            r.worst_q[3] = tmp[3];
            r.iterations = i;
            _mm256_zeroupper();
            return r;
        }
        r.iterations = i;
        if ((i & ITER_CHECK_FREQ) == 0 && (g_stop || now_sec() >= deadline)) break;
    }
    _mm256_zeroupper();
    return r;
}

// XLANE - AVX2 cross-lane data integrity (bit-exact)

// Each iteration performs 3 independent permutation checks (swap, reverse, rotate),
// so iterations are counted as i*3 to reflect actual cross-lane verifications.
TARGET_AVX2
static TestResult run_xlane(PRNG& rng, double deadline)
{
    TestResult r = {};
    r.passed     = true;
    for (uint64_t i = 1;; i++)
    {
        volatile float f0 = rng.randf(), f1 = rng.randf(), f2 = rng.randf(), f3 = rng.randf();
        volatile float f4 = rng.randf(), f5 = rng.randf(), f6 = rng.randf(), f7 = rng.randf();

        __m256 src = _mm256_set_ps(
            static_cast<float>(f7), static_cast<float>(f6), static_cast<float>(f5), static_cast<float>(f4),
            static_cast<float>(f3), static_cast<float>(f2), static_cast<float>(f1), static_cast<float>(f0));

        __m256 swapped = _mm256_permute2f128_ps(src, src, 0x01);
        float  sw_raw[8];
        _mm256_storeu_ps(sw_raw, swapped);
        volatile float expected_sw[8] = {f4, f5, f6, f7, f0, f1, f2, f3};
        for (int j = 0; j < 8; j++)
        {
            volatile float sw_j = sw_raw[j];
            if (std::bit_cast<uint32_t>(static_cast<float>(sw_j)) !=
                std::bit_cast<uint32_t>(static_cast<float>(expected_sw[j])))
            {
                r.passed     = false;
                r.worst_dev  = 1.0;
                r.worst_iter = i;
                r.worst_q[0] = sw_j;
                r.worst_q[1] = expected_sw[j];
                r.worst_q[2] = static_cast<float>(j);
                r.worst_q[3] = 0.0f;
                r.iterations = i * 3;
                _mm256_zeroupper();
                return r;
            }
        }

        __m256i        idx_rev = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        __m256 rev = _mm256_permutevar8x32_ps(src, idx_rev);
        float  rv_raw[8];
        _mm256_storeu_ps(rv_raw, rev);

        volatile float expected_rv[8] = {f7, f6, f5, f4, f3, f2, f1, f0};
        for (int j = 0; j < 8; j++)
        {
            volatile float rv_j = rv_raw[j];
            if (std::bit_cast<uint32_t>(static_cast<float>(rv_j)) !=
                std::bit_cast<uint32_t>(static_cast<float>(expected_rv[j])))
            {
                r.passed     = false;
                r.worst_dev  = 1.0;
                r.worst_iter = i;
                r.worst_q[0] = rv_j;
                r.worst_q[1] = expected_rv[j];
                r.worst_q[2] = static_cast<float>(j);
                r.worst_q[3] = 1.0f;
                r.iterations = i * 3;
                _mm256_zeroupper();
                return r;
            }
        }

        __m256i        idx_rot = _mm256_set_epi32(2, 1, 0, 7, 6, 5, 4, 3);
        __m256 rot = _mm256_permutevar8x32_ps(src, idx_rot);
        float  rt_raw[8];
        _mm256_storeu_ps(rt_raw, rot);

        volatile float expected_rt[8] = {f3, f4, f5, f6, f7, f0, f1, f2};
        for (int j = 0; j < 8; j++)
        {
            volatile float rt_j = rt_raw[j];
            if (std::bit_cast<uint32_t>(static_cast<float>(rt_j)) !=
                std::bit_cast<uint32_t>(static_cast<float>(expected_rt[j])))
            {
                r.passed     = false;
                r.worst_dev  = 1.0;
                r.worst_iter = i;
                r.worst_q[0] = rt_j;
                r.worst_q[1] = expected_rt[j];
                r.worst_q[2] = static_cast<float>(j);
                r.worst_q[3] = 2.0f;
                r.iterations = i * 3;
                _mm256_zeroupper();
                return r;
            }
        }

        r.iterations = i * 3;
        if ((i & ITER_CHECK_FREQ) == 0 && (g_stop || now_sec() >= deadline)) break;
    }
    _mm256_zeroupper();
    return r;
}

// ============================================================================
// Test dispatch
// ============================================================================

typedef TestResult (*TestFunc)(PRNG&, double);
static TestFunc test_funcs[NUM_TESTS] = {run_scalar, run_sse, run_avx2, run_fma3, run_xlane};

// Deterministic rerun to confirm failures

static TestResult rerun_single(TestFunc fn, uint32_t seed, double duration)
{
    PRNG rng;
    rng.seed(seed);
    return fn(rng, now_sec() + duration);
}

COREPROBE_RESTORE_OPT

// ============================================================================
// Argument parsing
// ============================================================================

// Parse a base-10 integer; reject empty/malformed/overflow. Returns true on success.
static bool parse_int(const char* s, int* out)
{
    if (!s || !*s) return false;
    errno     = 0;
    char* end = nullptr;
    long  v   = strtol(s, &end, 10);
    if (errno != 0 || end == s || (end && *end != '\0')) return false;
    if (v < INT_MIN || v > INT_MAX) return false;
    *out = static_cast<int>(v);
    return true;
}

// Parse "lo-hi". The first char must not be '-' (that's a flag).
static bool parse_range(const char* s, int* lo, int* hi)
{
    const char* dash = strchr(s, '-');
    if (!dash || dash == s) return false;
    size_t lo_len = static_cast<size_t>(dash - s);
    char   buf[32];
    if (lo_len >= sizeof(buf)) return false;
    memcpy(buf, s, lo_len);
    buf[lo_len] = '\0';
    if (!parse_int(buf, lo)) return false;
    if (!parse_int(dash + 1, hi)) return false;
    return true;
}

static const char* flags_with_values[] = {"--socket", "--repeat", nullptr};

static bool is_flag_value(int idx, int argc, char** argv)
{
    if (idx <= 1) return false;
    (void) argc;
    const char* prev = argv[idx - 1];
    for (const char** f = flags_with_values; *f; f++)
    {
        if (strcmp(prev, *f) == 0) return true;
    }
    return false;
}

// Index of first positional (non-flag, non-flag-value) argument, or -1.
static int find_first_positional(int argc, char** argv)
{
    for (int a = 1; a < argc; a++)
    {
        if (argv[a][0] == '-') continue;
        if (is_flag_value(a, argc, argv)) continue;
        return a;
    }
    return -1;
}

static int parse_threads(int argc, char** argv, int* out, int max_threads, bool* parse_error)
{
    int                    count = 0;
    int                    cap   = max_threads < MAX_THREADS ? max_threads : MAX_THREADS;
    std::vector<bool>      seen(static_cast<size_t>(cap), false);
    if (cap < 0) cap = 0;
    if (parse_error) *parse_error = false;
    int  duration_idx   = find_first_positional(argc, argv);
    bool selector_given = false;

    for (int a = 1; a < argc && count < MAX_THREADS; a++)
    {
        if (a == duration_idx) continue;
        if (argv[a][0] == '-')
        {
            for (const char** f = flags_with_values; *f; f++)
            {
                if (strcmp(argv[a], *f) == 0) { if (a + 1 < argc) a++; break; }
            }
            continue;
        }
        if (is_flag_value(a, argc, argv)) continue;

        const char* arg = argv[a];
        selector_given  = true;
        if (strchr(arg, '-'))
        {
            int lo = 0, hi = 0;
            if (!parse_range(arg, &lo, &hi))
            {
                fprintf(stderr, COL_RED "  Error: invalid thread range: %s\n" COL_RESET, arg);
                if (parse_error) *parse_error = true;
                return 0;
            }
            if (lo > hi)
            {
                fprintf(stderr, COL_RED "  Error: inverted thread range: %s\n" COL_RESET, arg);
                if (parse_error) *parse_error = true;
                return 0;
            }
            if (lo < 0 || hi >= cap)
            {
                fprintf(stderr, COL_RED "  Error: thread range %s outside 0..%d\n" COL_RESET, arg, cap - 1);
                if (parse_error) *parse_error = true;
                return 0;
            }
            for (int t = lo; t <= hi && count < MAX_THREADS; t++)
            {
                if (!seen[static_cast<size_t>(t)])
                {
                    seen[static_cast<size_t>(t)]      = true;
                    out[count++] = t;
                }
            }
        }
        else
        {
            int t = 0;
            if (!parse_int(arg, &t))
            {
                fprintf(stderr, COL_RED "  Error: invalid thread id: %s\n" COL_RESET, arg);
                if (parse_error) *parse_error = true;
                return 0;
            }
            if (t < 0 || t >= cap)
            {
                fprintf(stderr, COL_RED "  Error: thread id %d outside 0..%d\n" COL_RESET, t, cap - 1);
                if (parse_error) *parse_error = true;
                return 0;
            }
            if (!seen[static_cast<size_t>(t)])
            {
                seen[static_cast<size_t>(t)]      = true;
                out[count++] = t;
            }
        }
    }
    if (!selector_given)
    {
        for (int i = 0; i < cap; i++)
            out[i] = i;
        return cap;
    }
    if (count == 0)
    {
        fprintf(stderr, COL_RED "  Error: no valid threads in selector\n" COL_RESET);
        if (parse_error) *parse_error = true;
        return 0;
    }
    return count;
}

// ============================================================================
// JSON output
// ============================================================================

struct CoreResult
{
    int        thread_id;
    bool       affinity_ok;
    TestResult tests[NUM_TESTS];
};

static void json_escape(const char* in, char* out, size_t out_size)
{
    size_t j = 0;
    for (size_t i = 0; in[i] != '\0' && j + 2 < out_size; i++)
    {
        auto c = static_cast<unsigned char>(in[i]);
        if (c == '"' || c == '\\')
        {
            if (j + 3 >= out_size) break;
            out[j++] = '\\';
            out[j++] = static_cast<char>(c);
        }
        else if (c < 0x20)
        {
            if (j + 7 >= out_size) break;
            int n = snprintf(out + j, out_size - j, "\\u%04x", c);
            if (n < 0) break;
            j += static_cast<size_t>(n);
        }
        else { out[j++] = static_cast<char>(c); }
    }
    out[j < out_size ? j : out_size - 1] = '\0';
}

static bool write_json(const char*         path,
                       const CPUFeatures&  cpu,
                       const TopologyInfo& topo,
                       const CoreResult*   all,
                       int                 num_results,
                       double              wall_time)
{
    char tmp_path[512];
    snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", path);
    FILE* fp = fopen(tmp_path, "w");
    if (!fp)
    {
        fprintf(stderr, "  Warning: could not write %s\n", path);
        return false;
    }

    char brand_esc[128], vendor_esc[32];
    json_escape(cpu.brand, brand_esc, sizeof(brand_esc));
    json_escape(cpu.vendor, vendor_esc, sizeof(vendor_esc));

    fprintf(fp, "{\n");
    fprintf(fp, "  \"version\": \"%s\",\n", COREPROBE_VERSION);
    fprintf(fp, "  \"cpu\": {\n");
    fprintf(fp, "    \"brand\": \"%s\",\n", brand_esc);
    fprintf(fp, "    \"vendor\": \"%s\",\n", vendor_esc);
    fprintf(fp, "    \"has_sse\": %s,\n", cpu.has_sse ? "true" : "false");
    fprintf(fp, "    \"has_sse3\": %s,\n", cpu.has_sse3 ? "true" : "false");
    fprintf(fp, "    \"has_avx2\": %s,\n", cpu.has_avx2 ? "true" : "false");
    fprintf(fp, "    \"has_fma3\": %s,\n", cpu.has_fma3 ? "true" : "false");
    fprintf(fp, "    \"os_avx_enabled\": %s\n", cpu.os_avx_enabled ? "true" : "false");
    fprintf(fp, "  },\n");
    fprintf(fp, "  \"topology_valid\": %s,\n", topo.valid ? "true" : "false");
    fprintf(fp, "  \"physical_cores\": %d,\n", topo.core_count);
    fprintf(fp, "  \"tolerance\": %.6f,\n", TOLERANCE);
    fprintf(fp, "  \"wall_time_sec\": %.2f,\n", wall_time);
    fprintf(fp, "  \"results\": [\n");

    for (int i = 0; i < num_results; i++)
    {
        const CoreResult* cr   = &all[i];
        int               phys = 0, pkg = 0;
        if (cr->thread_id >= 0 && cr->thread_id < static_cast<int>(topo.physical_core.size()))
        {
            phys = topo.physical_core[static_cast<size_t>(cr->thread_id)];
            pkg  = topo.package_id[static_cast<size_t>(cr->thread_id)];
        }
        fprintf(fp, "    {\n");
        fprintf(fp, "      \"thread\": %d,\n", cr->thread_id);
        fprintf(fp, "      \"physical_core\": %d,\n", phys);
        fprintf(fp, "      \"package\": %d,\n", pkg);
        fprintf(fp, "      \"affinity_ok\": %s,\n", cr->affinity_ok ? "true" : "false");
        fprintf(fp, "      \"tests\": {\n");
        for (int t = 0; t < NUM_TESTS; t++)
        {
            const TestResult* tr = &cr->tests[t];
            fprintf(fp, "        \"%s\": {\n", tname[t]);
            if (tr->skipped) { fprintf(fp, "          \"status\": \"skipped\"\n"); }
            else if (tr->passed)
            {
                fprintf(fp, "          \"status\": \"pass\",\n");
                fprintf(fp, "          \"iterations\": %llu\n", static_cast<unsigned long long>(tr->iterations));
            }
            else
            {
                fprintf(fp, "          \"status\": \"FAIL\",\n");
                fprintf(fp, "          \"iterations\": %llu,\n", static_cast<unsigned long long>(tr->iterations));
                fprintf(fp, "          \"fail_iteration\": %llu,\n", static_cast<unsigned long long>(tr->worst_iter));
                if (std::isfinite(tr->worst_dev))
                    fprintf(fp, "          \"deviation\": %.10f,\n", tr->worst_dev);
                else
                    fprintf(fp, "          \"deviation\": null,\n");
                if (std::isfinite(tr->worst_q[0]) && std::isfinite(tr->worst_q[1]) &&
                    std::isfinite(tr->worst_q[2]) && std::isfinite(tr->worst_q[3]))
                    fprintf(fp,
                            "          \"quaternion\": [%.8f, %.8f, %.8f, %.8f],\n",
                            static_cast<double>(tr->worst_q[0]),
                            static_cast<double>(tr->worst_q[1]),
                            static_cast<double>(tr->worst_q[2]),
                            static_cast<double>(tr->worst_q[3]));
                else
                    fprintf(fp, "          \"quaternion\": null,\n");
                fprintf(fp, "          \"confirmed\": %s,\n", tr->confirmed ? "true" : "false");
                fprintf(fp, "          \"rerun_fails\": %d\n", tr->rerun_fails);
            }
            fprintf(fp, "        }%s\n", t < NUM_TESTS - 1 ? "," : "");
        }
        fprintf(fp, "      }\n");
        fprintf(fp, "    }%s\n", i < num_results - 1 ? "," : "");
    }

    fprintf(fp, "  ]\n");
    fprintf(fp, "}\n");
    fflush(fp);
    bool ok = (ferror(fp) == 0);
    if (fclose(fp) != 0) ok = false;
    if (ok)
    {
        remove(path);
        if (rename(tmp_path, path) != 0) ok = false;
    }
    else { remove(tmp_path); }
    return ok;
}

// ============================================================================
// Help
// ============================================================================

static void print_help(const char* argv0)
{
    printf("\n");
    printf("  coreprobe v%s - Per-Core FPU/SIMD Correctness Diagnostic\n\n", COREPROBE_VERSION);
    printf("  Stress-tests quaternion normalization across SCALAR, SSE3, AVX2, FMA3\n");
    printf("  plus cross-lane AVX2 data integrity (XLANE) on each logical processor\n");
    printf("  to detect faulty floating-point and SIMD execution units.\n");
    printf("  Catches defects that memtest86+, Prime95, and WHEA reporting miss.\n\n");
    printf("  Usage:\n");
    printf("    %s [seconds] [threads...] [flags]\n\n", argv0);
    printf("  Examples:\n");
    printf("    %s                  test all threads, ~120s (min 2s/test)\n", argv0);
    printf("    %s 60               test all threads, ~60s\n", argv0);
    printf("    %s 20 4             test thread 4 only, 20s\n", argv0);
    printf("    %s 20 4 5           test threads 4 and 5\n", argv0);
    printf("    %s 60 0-31          test threads 0-31\n", argv0);
    printf("    %s --soak           extended 10-minute soak test\n", argv0);
    printf("    %s --socket 0       test only socket/package 0\n", argv0);
    printf("    %s --repeat 5       run 5 full passes\n", argv0);
    printf("    %s --until-fail     repeat until failure detected\n", argv0);
    printf("    %s 30 --json        also write coreprobe_results.json\n\n", argv0);
    printf("  Flags:\n");
    printf("    --soak              extended 10-minute soak test (recommended for\n");
    printf("                        intermittent faults or final stability validation)\n");
    printf("    --socket N          test only threads on socket/package N (multi-socket)\n");
    printf("    --repeat N          run N full passes (default: 1)\n");
    printf("    --until-fail        repeat indefinitely until a failure is detected\n");
    printf("    --json              write results to coreprobe_results.json\n");
    printf("    --pause             wait for Enter before exiting (for double-click)\n");
    printf("    --help              show this help\n\n");
    printf("  This is an arithmetic correctness test, not a throughput stress test.\n");
    printf("  -O0 and volatile barriers ensure every FP op executes through hardware.\n");
    printf("  Sequential per-core testing gives clean fault attribution.\n\n");
    printf("  Exit code: 0 = all pass, 1 = failures detected\n\n");
}

// ============================================================================
// Core map
// ============================================================================

static void print_core_map(const CoreResult* all, int num_results, int max_threads, const TopologyInfo& topo)
{
    int cap = max_threads < MAX_THREADS ? max_threads : MAX_THREADS;
    if (cap < 0) cap = 0;
    std::vector<int> thread_status(static_cast<size_t>(cap), 0);
    for (int i = 0; i < num_results; i++)
    {
        if (!all[i].affinity_ok) continue;
        int tid = all[i].thread_id;
        if (tid < 0 || tid >= cap) continue;
        bool ok = true;
        for (int t = 0; t < NUM_TESTS; t++)
        {
            if (!all[i].tests[t].passed && !all[i].tests[t].skipped) ok = false;
        }
        thread_status[static_cast<size_t>(tid)] = ok ? 1 : 2;
    }

    int              core_count = topo.core_count > 0 ? topo.core_count : 0;
    std::vector<int> core_status(static_cast<size_t>(core_count), 0);
    for (int tid = 0; tid < cap; tid++)
    {
        int phys = topo.physical_core[static_cast<size_t>(tid)];
        if (phys >= 0 && phys < core_count)
        {
            if (thread_status[static_cast<size_t>(tid)] > core_status[static_cast<size_t>(phys)]) core_status[static_cast<size_t>(phys)] = thread_status[static_cast<size_t>(tid)];
        }
    }

    printf("  Core Map (%d physical cores%s):\n\n  ", topo.core_count, topo.valid ? ", OS topology" : ", heuristic");

    for (int phys = 0; phys < topo.core_count; phys++)
    {
        if (core_status[static_cast<size_t>(phys)] == 2)
            printf(COL_RED);
        else if (core_status[static_cast<size_t>(phys)] == 1)
            printf(COL_GREEN);
        else
            printf(COL_GRAY);

        printf("[%2d]", phys);
        printf(COL_RESET);

        if ((phys + 1) % 8 == 0 && phys + 1 < topo.core_count) { printf("  |  "); }
        else { printf(" "); }
    }
    printf("\n");

    printf("\n  ");
    printf(COL_GREEN "[OK]" COL_RESET " = pass  ");
    printf(COL_RED "[XX]" COL_RESET " = FAIL  ");
    printf(COL_GRAY "[--]" COL_RESET " = not tested\n");
}

// ============================================================================
// Main – decomposed helpers
// ============================================================================

struct RunConfig
{
    bool json_output     = false;
    bool soak_mode       = false;
    bool pause_at_end    = false;
    bool until_fail      = false;
    int  socket_filter   = -1;
    int  repeat_count    = 1;
    int  total_seconds   = 120;
    int  num_threads     = 0;
    int  max_threads     = 0;
    double secs_per_test   = 0;
    double secs_per_thread = 0;
    int  actual_total      = 0;
    std::vector<int> thread_list;
};

// Returns: -1 = success (continue), 0 = help printed (exit 0), 1 = error (exit 1)
static int parse_args(int argc, char** argv, RunConfig& cfg, const TopologyInfo& topo)
{
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0)
        {
            print_help(argv[0]);
            return 0;
        }
    }

    for (int i = 1; i < argc; i++)
    {
        const char* arg = argv[i];
        if (strcmp(arg, "--json") == 0)
        {
            cfg.json_output = true;
            continue;
        }
        if (strcmp(arg, "--soak") == 0)
        {
            cfg.soak_mode = true;
            continue;
        }
        if (strcmp(arg, "--pause") == 0)
        {
            cfg.pause_at_end = true;
            continue;
        }
        if (strcmp(arg, "--until-fail") == 0)
        {
            cfg.until_fail = true;
            continue;
        }
        if (strcmp(arg, "--socket") == 0)
        {
            if (i + 1 >= argc || !parse_int(argv[i + 1], &cfg.socket_filter))
            {
                fprintf(stderr, COL_RED "  Error: --socket requires an integer\n" COL_RESET);
                return 1;
            }
            if (cfg.socket_filter < 0)
            {
                fprintf(stderr, COL_RED "  Error: --socket must be >= 0\n" COL_RESET);
                return 1;
            }
            i++;
            continue;
        }
        if (strcmp(arg, "--repeat") == 0)
        {
            if (i + 1 >= argc || !parse_int(argv[i + 1], &cfg.repeat_count))
            {
                fprintf(stderr, COL_RED "  Error: --repeat requires an integer\n" COL_RESET);
                return 1;
            }
            i++;
            continue;
        }
        if (arg[0] == '-')
        {
            fprintf(stderr, COL_RED "  Error: unknown flag: %s\n" COL_RESET, arg);
            return 1;
        }
    }
    if (cfg.repeat_count < 1) cfg.repeat_count = 1;
    if (cfg.until_fail) cfg.repeat_count = INT32_MAX;

    cfg.total_seconds = cfg.soak_mode ? 600 : 120;
    int duration_idx  = find_first_positional(argc, argv);
    if (duration_idx > 0)
    {
        int parsed = 0;
        if (!parse_int(argv[duration_idx], &parsed) || parsed <= 0)
        {
            fprintf(stderr, COL_RED "  Error: invalid duration: %s\n" COL_RESET, argv[duration_idx]);
            return 1;
        }
        cfg.total_seconds = parsed;
    }

    cfg.thread_list.resize(static_cast<size_t>(cfg.max_threads));
    bool thread_parse_error = false;
    cfg.num_threads = parse_threads(argc, argv, cfg.thread_list.data(), cfg.max_threads, &thread_parse_error);
    if (thread_parse_error) return 1;

    if (cfg.socket_filter >= 0)
    {
        if (!topo.valid)
        {
            printf(COL_YELLOW "  Warning: --socket %d requested but topology detection failed, "
                              "testing all threads\n" COL_RESET,
                   cfg.socket_filter);
        }
        else
        {
            std::vector<int> filtered;
            filtered.reserve(static_cast<size_t>(cfg.num_threads));
            for (int i = 0; i < cfg.num_threads; i++)
            {
                if (topo.package_id[static_cast<size_t>(cfg.thread_list[static_cast<size_t>(i)])] == cfg.socket_filter) filtered.push_back(cfg.thread_list[static_cast<size_t>(i)]);
            }
            if (!filtered.empty())
            {
                memcpy(cfg.thread_list.data(), filtered.data(), filtered.size() * sizeof(int));
                cfg.num_threads = static_cast<int>(filtered.size());
            }
            else
            {
                printf(COL_RED "  Error: no threads found for socket %d\n" COL_RESET, cfg.socket_filter);
                return 1;
            }
        }
    }

    if (cfg.num_threads == 0)
    {
        printf(COL_RED "  Error: no threads to test\n" COL_RESET);
        return 1;
    }

    cfg.secs_per_thread = static_cast<double>(cfg.total_seconds) / cfg.num_threads;
    cfg.secs_per_test   = cfg.secs_per_thread / static_cast<double>(NUM_TESTS);
    if (cfg.secs_per_test < 2.0) cfg.secs_per_test = 2.0;
    cfg.actual_total = static_cast<int>(cfg.secs_per_test * static_cast<double>(NUM_TESTS) * cfg.num_threads + 0.5);

    return -1;
}

static void print_header(const RunConfig& cfg, const CPUFeatures& cpu, const TopologyInfo& topo)
{
    printf(COL_CYAN "\n");
    printf("  +================================================================+\n");
    printf("  |       coreprobe v%-6s - FPU/SIMD Correctness Diagnostic    |\n", COREPROBE_VERSION);
    printf("  +================================================================+\n\n" COL_RESET);

    printf("  CPU:                 %s\n", cpu.brand);
    printf("  Vendor:              %s\n", cpu.vendor);
    printf("  Logical processors:  %d\n", cfg.max_threads);
    printf("  Physical cores:      %d%s\n",
           topo.core_count,
           topo.valid ? " (OS topology)" : " (heuristic — 1 thread per core)");
    printf("  Instruction sets:    SSE3=%s  AVX2=%s  FMA3=%s\n",
           cpu.has_sse3 ? COL_GREEN "yes" COL_RESET : COL_RED "no" COL_RESET,
           cpu.has_avx2 ? COL_GREEN "yes" COL_RESET : COL_RED "no" COL_RESET,
           cpu.has_fma3 ? COL_GREEN "yes" COL_RESET : COL_RED "no" COL_RESET);
    printf("  OS AVX state:        %s\n",
           cpu.os_avx_enabled ? COL_GREEN "enabled (XSAVE/XGETBV)" COL_RESET
                              : COL_YELLOW "disabled - AVX/FMA/XLANE tests will be skipped" COL_RESET);
    printf("  Testing threads:     ");
    if (cfg.socket_filter >= 0) { printf("%d (socket %d only)\n", cfg.num_threads, cfg.socket_filter); }
    else if (cfg.num_threads == cfg.max_threads) { printf("ALL (%d)\n", cfg.num_threads); }
    else
    {
        for (int i = 0; i < cfg.num_threads; i++)
            printf("%d%s", cfg.thread_list[static_cast<size_t>(i)], i < cfg.num_threads - 1 ? ", " : "\n");
    }
    printf("  Mode:                %s\n", cfg.soak_mode ? "SOAK (extended)" : "standard");
    if (cfg.actual_total != cfg.total_seconds)
        printf("  Duration:            ~%ds total (requested %ds, %.1fs/test, min 2s/test)\n",
               cfg.actual_total,
               cfg.total_seconds,
               cfg.secs_per_test);
    else
        printf("  Duration:            ~%ds total (%.1fs/thread, %.1fs/test)\n",
               cfg.total_seconds,
               cfg.secs_per_thread,
               cfg.secs_per_test);
    printf("  Tolerance:           %.6f\n", TOLERANCE);
    printf("  Rerun on fail:       %dx (deterministic seed replay)\n", RERUN_COUNT);
    printf("  Compile flags:       -O0, volatile floats (correctness test, not throughput)\n");

    if (platform_set_high_priority())
        printf("  Priority:            " COL_GREEN "HIGH" COL_RESET "\n");
    else
        printf("  Priority:            " COL_YELLOW "normal (needs elevated privileges)" COL_RESET "\n");

    if (!cpu.os_avx_enabled)
    {
        printf("  " COL_YELLOW "Warning: OS has not enabled AVX state (XGETBV XCR0 bits 1:2)." COL_RESET "\n");
        printf("  " COL_YELLOW "  AVX2 and FMA3 tests will be skipped. This is unusual on" COL_RESET "\n");
        printf("  " COL_YELLOW "  modern systems -check BIOS settings or OS configuration." COL_RESET "\n");
    }
    else
    {
        if (!cpu.has_avx2)
            printf("  " COL_YELLOW "Note: AVX2 not supported by CPU, test will be skipped" COL_RESET "\n");
        if (!cpu.has_fma3)
            printf("  " COL_YELLOW "Note: FMA3 not supported by CPU, test will be skipped" COL_RESET "\n");
    }
    if (!cpu.has_sse3) printf("  " COL_YELLOW "Note: SSE3 not supported by CPU, test will be skipped" COL_RESET "\n");

    printf("\n");
}

static void print_table_header()
{
    printf(COL_GRAY "  %-5s %-13s %-13s %-13s %-13s %-13s %s" COL_RESET "\n",
           "THR",
           "SCALAR",
           "SSE3",
           "AVX2",
           "FMA3",
           "XLANE",
           "STATUS");
    printf(COL_GRAY "  %-5s %-13s %-13s %-13s %-13s %-13s %s" COL_RESET "\n",
           "---",
           "--------",
           "--------",
           "--------",
           "--------",
           "--------",
           "------");
}

static void print_fail_report(const std::vector<CoreResult>& all,
                              const std::vector<int>&        fail_indices,
                              const TopologyInfo&            topo)
{
    int confirmed_fails = 0;
    for (int fi = 0; fi < static_cast<int>(fail_indices.size()); fi++)
    {
        const CoreResult* cr = &all[static_cast<size_t>(fail_indices[static_cast<size_t>(fi)])];
        for (int t = 0; t < NUM_TESTS; t++)
        {
            if (!cr->tests[t].passed && !cr->tests[t].skipped && cr->tests[t].confirmed) confirmed_fails++;
        }
    }

    printf(COL_RED "  *** FPU ERRORS DETECTED ***" COL_RESET "\n\n");

    for (int fi = 0; fi < static_cast<int>(fail_indices.size()); fi++)
    {
        const CoreResult* cr   = &all[static_cast<size_t>(fail_indices[static_cast<size_t>(fi)])];
        int               tid  = cr->thread_id;
        int               phys = (tid >= 0 && tid < static_cast<int>(topo.physical_core.size())) ? topo.physical_core[static_cast<size_t>(tid)] : -1;
        int               pkg  = (tid >= 0 && tid < static_cast<int>(topo.package_id.size())) ? topo.package_id[static_cast<size_t>(tid)] : -1;

        printf("  Thread %d (physical core %d, package %d%s):\n",
               cr->thread_id,
               phys,
               pkg,
               topo.valid ? "" : ", heuristic");

        for (int t = 0; t < NUM_TESTS; t++)
        {
            const TestResult* tr = &cr->tests[t];
            if (tr->skipped) continue;
            if (!tr->passed)
            {
                printf("    " COL_RED "%s FAILED" COL_RESET, tname[t]);
                printf(" at iter %llu  deviation=%.10f", static_cast<unsigned long long>(tr->worst_iter), tr->worst_dev);
                if (tr->confirmed)
                    printf("  " COL_RED "[confirmed %d/%d reruns]" COL_RESET, tr->rerun_fails, RERUN_COUNT);
                else
                    printf("  " COL_YELLOW "[transient, 0/%d reruns]" COL_RESET, RERUN_COUNT);
                printf("\n");
                printf("      quat(%.8f, %.8f, %.8f, %.8f)\n",
                       static_cast<double>(tr->worst_q[0]),
                       static_cast<double>(tr->worst_q[1]),
                       static_cast<double>(tr->worst_q[2]),
                       static_cast<double>(tr->worst_q[3]));
            }
            else { printf("    " COL_GREEN "%s OK" COL_RESET "\n", tname[t]); }
        }
        printf("\n");
    }

    printf("  " COL_MAGENTA "Diagnosis:" COL_RESET "\n");
    printf("  Affected physical core(s): ");
    std::vector<bool> seen_core(static_cast<size_t>(topo.core_count > 0 ? topo.core_count : 0), false);
    for (int fi = 0; fi < static_cast<int>(fail_indices.size()); fi++)
    {
        int tid = all[static_cast<size_t>(fail_indices[static_cast<size_t>(fi)])].thread_id;
        if (tid < 0 || tid >= static_cast<int>(topo.physical_core.size())) continue;
        int phys = topo.physical_core[static_cast<size_t>(tid)];
        if (phys >= 0 && phys < static_cast<int>(seen_core.size()) && !seen_core[static_cast<size_t>(phys)])
        {
            printf(COL_RED "Core %d " COL_RESET, phys);
            seen_core[static_cast<size_t>(phys)] = true;
        }
    }
    printf("\n");

    bool scalar_ok_simd_fail = false;
    for (int fi = 0; fi < static_cast<int>(fail_indices.size()); fi++)
    {
        const CoreResult* cr = &all[static_cast<size_t>(fail_indices[static_cast<size_t>(fi)])];
        if (cr->tests[T_SCALAR].passed)
        {
            for (int t = T_SSE3; t < NUM_TESTS; t++)
            {
                if (!cr->tests[t].passed && !cr->tests[t].skipped) scalar_ok_simd_fail = true;
            }
        }
    }

    if (scalar_ok_simd_fail)
    {
        printf("\n  " COL_YELLOW "Pattern: SCALAR passes but SIMD fails" COL_RESET "\n");
        printf("  This indicates SIMD execution units (SSE/AVX/FMA/lane-crossing)\n");
        printf("  are faulty while the scalar FP pipeline is intact. Common causes:\n");
        printf("    - Silicon defect in SIMD execution unit on affected core\n");
        printf("    - Degraded CPU (age, heat damage, electromigration)\n");
        printf("    - If on OC/PBO: reduce clocks or increase voltage\n");
        printf("    - If on stock: CPU hardware fault, consider RMA or replacement\n");
    }

    bool xlane_only = false;
    for (int fi = 0; fi < static_cast<int>(fail_indices.size()); fi++)
    {
        const CoreResult* cr = &all[static_cast<size_t>(fail_indices[static_cast<size_t>(fi)])];
        if (!cr->tests[T_XLANE].skipped && !cr->tests[T_XLANE].passed && cr->tests[T_SCALAR].passed &&
            cr->tests[T_SSE3].passed && (cr->tests[T_AVX2].passed || cr->tests[T_AVX2].skipped) &&
            (cr->tests[T_FMA3].passed || cr->tests[T_FMA3].skipped))
            xlane_only = true;
    }
    if (xlane_only)
    {
        printf("\n  " COL_YELLOW "Pattern: only XLANE fails" COL_RESET "\n");
        printf("  Arithmetic is correct but cross-lane data movement is corrupted.\n");
        printf("  This points to the AVX2 lane-crossing interconnect specifically.\n");
    }

    if (confirmed_fails > 0)
    {
        printf("\n  " COL_RED "Confirmed failures reproduce with identical seeds." COL_RESET "\n");
        printf("  " COL_RED "This is a hardware defect, not a transient error." COL_RESET "\n");
    }
}

// ============================================================================
// Main – test execution helpers
// ============================================================================

static bool should_skip_test(int t, const CPUFeatures& cpu)
{
    if (t == T_SSE3 && !cpu.has_sse3) return true;
    if (t == T_AVX2 && !cpu.has_avx2) return true;
    if (t == T_FMA3 && !cpu.has_fma3) return true;
    if (t == T_XLANE && !cpu.has_avx2) return true;
    return false;
}

static void print_test_pass(const TestResult& tr, int t)
{
    char buf[32];
    double wd = tr.worst_dev;
    if (t != T_XLANE && wd > WARN_THRESHOLD)
    {
        snprintf(buf, sizeof(buf), "WARN %.3f%%", wd * 100.0);
        printf(COL_YELLOW "%-13s" COL_RESET, buf);
    }
    else
    {
        snprintf(buf, sizeof(buf), "PASS %lluM",
                 static_cast<unsigned long long>(tr.iterations / 1000000ULL));
        printf(COL_GREEN "%-13s" COL_RESET, buf);
    }
}

static void confirm_and_print_fail(TestResult& tr, int t, uint32_t seed, double secs_per_test)
{
    tr.rerun_fails = 0;
    double rerun_dur = secs_per_test > RERUN_DURATION ? secs_per_test : RERUN_DURATION;
    for (int rr = 0; rr < RERUN_COUNT; rr++)
    {
        TestResult rerun = rerun_single(test_funcs[t], seed, rerun_dur);
        if (!rerun.passed) tr.rerun_fails++;
    }
    tr.confirmed = (tr.rerun_fails > 0);

    char buf[32];
    if (t == T_XLANE) { snprintf(buf, sizeof(buf), "FAIL mismatch"); }
    else { snprintf(buf, sizeof(buf), "FAIL %.3f%%", tr.worst_dev * 100.0); }

    if (tr.confirmed) printf(COL_RED "%-13s" COL_RESET, buf);
    else              printf(COL_YELLOW "%-13s" COL_RESET, buf);
}

static bool run_thread_tests(CoreResult& cr, int tid, int pass,
                             const CPUFeatures& cpu, const RunConfig& cfg)
{
    if (!platform_set_affinity(tid))
    {
        cr.affinity_ok = false;
        printf("  " COL_RED "T%-3d  affinity FAILED, skipping" COL_RESET "\n", tid);
        return false;
    }
    cr.affinity_ok = true;
    printf("  " COL_YELLOW "T%-3d" COL_RESET "  ", tid);
    fflush(stdout);

    bool core_ok = true;
    PRNG rng;

    for (int t = 0; t < NUM_TESTS; t++)
    {
        if (g_stop) break;

        if (should_skip_test(t, cpu))
        {
            cr.tests[t].skipped = true;
            cr.tests[t].passed  = true;
            printf(COL_GRAY "%-13s" COL_RESET, "skip");
            fflush(stdout);
            continue;
        }

        uint32_t seed = 0xF00D0000u + (static_cast<uint32_t>(pass) << 20) +
                        static_cast<uint32_t>(tid) * NUM_TESTS + static_cast<uint32_t>(t);
        rng.seed(seed);
        cr.tests[t]           = test_funcs[t](rng, now_sec() + cfg.secs_per_test);
        cr.tests[t].fail_seed = seed;

        if (cr.tests[t].passed) { print_test_pass(cr.tests[t], t); }
        else
        {
            core_ok = false;
            confirm_and_print_fail(cr.tests[t], t, seed, cfg.secs_per_test);
        }
        fflush(stdout);
    }

    if (core_ok) printf(COL_GREEN " OK" COL_RESET);
    else         printf(COL_RED " ** FAIL **" COL_RESET);
    printf("\n");
    return true;
}

static int tally_failures(const std::vector<CoreResult>& all, std::vector<int>& fail_indices)
{
    int total = 0;
    for (size_t ci = 0; ci < all.size(); ci++)
    {
        if (!all[ci].affinity_ok) continue;
        bool has_fail = false;
        for (int t = 0; t < NUM_TESTS; t++)
        {
            if (!all[ci].tests[t].passed && !all[ci].tests[t].skipped)
            {
                total++;
                has_fail = true;
            }
        }
        if (has_fail) fail_indices.push_back(static_cast<int>(ci));
    }
    return total;
}

static int print_pass_summary(const std::vector<CoreResult>& all, int affinity_fails,
                               const RunConfig& cfg, const CPUFeatures& cpu,
                               const TopologyInfo& topo, double wall_secs)
{
    printf("\n");
    printf(COL_CYAN "  +================================================================+\n");
    printf("  |                        SUMMARY                                 |\n");
    printf("  +================================================================+\n" COL_RESET);
    printf("\n");

    std::vector<int> fail_indices;
    int total_fails = tally_failures(all, fail_indices);

    if (total_fails > 0)
        print_fail_report(all, fail_indices, topo);
    else if (affinity_fails == cfg.num_threads)
        printf(COL_RED "  *** No tests ran: affinity failed on every requested thread ***" COL_RESET "\n");
    else
    {
        printf(COL_GREEN "  ALL TESTS PASSED -no FPU/SIMD errors detected." COL_RESET "\n");
        if (affinity_fails > 0)
            printf(COL_YELLOW "  Note: %d thread(s) skipped due to affinity failures." COL_RESET "\n",
                   affinity_fails);
    }

    printf("\n");
    print_core_map(all.data(), cfg.num_threads, cfg.max_threads, topo);
    printf("\n  Wall time: %.1f seconds\n", wall_secs);

    if (cfg.json_output)
    {
        if (write_json("coreprobe_results.json", cpu, topo, all.data(), cfg.num_threads, wall_secs))
            printf("  Results written to: coreprobe_results.json\n");
        else
            printf(COL_YELLOW "  Warning: JSON write encountered errors" COL_RESET "\n");
    }

    printf("\n");
    return total_fails;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv)
{
    platform_init();

    int max_threads = platform_num_threads();
    if (max_threads <= 0)
    {
        fprintf(stderr, COL_RED "  Error: could not determine CPU count\n" COL_RESET);
        return 1;
    }
    if (max_threads > MAX_THREADS) max_threads = MAX_THREADS;
    CPUFeatures  cpu  = detect_cpu();
    TopologyInfo topo = detect_topology(max_threads);

    RunConfig cfg;
    cfg.max_threads = max_threads;
    int parse_rc = parse_args(argc, argv, cfg, topo);
    if (parse_rc >= 0) return parse_rc;

    print_header(cfg, cpu, topo);
    print_table_header();

    int overall_fails = 0;
    int pass_number   = 0;

    for (int pass = 0; pass < cfg.repeat_count && !g_stop; pass++)
    {
        pass_number = pass + 1;
        if (cfg.repeat_count > 1)
        {
            printf(COL_CYAN "\n  === Pass %d%s ===" COL_RESET "\n\n",
                   pass_number, cfg.until_fail ? " (until-fail mode)" : "");
            print_table_header();
        }

        std::vector<CoreResult> all(static_cast<size_t>(cfg.num_threads));
        int    affinity_fails = 0;
        double wall_start     = now_sec();

        for (int ci = 0; ci < cfg.num_threads && !g_stop; ci++)
        {
            int tid = cfg.thread_list[static_cast<size_t>(ci)];
            all[static_cast<size_t>(ci)].thread_id = tid;
            if (!run_thread_tests(all[static_cast<size_t>(ci)], tid, pass, cpu, cfg))
                affinity_fails++;
        }

        int total_fails = print_pass_summary(all, affinity_fails, cfg, cpu, topo, now_sec() - wall_start);

        overall_fails += total_fails;
        if (affinity_fails == cfg.num_threads && overall_fails == 0) overall_fails = 1;

        if (total_fails > 0) break;
        if (cfg.repeat_count > 1 && pass + 1 < cfg.repeat_count)
            printf("  Pass %d complete -no errors. Continuing...\n", pass_number);
    }

    if (g_stop)
        printf(COL_YELLOW "\n  Interrupted by user." COL_RESET "\n");
    else if (cfg.repeat_count > 1 && overall_fails == 0)
        printf(COL_GREEN "\n  All %d passes completed with no failures." COL_RESET "\n", pass_number);

    if (cfg.pause_at_end)
    {
        printf("  Press Enter to exit...");
        fflush(stdout);
        getchar();
    }

    return overall_fails > 0 ? 1 : 0;
}
