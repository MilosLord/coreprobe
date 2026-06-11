/*
 * coreprobe - Per-Core FPU/SIMD Correctness Diagnostic
 * https://github.com/MilosLord/coreprobe
 *
 * Detects faulty floating-point execution units by stress-testing quaternion
 * normalization across SCALAR, SSE3, AVX2, FMA3, AVX-512F, and cross-lane
 * AVX2 (XLANE) on each logical processor independently. Catches silicon
 * defects that memtest86+, Prime95, and WHEA miss.
 *
 * This is an arithmetic CORRECTNESS test, not a throughput stress test.
 * -O0 and volatile barriers ensure every FP op executes through hardware.
 *
 * License: MIT
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#define COREPROBE_VERSION "1.1.0"
#define MAX_THREADS       4096

#include <bit>
#include <cerrno>
#include <climits>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static volatile sig_atomic_t g_stop         = 0;
static volatile sig_atomic_t g_pause_prompt = 0;
static volatile sig_atomic_t g_exit_code    = 130;

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

// Per-function ISA target attributes (GCC/Clang; MSVC exposes intrinsics without /arch:)
#if defined(__GNUC__) || defined(__clang__)
#define TARGET_SSE3   __attribute__((target("sse3")))
#define TARGET_AVX2   __attribute__((target("avx2")))
#define TARGET_FMA    __attribute__((target("avx,fma")))
#define TARGET_AVX512 __attribute__((target("avx512f")))
#define TARGET_XSAVE  __attribute__((target("xsave")))
#else
#define TARGET_SSE3
#define TARGET_AVX2
#define TARGET_FMA
#define TARGET_AVX512
#define TARGET_XSAVE
#endif

// MinGW GCC misaligns 256/512-bit vector args across non-inlined calls; keep these force-inlined.
#if defined(__GNUC__) || defined(__clang__)
#define VEC_INLINE inline __attribute__((always_inline))
#else
#define VEC_INLINE inline
#endif

// ============================================================================
// Platform abstraction
// ============================================================================

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <intrin.h>

static BOOL WINAPI console_ctrl_handler(DWORD ctrl_type)
{
    if (ctrl_type != CTRL_C_EVENT && ctrl_type != CTRL_BREAK_EVENT) return FALSE;
    if (g_pause_prompt != 0) _Exit(g_exit_code);
    if (g_stop != 0) _Exit(130);
    g_stop = 1;
    return TRUE;
}

static void platform_init()
{
    HANDLE console = GetStdHandle(STD_OUTPUT_HANDLE);
    if (console != INVALID_HANDLE_VALUE && console != nullptr)
    {
        DWORD mode = 0;
        if (GetConsoleMode(console, &mode) != 0) SetConsoleMode(console, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
    }
    SetConsoleCtrlHandler(console_ctrl_handler, TRUE);
}

struct CpuSlot
{
    WORD group  = 0;
    BYTE number = 0;
};

static std::vector<uint8_t> win_query_relation(LOGICAL_PROCESSOR_RELATIONSHIP rel)
{
    DWORD len = 0;
    GetLogicalProcessorInformationEx(rel, nullptr, &len);
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || len == 0) return {};
    std::vector<uint8_t> buf(len);
    auto*                info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data());
    if (GetLogicalProcessorInformationEx(rel, info, &len) == 0) return {};
    return buf;
}

static void win_append_group_slots(WORD group, KAFFINITY mask, std::vector<CpuSlot>& slots)
{
    for (BYTE bit = 0; bit < 64; bit++)
    {
        if ((mask & (static_cast<KAFFINITY>(1) << bit)) != 0) slots.push_back({group, bit});
    }
}

static std::vector<CpuSlot> win_build_cpu_slots()
{
    std::vector<CpuSlot> slots;
    std::vector<uint8_t> buf = win_query_relation(RelationGroup);
    DWORD                off = 0;
    while (off < buf.size())
    {
        const auto* info = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data() + off);
        if (info->Size == 0 || off + info->Size > buf.size()) break;
        if (info->Relationship == RelationGroup)
        {
            for (WORD g = 0; g < info->Group.ActiveGroupCount; g++)
                win_append_group_slots(g, info->Group.GroupInfo[g].ActiveProcessorMask, slots);
        }
        off += info->Size;
    }
    if (!slots.empty()) return slots;
    for (WORD g = 0; g < GetActiveProcessorGroupCount(); g++)
    {
        for (DWORD p = 0; p < GetActiveProcessorCount(g); p++) slots.push_back({g, static_cast<BYTE>(p)});
    }
    return slots;
}

static const std::vector<CpuSlot>& win_cpu_slots()
{
    static std::vector<CpuSlot> slots = win_build_cpu_slots();
    return slots;
}

static int win_linear_id(WORD group, int bit)
{
    const std::vector<CpuSlot>& slots = win_cpu_slots();
    for (size_t k = 0; k < slots.size(); k++)
    {
        if (slots[k].group == group && static_cast<int>(slots[k].number) == bit) return static_cast<int>(k);
    }
    return -1;
}

static int platform_num_threads()
{
    int n = static_cast<int>(win_cpu_slots().size());
    if (n > 0) return n;
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return static_cast<int>(si.dwNumberOfProcessors);
}

static bool win_affinity_landed(WORD group, BYTE number)
{
    for (int tries = 0; tries < 64; tries++)
    {
        PROCESSOR_NUMBER pn = {};
        GetCurrentProcessorNumberEx(&pn);
        if (pn.Group == group && pn.Number == number) return true;
        Sleep(tries < 8 ? 0 : 1);
    }
    return false;
}

static bool platform_set_affinity(int thread_id)
{
    const std::vector<CpuSlot>& slots = win_cpu_slots();
    if (thread_id < 0 || static_cast<size_t>(thread_id) >= slots.size()) return false;
    CpuSlot        slot = slots[static_cast<size_t>(thread_id)];
    GROUP_AFFINITY ga   = {};
    ga.Group            = slot.group;
    ga.Mask             = static_cast<KAFFINITY>(1) << slot.number;
    if (SetThreadGroupAffinity(GetCurrentThread(), &ga, nullptr) == 0) return false;
    return win_affinity_landed(slot.group, slot.number);
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
#include <ctime>

#include <cpuid.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

static void posix_signal_handler(int)
{
    if (g_pause_prompt != 0) _Exit(g_exit_code);
    if (g_stop != 0) _Exit(130);
    g_stop = 1;
}

static void platform_init()
{
    struct sigaction sa = {};
    sa.sa_handler       = posix_signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART;
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);
}

static int platform_num_threads()
{
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? static_cast<int>(n) : 0;
}

static bool linux_affinity_landed(int cpu)
{
    for (int tries = 0; tries < 64; tries++)
    {
        if (sched_getcpu() == cpu) return true;
        if (tries < 8)
        {
            sched_yield();
        } else
        {
            struct timespec ts = {0, 1'000'000};
            nanosleep(&ts, nullptr);
        }
    }
    return false;
}

static bool platform_set_affinity(int thread_id)
{
    int num_cpus = static_cast<int>(sysconf(_SC_NPROCESSORS_CONF));
    if (num_cpus < thread_id + 1) num_cpus = thread_id + 1;
    size_t     size   = CPU_ALLOC_SIZE(static_cast<unsigned>(num_cpus));
    cpu_set_t* cpuset = CPU_ALLOC(static_cast<unsigned>(num_cpus));
    if (cpuset == nullptr) return false;
    CPU_ZERO_S(size, cpuset);
    CPU_SET_S(static_cast<unsigned>(thread_id), size, cpuset);
    int ret = pthread_setaffinity_np(pthread_self(), size, cpuset);
    CPU_FREE(cpuset);
    if (ret != 0) return false;
    return linux_affinity_landed(thread_id);
}

static bool platform_set_high_priority()
{
    errno = 0;
    if (nice(-20) == -1 && errno != 0) return false;
    return errno == 0;
}

static double now_sec()
{
    struct timespec ts = {};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) / 1e9;
}

// NOLINTNEXTLINE(readability-non-const-parameter) - the asm macro writes through out
static void cpuid(uint32_t leaf, uint32_t subleaf, uint32_t out[4])
{ __cpuid_count(leaf, subleaf, out[0], out[1], out[2], out[3]); }
#endif

// ============================================================================
// CPU topology detection
// ============================================================================

struct TopologyInfo
{
    std::vector<int> physical_core;
    std::vector<int> package_id;
    int              core_count = 0;
    bool             valid      = false;
};

#ifdef _WIN32

static void win_mark_mask(const GROUP_AFFINITY& gm, std::vector<int>& out, int idx)
{
    for (int bit = 0; bit < 64; bit++)
    {
        if ((gm.Mask & (static_cast<KAFFINITY>(1) << bit)) == 0) continue;
        int tid = win_linear_id(gm.Group, bit);
        if (tid >= 0 && tid < static_cast<int>(out.size())) out[static_cast<size_t>(tid)] = idx;
    }
}

static int win_walk_relation(const std::vector<uint8_t>& buf, int relationship, std::vector<int>& out)
{
    int   idx = 0;
    DWORD off = 0;
    while (off < buf.size())
    {
        const auto* info = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data() + off);
        if (info->Size == 0 || off + info->Size > buf.size()) break;
        if (static_cast<int>(info->Relationship) == relationship)
        {
            for (WORD g = 0; g < info->Processor.GroupCount; g++) win_mark_mask(info->Processor.GroupMask[g], out, idx);
            idx++;
        }
        off += info->Size;
    }
    return idx;
}

static void platform_detect_topology(TopologyInfo& topo, int /*n*/)
{
    std::vector<uint8_t> cores = win_query_relation(RelationProcessorCore);
    if (cores.empty()) return;
    int count = win_walk_relation(cores, RelationProcessorCore, topo.physical_core);
    if (count <= 0) return;
    topo.core_count           = count;
    topo.valid                = true;
    std::vector<uint8_t> pkgs = win_query_relation(RelationProcessorPackage);
    if (!pkgs.empty()) win_walk_relation(pkgs, RelationProcessorPackage, topo.package_id);
}

#else

struct PkgCore
{
    int pkg  = 0;
    int core = 0;
};

static bool read_sysfs_int(const char* path, int* out)
{
    FILE* f = fopen(path, "r");
    if (f == nullptr) return false;
    int  v  = 0;
    bool ok = fscanf(f, "%d", &v) == 1;
    fclose(f);
    if (ok) *out = v;
    return ok;
}

static bool linux_cpu_ids(int cpu, int* pkg, int* core)
{
    char path[128];
    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/core_id", cpu);
    if (!read_sysfs_int(path, core)) return false;
    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", cpu);
    return read_sysfs_int(path, pkg);
}

static void linux_assign_cores(const std::vector<PkgCore>& raw, TopologyInfo& topo)
{
    std::vector<PkgCore> unique;
    unique.reserve(raw.size());
    for (size_t i = 0; i < raw.size(); i++)
    {
        int idx = -1;
        for (size_t u = 0; u < unique.size(); u++)
        {
            if (unique[u].pkg == raw[i].pkg && unique[u].core == raw[i].core)
            {
                idx = static_cast<int>(u);
                break;
            }
        }
        if (idx < 0)
        {
            idx = static_cast<int>(unique.size());
            unique.push_back(raw[i]);
        }
        topo.physical_core[i] = idx;
    }
    topo.core_count = static_cast<int>(unique.size());
    topo.valid      = true;
}

static void platform_detect_topology(TopologyInfo& topo, int n)
{
    std::vector<PkgCore> raw(static_cast<size_t>(n));
    for (int i = 0; i < n; i++)
    {
        if (!linux_cpu_ids(i, &raw[static_cast<size_t>(i)].pkg, &raw[static_cast<size_t>(i)].core)) return;
        topo.package_id[static_cast<size_t>(i)] = raw[static_cast<size_t>(i)].pkg;
    }
    linux_assign_cores(raw, topo);
}

#endif

static TopologyInfo detect_topology(int max_threads)
{
    TopologyInfo topo;
    int          n = max_threads < MAX_THREADS ? max_threads : MAX_THREADS;
    topo.physical_core.resize(static_cast<size_t>(n));
    topo.package_id.assign(static_cast<size_t>(n), 0);
    for (int i = 0; i < n; i++) topo.physical_core[static_cast<size_t>(i)] = i;
    topo.core_count = n;
    topo.valid      = false;
    platform_detect_topology(topo, n);
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
    bool has_sse           = false;
    bool has_sse3          = false;
    bool has_avx2          = false;
    bool has_fma3          = false;
    bool has_avx512f       = false;
    bool cpu_avx512f       = false; // CPUID bit, regardless of OS state
    bool os_avx_enabled    = false;
    bool os_avx512_enabled = false;
    char brand[49]         = {};
    char vendor[13]        = {};
};

// Check if OS enabled AVX state via XGETBV (required for AVX/FMA instructions)
TARGET_XSAVE
static uint64_t xgetbv(uint32_t xcr)
{
#ifdef _WIN32
    return static_cast<uint64_t>(_xgetbv(xcr));
#else
    uint32_t lo = 0, hi = 0;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(xcr));
    return (static_cast<uint64_t>(hi) << 32) | lo;
#endif
}

static void detect_brand(CPUFeatures& f)
{
    uint32_t r[4];
    cpuid(0x80'00'00'00, 0, r);
    if (r[0] >= 0x80'00'00'04)
    {
        cpuid(0x80'00'00'02, 0, r);
        memcpy(f.brand + 0, r, 16);
        cpuid(0x80'00'00'03, 0, r);
        memcpy(f.brand + 16, r, 16);
        cpuid(0x80'00'00'04, 0, r);
        memcpy(f.brand + 32, r, 16);
        f.brand[48] = 0;
        char* p     = f.brand;
        while (*p == ' ') p++;
        if (p != f.brand) memmove(f.brand, p, strlen(p) + 1);
        size_t len = strlen(f.brand);
        while (len > 0 && f.brand[len - 1] == ' ') f.brand[--len] = '\0';
    }
    if (f.brand[0] == 0) snprintf(f.brand, sizeof(f.brand), "(unknown)");
}

// XCR0 bits: 1|2 = SSE|AVX state, 5|6|7 = opmask|ZMM_Hi256|Hi16_ZMM
static void detect_os_xsave_state(CPUFeatures& f, bool has_osxsave, bool cpu_avx)
{
    if (!has_osxsave || !cpu_avx) return;
    uint64_t xcr0       = xgetbv(0);
    f.os_avx_enabled    = (xcr0 & 0x06) == 0x06;
    f.os_avx512_enabled = (xcr0 & 0xE6) == 0xE6;
}

static void detect_features(CPUFeatures& f, uint32_t max_basic)
{
    uint32_t r[4];
    bool     has_osxsave = false;
    bool     cpu_fma3 = false, cpu_avx = false, cpu_avx2 = false;
    if (max_basic >= 1)
    {
        cpuid(1, 0, r);
        f.has_sse   = (r[3] & (1U << 25)) != 0;
        f.has_sse3  = (r[2] & (1U << 0)) != 0;
        has_osxsave = (r[2] & (1U << 27)) != 0;
        cpu_fma3    = (r[2] & (1U << 12)) != 0;
        cpu_avx     = (r[2] & (1U << 28)) != 0;
    }
    if (max_basic >= 7)
    {
        cpuid(7, 0, r);
        cpu_avx2      = (r[1] & (1U << 5)) != 0;
        f.cpu_avx512f = (r[1] & (1U << 16)) != 0;
    }
    detect_os_xsave_state(f, has_osxsave, cpu_avx);
    f.has_avx2    = cpu_avx2 && f.os_avx_enabled;
    f.has_fma3    = cpu_fma3 && f.os_avx_enabled;
    f.has_avx512f = f.cpu_avx512f && f.os_avx512_enabled;
}

static CPUFeatures detect_cpu()
{
    CPUFeatures f;
    uint32_t    r[4];
    cpuid(0, 0, r);
    uint32_t max_basic = r[0];
    memcpy(f.vendor + 0, &r[1], 4);
    memcpy(f.vendor + 4, &r[3], 4);
    memcpy(f.vendor + 8, &r[2], 4);
    f.vendor[12] = 0;
    detect_brand(f);
    detect_features(f, max_basic);
    return f;
}

// ============================================================================
// PRNG -xoshiro128**
// ============================================================================

struct PRNG
{
    uint32_t s[4] = {};

    void seed(uint32_t v)
    {
        for (uint32_t& slot : s)
        {
            v          += 0x9E'37'79'B9U;
            uint32_t z  = v;
            z           = (z ^ (z >> 16)) * 0x85'EB'CA'6BU;
            z           = (z ^ (z >> 13)) * 0xC2'B2'AE'35U;
            slot        = z ^ (z >> 16);
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
static const uint64_t ITER_CHECK_FREQ = 0x3'FF'FF;

enum TestType
{
    T_SCALAR = 0,
    T_SSE3,
    T_AVX2,
    T_FMA3,
    T_AVX512,
    T_XLANE,
    NUM_TESTS
};
static const char* const tname[NUM_TESTS] = {"SCALAR", "SSE3", "AVX2", "FMA3", "AVX512", "XLANE"};

struct TestResult
{
    bool     passed      = false;
    bool     skipped     = false;
    bool     confirmed   = false;
    int      rerun_fails = 0;
    uint64_t iterations  = 0;
    double   worst_dev   = 0.0;
    float    worst_q[4]  = {};
    uint64_t worst_iter  = 0;
    uint32_t fail_seed   = 0;
};

// ============================================================================
// Test functions (optimization-protected)
// ============================================================================
COREPROBE_NO_OPTIMIZE

static bool test_should_stop(uint64_t i, double deadline)
{
    if (g_stop != 0) return true;
    return (i & ITER_CHECK_FREQ) == 0 && now_sec() >= deadline;
}

static void record_fail(TestResult& r, double dev, uint64_t iter, const float q[4])
{
    r.passed     = false;
    r.worst_dev  = dev;
    r.worst_iter = iter;
    for (int k = 0; k < 4; k++) r.worst_q[k] = q[k];
    r.iterations = iter;
}

static void record_fail_m128(TestResult& r, __m128 v, double dev, uint64_t iter)
{
    float tmp[4];
    _mm_storeu_ps(tmp, v);
    record_fail(r, dev, iter, tmp);
}

static bool check_pair_devs(TestResult& r, __m128 na, __m128 nb, double da, double db, uint64_t iter)
{
    double worst = da > db ? da : db;
    if (std::isfinite(worst) && worst > r.worst_dev) r.worst_dev = worst;
    if (!std::isfinite(da) || !std::isfinite(db) || worst > TOLERANCE)
    {
        record_fail_m128(r, da > db ? na : nb, worst, iter);
        return false;
    }
    r.iterations = iter;
    return true;
}

// SCALAR -single-operation FP pipeline

static TestResult run_scalar(PRNG& rng, double deadline)
{
    TestResult r;
    r.passed = true;
    for (uint64_t i = 1;; i++)
    {
        if (test_should_stop(i, deadline)) break;
        volatile float qx = rng.randf(), qy = rng.randf();
        volatile float qz = rng.randf(), qw = rng.randf();
        volatile float x = qx, y = qy, z = qz, w = qw;
        volatile float ls = x * x + y * y + z * z + w * w;
        if (ls < 1e-10F) continue;
        volatile float il = 1.0F / sqrtf(static_cast<float>(ls));
        volatile float nx = x * il, ny = y * il, nz = z * il, nw = w * il;
        volatile float ck  = nx * nx + ny * ny + nz * nz + nw * nw;
        double         dev = fabs(static_cast<double>(static_cast<float>(ck)) - 1.0);
        if (std::isfinite(dev) && dev > r.worst_dev) r.worst_dev = dev;
        if (!std::isfinite(dev) || dev > TOLERANCE)
        {
            float q[4] = {nx, ny, nz, nw};
            record_fail(r, dev, i, q);
            return r;
        }
        r.iterations = i;
    }
    return r;
}

// SSE3 helper - dot product for verification
TARGET_SSE3
static float sse3_dot4(__m128 v)
{
    __m128 sq2 = _mm_mul_ps(v, v);
    __m128 h1 = _mm_hadd_ps(sq2, sq2), h2 = _mm_hadd_ps(h1, h1);
    float  c_raw = 0.0F;
    _mm_store_ss(&c_raw, h2);
    volatile float c = c_raw;
    return c;
}

TARGET_SSE3
static void sse3_normalize(__m128 q, __m128 lv, __m128* na, __m128* nb)
{
    __m128 inv = _mm_rsqrt_ps(lv);
    __m128 nr  = _mm_mul_ps(_mm_mul_ps(_mm_set1_ps(0.5F), lv), _mm_mul_ps(inv, inv));
    inv        = _mm_mul_ps(inv, _mm_sub_ps(_mm_set1_ps(1.5F), nr));
    *na        = _mm_mul_ps(q, inv);
    *nb        = _mm_div_ps(q, _mm_sqrt_ps(lv));
}

// SSE3 -128-bit SIMD

TARGET_SSE3
static TestResult run_sse(PRNG& rng, double deadline)
{
    TestResult r;
    r.passed = true;
    for (uint64_t i = 1;; i++)
    {
        if (test_should_stop(i, deadline)) break;
        volatile float qx = rng.randf(), qy = rng.randf(), qz = rng.randf(), qw = rng.randf();
        __m128         q =
            _mm_set_ps(static_cast<float>(qw), static_cast<float>(qz), static_cast<float>(qy), static_cast<float>(qx));
        __m128 sq = _mm_mul_ps(q, q);
        __m128 s1 = _mm_hadd_ps(sq, sq), s2 = _mm_hadd_ps(s1, s1);
        float  ls_raw = 0.0F;
        _mm_store_ss(&ls_raw, s2);
        volatile float ls = ls_raw;
        if (static_cast<float>(ls) < 1e-10F) continue;
        __m128 na, nb;
        sse3_normalize(q, _mm_shuffle_ps(s2, s2, 0), &na, &nb);
        double da = fabs(static_cast<double>(sse3_dot4(na)) - 1.0);
        double db = fabs(static_cast<double>(sse3_dot4(nb)) - 1.0);
        if (!check_pair_devs(r, na, nb, da, db, i)) break;
    }
    return r;
}

// AVX2 helpers - per-quaternion dot products and lane extraction

TARGET_AVX2
static VEC_INLINE __m256 avx2_quad_dots(__m256 v)
{
    __m256 sq = _mm256_mul_ps(v, v);
    __m256 h1 = _mm256_hadd_ps(sq, sq);
    return _mm256_hadd_ps(h1, h1);
}

TARGET_AVX2
static VEC_INLINE void avx2_extract_pair(__m256 v, float* lo, float* hi)
{
    float lo_raw = 0.0F, hi_raw = 0.0F;
    _mm_store_ss(&lo_raw, _mm256_castps256_ps128(v));
    _mm_store_ss(&hi_raw, _mm256_extractf128_ps(v, 1));
    volatile float l = lo_raw, h = hi_raw;
    *lo = l;
    *hi = h;
}

TARGET_AVX2
static VEC_INLINE void avx2_record(TestResult& r, __m256 norm, double d1, double d2, uint64_t iter)
{
    float tmp[8];
    _mm256_storeu_ps(tmp, norm);
    record_fail(r, d1 > d2 ? d1 : d2, iter, tmp + (d1 > d2 ? 0 : 4));
}

TARGET_AVX2
static VEC_INLINE bool avx2_check_devs(TestResult& r, __m256 norm, double d1, double d2, uint64_t iter)
{
    double w = d1 > d2 ? d1 : d2;
    if (std::isfinite(w) && w > r.worst_dev) r.worst_dev = w;
    if (!std::isfinite(d1) || !std::isfinite(d2) || w > TOLERANCE)
    {
        avx2_record(r, norm, d1, d2, iter);
        return false;
    }
    r.iterations = iter;
    return true;
}

// AVX2 -256-bit SIMD (two quaternions simultaneously)

TARGET_AVX2
static TestResult run_avx2(PRNG& rng, double deadline)
{
    TestResult r;
    r.passed = true;
    for (uint64_t i = 1;; i++)
    {
        if (test_should_stop(i, deadline)) break;
        volatile float a = rng.randf(), b = rng.randf(), c = rng.randf(), d = rng.randf();
        volatile float e = rng.randf(), f = rng.randf(), g = rng.randf(), h = rng.randf();
        float          in[8] = {a, b, c, d, e, f, g, h};
        __m256         q = _mm256_loadu_ps(in), lens = avx2_quad_dots(q);
        float          l1 = 0.0F, l2 = 0.0F;
        avx2_extract_pair(lens, &l1, &l2);
        if (l1 < 1e-10F || l2 < 1e-10F) continue;
        __m256 norm = _mm256_div_ps(q, _mm256_sqrt_ps(lens));
        float  c1 = 0.0F, c2 = 0.0F;
        avx2_extract_pair(avx2_quad_dots(norm), &c1, &c2);
        double d1 = fabs(static_cast<double>(c1) - 1.0);
        double d2 = fabs(static_cast<double>(c2) - 1.0);
        if (!avx2_check_devs(r, norm, d1, d2, i)) break;
    }
    _mm256_zeroupper();
    return r;
}

// FMA3 helpers

TARGET_FMA
static float fma3_check(__m128 n)
{
    float v_raw[4];
    _mm_storeu_ps(v_raw, n);
    volatile float v0 = v_raw[0], v1 = v_raw[1], v2 = v_raw[2], v3 = v_raw[3];
    __m128         a2 = _mm_set1_ps(static_cast<float>(v0)), b2 = _mm_set1_ps(static_cast<float>(v1));
    __m128         c2 = _mm_set1_ps(static_cast<float>(v2)), d2 = _mm_set1_ps(static_cast<float>(v3));
    __m128         r2 = _mm_mul_ss(a2, a2);
    r2                = _mm_fmadd_ss(b2, b2, r2);
    r2                = _mm_fmadd_ss(c2, c2, r2);
    r2                = _mm_fmadd_ss(d2, d2, r2);
    float o_raw       = 0.0F;
    _mm_store_ss(&o_raw, r2);
    volatile float o = o_raw;
    return o;
}

TARGET_FMA
static __m128 fma3_dot_ss(__m128 x, __m128 y, __m128 z, __m128 w)
{
    __m128 dot = _mm_mul_ss(x, x);
    dot        = _mm_fmadd_ss(y, y, dot);
    dot        = _mm_fmadd_ss(z, z, dot);
    return _mm_fmadd_ss(w, w, dot);
}

TARGET_FMA
static __m128 fma3_rsqrt_nr(__m128 dot)
{
    __m128 inv = _mm_rsqrt_ss(dot);
    __m128 nrf = _mm_fnmadd_ss(_mm_mul_ss(dot, _mm_set_ss(0.5F)), _mm_mul_ss(inv, inv), _mm_set_ss(1.5F));
    return _mm_mul_ss(inv, nrf);
}

TARGET_FMA
static __m128 fma3_len_sq(float x, float y, float z, float w)
{ return fma3_dot_ss(_mm_set1_ps(x), _mm_set1_ps(y), _mm_set1_ps(z), _mm_set1_ps(w)); }

TARGET_FMA
static void fma3_normalize_pair(__m128 q, __m128 dot, __m128* na, __m128* nb)
{
    __m128 inv = fma3_rsqrt_nr(dot);
    *na        = _mm_mul_ps(q, _mm_shuffle_ps(inv, inv, 0));
    __m128 len = _mm_sqrt_ss(dot);
    *nb        = _mm_div_ps(q, _mm_shuffle_ps(len, len, 0));
}

// FMA3 -fused multiply-add pipeline

TARGET_FMA
static TestResult run_fma3(PRNG& rng, double deadline)
{
    TestResult r;
    r.passed = true;
    for (uint64_t i = 1;; i++)
    {
        if (test_should_stop(i, deadline)) break;
        volatile float qx = rng.randf(), qy = rng.randf(), qz = rng.randf(), qw = rng.randf();
        __m128         q =
            _mm_set_ps(static_cast<float>(qw), static_cast<float>(qz), static_cast<float>(qy), static_cast<float>(qx));
        __m128 dot =
            fma3_len_sq(static_cast<float>(qx), static_cast<float>(qy), static_cast<float>(qz), static_cast<float>(qw));
        float ls_raw = 0.0F;
        _mm_store_ss(&ls_raw, dot);
        volatile float ls = ls_raw;
        if (static_cast<float>(ls) < 1e-10F) continue;
        __m128 na, nb;
        fma3_normalize_pair(q, dot, &na, &nb);
        double da = fabs(static_cast<double>(fma3_check(na)) - 1.0);
        double db = fabs(static_cast<double>(fma3_check(nb)) - 1.0);
        if (!check_pair_devs(r, na, nb, da, db, i)) break;
    }
    _mm256_zeroupper();
    return r;
}

// AVX-512F helpers - per-quaternion dots within each 128-bit quad

TARGET_AVX512
static VEC_INLINE __m512 avx512_quad_dots(__m512 v)
{
    __m512 sq    = _mm512_mul_ps(v, v);
    __m512 pairs = _mm512_add_ps(sq, _mm512_shuffle_ps(sq, sq, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm512_add_ps(pairs, _mm512_shuffle_ps(pairs, pairs, _MM_SHUFFLE(1, 0, 3, 2)));
}

TARGET_AVX512
static VEC_INLINE bool avx512_too_small(__m512 lens)
{
    float l[16];
    _mm512_storeu_ps(l, lens);
    return l[0] < 1e-10F || l[4] < 1e-10F || l[8] < 1e-10F || l[12] < 1e-10F;
}

TARGET_AVX512
static VEC_INLINE int avx512_check_devs(__m512 chk, double devs[4])
{
    float c[16];
    _mm512_storeu_ps(c, chk);
    int worst = 0;
    for (int k = 0; k < 4; k++)
    {
        volatile float ck = c[k * 4];
        devs[k]           = fabs(static_cast<double>(static_cast<float>(ck)) - 1.0);
        if (devs[k] > devs[worst]) worst = k;
    }
    return worst;
}

TARGET_AVX512
static VEC_INLINE void avx512_record(TestResult& r, __m512 norm, double dev, int quad, uint64_t iter)
{
    float tmp[16];
    _mm512_storeu_ps(tmp, norm);
    record_fail(r, dev, iter, tmp + quad * 4);
}

TARGET_AVX512
static VEC_INLINE bool avx512_check(TestResult& r, __m512 norm, uint64_t iter)
{
    double devs[4] = {};
    int    w       = avx512_check_devs(avx512_quad_dots(norm), devs);
    bool finite = std::isfinite(devs[0]) && std::isfinite(devs[1]) && std::isfinite(devs[2]) && std::isfinite(devs[3]);
    if (finite && devs[w] > r.worst_dev) r.worst_dev = devs[w];
    if (!finite || devs[w] > TOLERANCE)
    {
        avx512_record(r, norm, devs[w], w, iter);
        return false;
    }
    r.iterations = iter;
    return true;
}

// AVX512 -512-bit SIMD (four quaternions simultaneously)

TARGET_AVX512
static TestResult run_avx512(PRNG& rng, double deadline)
{
    TestResult r;
    r.passed = true;
    for (uint64_t i = 1;; i++)
    {
        if (test_should_stop(i, deadline)) break;
        volatile float vf[16];
        float          in[16];
        for (int k = 0; k < 16; k++)
        {
            vf[k] = rng.randf();
            in[k] = vf[k];
        }
        __m512 q = _mm512_loadu_ps(in), lens = avx512_quad_dots(q);
        if (avx512_too_small(lens)) continue;
        __m512 norm = _mm512_div_ps(q, _mm512_sqrt_ps(lens));
        if (!avx512_check(r, norm, i)) break;
    }
    _mm256_zeroupper();
    return r;
}

// XLANE - AVX2 cross-lane data integrity (bit-exact)

static bool xlane_verify(TestResult& r, const float* raw, const volatile float* expected, int which, uint64_t iter)
{
    for (int j = 0; j < 8; j++)
    {
        volatile float got = raw[j];
        if (std::bit_cast<uint32_t>(static_cast<float>(got)) ==
            std::bit_cast<uint32_t>(static_cast<float>(expected[j])))
            continue;
        float info[4] = {got, expected[j], static_cast<float>(j), static_cast<float>(which)};
        record_fail(r, 1.0, iter, info);
        r.iterations = iter * 3;
        return false;
    }
    return true;
}

// Each iteration performs 3 independent permutation checks (swap, reverse, rotate),
// so iterations are counted as i*3 to reflect actual cross-lane verifications.
TARGET_AVX2
static TestResult run_xlane(PRNG& rng, double deadline)
{
    TestResult r;
    r.passed = true;
    for (uint64_t i = 1;; i++)
    {
        if (test_should_stop(i, deadline)) break;
        volatile float f0 = rng.randf(), f1 = rng.randf(), f2 = rng.randf(), f3 = rng.randf();
        volatile float f4 = rng.randf(), f5 = rng.randf(), f6 = rng.randf(), f7 = rng.randf();
        float          in[8] = {f0, f1, f2, f3, f4, f5, f6, f7};
        __m256         src   = _mm256_loadu_ps(in);
        float          raw[8];
        _mm256_storeu_ps(raw, _mm256_permute2f128_ps(src, src, 0x01));
        volatile float exp_swap[8] = {f4, f5, f6, f7, f0, f1, f2, f3};
        if (!xlane_verify(r, raw, exp_swap, 0, i)) break;
        _mm256_storeu_ps(raw, _mm256_permutevar8x32_ps(src, _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7)));
        volatile float exp_rev[8] = {f7, f6, f5, f4, f3, f2, f1, f0};
        if (!xlane_verify(r, raw, exp_rev, 1, i)) break;
        _mm256_storeu_ps(raw, _mm256_permutevar8x32_ps(src, _mm256_set_epi32(2, 1, 0, 7, 6, 5, 4, 3)));
        volatile float exp_rot[8] = {f3, f4, f5, f6, f7, f0, f1, f2};
        if (!xlane_verify(r, raw, exp_rot, 2, i)) break;
        r.iterations = i * 3;
    }
    _mm256_zeroupper();
    return r;
}

// ============================================================================
// Test dispatch
// ============================================================================

using TestFunc                              = TestResult (*)(PRNG&, double);
static TestFunc const test_funcs[NUM_TESTS] = {run_scalar, run_sse, run_avx2, run_fma3, run_avx512, run_xlane};

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
    if (s == nullptr || *s == '\0') return false;
    errno     = 0;
    char* end = nullptr;
    long  v   = strtol(s, &end, 10);
    if (errno != 0 || end == s || (end != nullptr && *end != '\0')) return false;
    if (v < INT_MIN || v > INT_MAX) return false;
    *out = static_cast<int>(v);
    return true;
}

// Parse "lo-hi". The first char must not be '-' (that's a flag).
static bool parse_range(const char* s, int* lo, int* hi)
{
    const char* dash = strchr(s, '-');
    if (dash == nullptr || dash == s) return false;
    size_t lo_len = static_cast<size_t>(dash - s);
    char   buf[32];
    if (lo_len >= sizeof(buf)) return false;
    memcpy(buf, s, lo_len);
    buf[lo_len] = '\0';
    if (!parse_int(buf, lo)) return false;
    return parse_int(dash + 1, hi);
}

static const char* const flags_with_values[] = {"--socket", "--repeat", nullptr};

static bool is_flag_value(int idx, int /*argc*/, char** argv)
{
    if (idx <= 1) return false;
    const char* prev = argv[idx - 1];
    for (const char* const* f = flags_with_values; *f != nullptr; f++)
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

static void skip_flag_value(int* a, int argc, char** argv)
{
    for (const char* const* f = flags_with_values; *f != nullptr; f++)
    {
        if (strcmp(argv[*a], *f) == 0)
        {
            if (*a + 1 < argc) (*a)++;
            return;
        }
    }
}

static void add_thread_id(int t, std::vector<bool>& seen, int* out, int* count)
{
    if (seen[static_cast<size_t>(t)]) return;
    seen[static_cast<size_t>(t)] = true;
    out[(*count)++]              = t;
}

static bool parse_selector_range(const char* arg, int cap, std::vector<bool>& seen, int* out, int* count)
{
    int lo = 0, hi = 0;
    if (!parse_range(arg, &lo, &hi))
    {
        fprintf(stderr, COL_RED "  Error: invalid thread range: %s\n" COL_RESET, arg);
        return false;
    }
    if (lo > hi)
    {
        fprintf(stderr, COL_RED "  Error: inverted thread range: %s\n" COL_RESET, arg);
        return false;
    }
    if (lo < 0 || hi >= cap)
    {
        fprintf(stderr, COL_RED "  Error: thread range %s outside 0..%d\n" COL_RESET, arg, cap - 1);
        return false;
    }
    for (int t = lo; t <= hi && *count < MAX_THREADS; t++) add_thread_id(t, seen, out, count);
    return true;
}

static bool parse_selector_single(const char* arg, int cap, std::vector<bool>& seen, int* out, int* count)
{
    int t = 0;
    if (!parse_int(arg, &t))
    {
        fprintf(stderr, COL_RED "  Error: invalid thread id: %s\n" COL_RESET, arg);
        return false;
    }
    if (t < 0 || t >= cap)
    {
        fprintf(stderr, COL_RED "  Error: thread id %d outside 0..%d\n" COL_RESET, t, cap - 1);
        return false;
    }
    if (*count < MAX_THREADS) add_thread_id(t, seen, out, count);
    return true;
}

static int finish_thread_selection(bool selector_given, int count, int cap, int* out, bool* parse_error)
{
    if (!selector_given)
    {
        for (int i = 0; i < cap; i++) out[i] = i;
        return cap;
    }
    if (count == 0)
    {
        fprintf(stderr, COL_RED "  Error: no valid threads in selector\n" COL_RESET);
        if (parse_error != nullptr) *parse_error = true;
    }
    return count;
}

static int parse_threads(int argc, char** argv, int* out, int max_threads, bool* parse_error)
{
    int cap = max_threads < MAX_THREADS ? max_threads : MAX_THREADS;
    if (cap < 0) cap = 0;
    std::vector<bool> seen(static_cast<size_t>(cap), false);
    if (parse_error != nullptr) *parse_error = false;
    int  duration_idx   = find_first_positional(argc, argv);
    bool selector_given = false;
    int  count          = 0;
    for (int a = 1; a < argc && count < MAX_THREADS; a++)
    {
        if (a == duration_idx) continue;
        if (argv[a][0] == '-')
        {
            skip_flag_value(&a, argc, argv);
            continue;
        }
        if (is_flag_value(a, argc, argv)) continue;
        selector_given = true;
        bool ok        = strchr(argv[a], '-') != nullptr ? parse_selector_range(argv[a], cap, seen, out, &count)
                                                         : parse_selector_single(argv[a], cap, seen, out, &count);
        if (!ok)
        {
            if (parse_error != nullptr) *parse_error = true;
            return 0;
        }
    }
    return finish_thread_selection(selector_given, count, cap, out, parse_error);
}

// ============================================================================
// JSON output
// ============================================================================

struct CoreResult
{
    int        thread_id   = 0;
    bool       affinity_ok = false;
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
        } else if (c < 0x20)
        {
            if (j + 7 >= out_size) break;
            int n = snprintf(out + j, out_size - j, "\\u%04x", c);
            if (n < 0) break;
            j += static_cast<size_t>(n);
        } else
        {
            out[j++] = static_cast<char>(c);
        }
    }
    out[j < out_size ? j : out_size - 1] = '\0';
}

static void json_write_cpu(FILE* fp, const CPUFeatures& cpu)
{
    char brand_esc[128], vendor_esc[32];
    json_escape(cpu.brand, brand_esc, sizeof(brand_esc));
    json_escape(cpu.vendor, vendor_esc, sizeof(vendor_esc));
    fprintf(fp, "  \"cpu\": {\n");
    fprintf(fp, "    \"brand\": \"%s\",\n", brand_esc);
    fprintf(fp, "    \"vendor\": \"%s\",\n", vendor_esc);
    fprintf(fp, "    \"has_sse\": %s,\n", cpu.has_sse ? "true" : "false");
    fprintf(fp, "    \"has_sse3\": %s,\n", cpu.has_sse3 ? "true" : "false");
    fprintf(fp, "    \"has_avx2\": %s,\n", cpu.has_avx2 ? "true" : "false");
    fprintf(fp, "    \"has_fma3\": %s,\n", cpu.has_fma3 ? "true" : "false");
    fprintf(fp, "    \"has_avx512f\": %s,\n", cpu.has_avx512f ? "true" : "false");
    fprintf(fp, "    \"os_avx_enabled\": %s,\n", cpu.os_avx_enabled ? "true" : "false");
    fprintf(fp, "    \"os_avx512_enabled\": %s\n", cpu.os_avx512_enabled ? "true" : "false");
    fprintf(fp, "  },\n");
}

static void json_write_fail_detail(FILE* fp, const TestResult* tr)
{
    fprintf(fp, "          \"fail_iteration\": %llu,\n", static_cast<unsigned long long>(tr->worst_iter));
    if (std::isfinite(tr->worst_dev))
    {
        fprintf(fp, "          \"deviation\": %.10f,\n", tr->worst_dev);
    } else
    {
        fprintf(fp, "          \"deviation\": null,\n");
    }
    bool q_finite = std::isfinite(tr->worst_q[0]) && std::isfinite(tr->worst_q[1]) && std::isfinite(tr->worst_q[2]) &&
                    std::isfinite(tr->worst_q[3]);
    if (q_finite)
    {
        fprintf(fp,
                "          \"quaternion\": [%.8f, %.8f, %.8f, %.8f],\n",
                static_cast<double>(tr->worst_q[0]),
                static_cast<double>(tr->worst_q[1]),
                static_cast<double>(tr->worst_q[2]),
                static_cast<double>(tr->worst_q[3]));
    } else
    {
        fprintf(fp, "          \"quaternion\": null,\n");
    }
    fprintf(fp, "          \"confirmed\": %s,\n", tr->confirmed ? "true" : "false");
    fprintf(fp, "          \"rerun_fails\": %d\n", tr->rerun_fails);
}

static void json_write_test(FILE* fp, const TestResult* tr, int t, bool last)
{
    fprintf(fp, "        \"%s\": {\n", tname[t]);
    if (tr->skipped)
    {
        fprintf(fp, "          \"status\": \"skipped\"\n");
    } else if (tr->passed)
    {
        fprintf(fp, "          \"status\": \"pass\",\n");
        fprintf(fp, "          \"iterations\": %llu\n", static_cast<unsigned long long>(tr->iterations));
    } else
    {
        fprintf(fp, "          \"status\": \"FAIL\",\n");
        fprintf(fp, "          \"iterations\": %llu,\n", static_cast<unsigned long long>(tr->iterations));
        json_write_fail_detail(fp, tr);
    }
    fprintf(fp, "        }%s\n", last ? "" : ",");
}

static void json_write_core(FILE* fp, const CoreResult* cr, const TopologyInfo& topo, bool last)
{
    int phys = 0, pkg = 0;
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
    for (int t = 0; t < NUM_TESTS; t++) json_write_test(fp, &cr->tests[t], t, t == NUM_TESTS - 1);
    fprintf(fp, "      }\n");
    fprintf(fp, "    }%s\n", last ? "" : ",");
}

static bool json_commit(FILE* fp, const char* tmp_path, const char* path)
{
    fflush(fp);
    bool ok = ferror(fp) == 0;
    if (fclose(fp) != 0) ok = false;
    if (ok)
    {
        remove(path);
        if (rename(tmp_path, path) != 0) ok = false;
    } else
    {
        remove(tmp_path);
    }
    return ok;
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
    if (fp == nullptr)
    {
        fprintf(stderr, "  Warning: could not write %s\n", path);
        return false;
    }
    fprintf(fp, "{\n");
    fprintf(fp, "  \"version\": \"%s\",\n", COREPROBE_VERSION);
    json_write_cpu(fp, cpu);
    fprintf(fp, "  \"topology_valid\": %s,\n", topo.valid ? "true" : "false");
    fprintf(fp, "  \"physical_cores\": %d,\n", topo.core_count);
    fprintf(fp, "  \"tolerance\": %.6f,\n", TOLERANCE);
    fprintf(fp, "  \"wall_time_sec\": %.2f,\n", wall_time);
    fprintf(fp, "  \"results\": [\n");
    for (int i = 0; i < num_results; i++) json_write_core(fp, &all[i], topo, i == num_results - 1);
    fprintf(fp, "  ]\n");
    fprintf(fp, "}\n");
    return json_commit(fp, tmp_path, path);
}

// ============================================================================
// Help
// ============================================================================

static void print_help_usage(const char* argv0)
{
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
}

static void print_help_flags()
{
    printf("  Flags:\n");
    printf("    --soak              extended 10-minute soak test (recommended for\n");
    printf("                        intermittent faults or final stability validation)\n");
    printf("    --socket N          test only threads on socket/package N (multi-socket)\n");
    printf("    --repeat N          run N full passes (default: 1)\n");
    printf("    --until-fail        repeat indefinitely until a failure is detected\n");
    printf("    --json              write results to coreprobe_results.json\n");
    printf("    --pause             wait for Enter before exiting (for double-click)\n");
    printf("    --version           print version and exit\n");
    printf("    --help              show this help\n\n");
}

static void print_help(const char* argv0)
{
    printf("\n");
    printf("  coreprobe v%s - Per-Core FPU/SIMD Correctness Diagnostic\n\n", COREPROBE_VERSION);
    printf("  Stress-tests quaternion normalization across SCALAR, SSE3, AVX2, FMA3,\n");
    printf("  and AVX-512F plus cross-lane AVX2 data integrity (XLANE) on each logical\n");
    printf("  processor to detect faulty floating-point and SIMD execution units.\n");
    printf("  Catches defects that memtest86+, Prime95, and WHEA reporting miss.\n\n");
    print_help_usage(argv0);
    print_help_flags();
    printf("  This is an arithmetic correctness test, not a throughput stress test.\n");
    printf("  -O0 and volatile barriers ensure every FP op executes through hardware.\n");
    printf("  Sequential per-core testing gives clean fault attribution.\n");
    printf("  Ctrl+C stops gracefully after the current check; press twice to abort.\n\n");
    printf("  Exit code: 0 = all pass, 1 = failures detected, 2 = usage error,\n");
    printf("             130 = interrupted\n\n");
}

// ============================================================================
// Core map
// ============================================================================

static bool core_result_ok(const CoreResult& cr)
{
    for (const TestResult& tr : cr.tests)
    {
        if (!tr.passed && !tr.skipped) return false;
    }
    return true;
}

// Status: 0 = untested, 1 = pass, 2 = fail. Worst status of any thread wins.
static std::vector<int> compute_core_status(const CoreResult* all, int num_results, int cap, const TopologyInfo& topo)
{
    int              core_count = topo.core_count > 0 ? topo.core_count : 0;
    std::vector<int> core_status(static_cast<size_t>(core_count), 0);
    for (int i = 0; i < num_results; i++)
    {
        if (!all[i].affinity_ok) continue;
        int tid = all[i].thread_id;
        if (tid < 0 || tid >= cap) continue;
        int status = core_result_ok(all[i]) ? 1 : 2;
        int phys   = topo.physical_core[static_cast<size_t>(tid)];
        if (phys < 0 || phys >= core_count) continue;
        if (status > core_status[static_cast<size_t>(phys)]) core_status[static_cast<size_t>(phys)] = status;
    }
    return core_status;
}

static void print_core_map(const CoreResult* all, int num_results, int max_threads, const TopologyInfo& topo)
{
    int cap = max_threads < MAX_THREADS ? max_threads : MAX_THREADS;
    if (cap < 0) cap = 0;
    std::vector<int> core_status = compute_core_status(all, num_results, cap, topo);
    printf("  Core Map (%d physical cores%s):\n\n  ", topo.core_count, topo.valid ? ", OS topology" : ", heuristic");
    for (int phys = 0; phys < topo.core_count; phys++)
    {
        int status = phys < static_cast<int>(core_status.size()) ? core_status[static_cast<size_t>(phys)] : 0;
        if (status == 2) printf(COL_RED);
        else if (status == 1) printf(COL_GREEN);
        else printf(COL_GRAY);
        printf("[%2d]", phys);
        printf(COL_RESET);
        if ((phys + 1) % 8 == 0 && phys + 1 < topo.core_count) printf("  |  ");
        else printf(" ");
    }
    printf("\n\n  ");
    printf(COL_GREEN "[OK]" COL_RESET " = pass  ");
    printf(COL_RED "[XX]" COL_RESET " = FAIL  ");
    printf(COL_GRAY "[--]" COL_RESET " = not tested\n");
}

// ============================================================================
// Run configuration
// ============================================================================

struct RunConfig
{
    bool             json_output     = false;
    bool             soak_mode       = false;
    bool             pause_at_end    = false;
    bool             until_fail      = false;
    bool             high_priority   = false;
    int              socket_filter   = -1;
    int              repeat_count    = 1;
    int              total_seconds   = 120;
    int              num_threads     = 0;
    int              max_threads     = 0;
    double           secs_per_test   = 0;
    double           secs_per_thread = 0;
    int              actual_total    = 0;
    std::vector<int> thread_list;
};

static bool parse_bool_flag(const char* arg, RunConfig& cfg)
{
    if (strcmp(arg, "--json") == 0)
    {
        cfg.json_output = true;
    } else if (strcmp(arg, "--soak") == 0)
    {
        cfg.soak_mode = true;
    } else if (strcmp(arg, "--pause") == 0)
    {
        cfg.pause_at_end = true;
    } else if (strcmp(arg, "--until-fail") == 0)
    {
        cfg.until_fail = true;
    } else
    {
        return false;
    }
    return true;
}

static int parse_value_flag(int* i, int argc, char** argv, RunConfig& cfg)
{
    if (strcmp(argv[*i], "--socket") == 0)
    {
        if (*i + 1 >= argc || !parse_int(argv[*i + 1], &cfg.socket_filter))
        {
            fprintf(stderr, COL_RED "  Error: --socket requires an integer\n" COL_RESET);
            return 2;
        }
        if (cfg.socket_filter < 0)
        {
            fprintf(stderr, COL_RED "  Error: --socket must be >= 0\n" COL_RESET);
            return 2;
        }
        (*i)++;
        return 0;
    }
    if (strcmp(argv[*i], "--repeat") == 0)
    {
        if (*i + 1 >= argc || !parse_int(argv[*i + 1], &cfg.repeat_count))
        {
            fprintf(stderr, COL_RED "  Error: --repeat requires an integer\n" COL_RESET);
            return 2;
        }
        (*i)++;
        return 0;
    }
    return -1;
}

static int scan_flags(int argc, char** argv, RunConfig& cfg)
{
    for (int i = 1; i < argc; i++)
    {
        const char* arg = argv[i];
        if (parse_bool_flag(arg, cfg)) continue;
        int rc = parse_value_flag(&i, argc, argv, cfg);
        if (rc >= 0)
        {
            if (rc != 0) return rc;
            continue;
        }
        if (arg[0] == '-')
        {
            fprintf(stderr, COL_RED "  Error: unknown flag: %s (try --help)\n" COL_RESET, arg);
            return 2;
        }
    }
    return -1;
}

static int parse_duration(int argc, char** argv, RunConfig& cfg)
{
    cfg.total_seconds = cfg.soak_mode ? 600 : 120;
    int duration_idx  = find_first_positional(argc, argv);
    if (duration_idx < 0) return -1;
    int parsed = 0;
    if (!parse_int(argv[duration_idx], &parsed) || parsed <= 0)
    {
        fprintf(stderr, COL_RED "  Error: invalid duration: %s\n" COL_RESET, argv[duration_idx]);
        return 2;
    }
    cfg.total_seconds = parsed;
    return -1;
}

static int apply_socket_filter(RunConfig& cfg, const TopologyInfo& topo)
{
    if (cfg.socket_filter < 0) return -1;
    if (!topo.valid)
    {
        printf(COL_YELLOW "  Warning: --socket %d requested but topology detection failed, "
                          "testing all threads\n" COL_RESET,
               cfg.socket_filter);
        return -1;
    }
    std::vector<int> filtered;
    filtered.reserve(static_cast<size_t>(cfg.num_threads));
    for (int i = 0; i < cfg.num_threads; i++)
    {
        int tid = cfg.thread_list[static_cast<size_t>(i)];
        if (topo.package_id[static_cast<size_t>(tid)] == cfg.socket_filter) filtered.push_back(tid);
    }
    if (filtered.empty())
    {
        fprintf(stderr, COL_RED "  Error: no threads found for socket %d\n" COL_RESET, cfg.socket_filter);
        return 2;
    }
    memcpy(cfg.thread_list.data(), filtered.data(), filtered.size() * sizeof(int));
    cfg.num_threads = static_cast<int>(filtered.size());
    return -1;
}

static void compute_schedule(RunConfig& cfg)
{
    cfg.secs_per_thread = static_cast<double>(cfg.total_seconds) / cfg.num_threads;
    cfg.secs_per_test   = cfg.secs_per_thread / static_cast<double>(NUM_TESTS);
    if (cfg.secs_per_test < 2.0) cfg.secs_per_test = 2.0;
    cfg.actual_total = static_cast<int>(lround(cfg.secs_per_test * static_cast<double>(NUM_TESTS) * cfg.num_threads));
}

static int handle_info_flags(int argc, char** argv)
{
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0)
        {
            print_help(argv[0]);
            return 0;
        }
        if (strcmp(argv[i], "--version") == 0)
        {
            printf("coreprobe %s\n", COREPROBE_VERSION);
            return 0;
        }
    }
    return -1;
}

// Returns: -1 = success (continue), otherwise the process exit code.
static int parse_args(int argc, char** argv, RunConfig& cfg, const TopologyInfo& topo)
{
    int rc = handle_info_flags(argc, argv);
    if (rc >= 0) return rc;
    rc = scan_flags(argc, argv, cfg);
    if (rc >= 0) return rc;
    if (cfg.repeat_count < 1) cfg.repeat_count = 1;
    if (cfg.until_fail) cfg.repeat_count = INT32_MAX;
    rc = parse_duration(argc, argv, cfg);
    if (rc >= 0) return rc;
    cfg.thread_list.resize(static_cast<size_t>(cfg.max_threads));
    bool thread_parse_error = false;
    cfg.num_threads         = parse_threads(argc, argv, cfg.thread_list.data(), cfg.max_threads, &thread_parse_error);
    if (thread_parse_error) return 2;
    rc = apply_socket_filter(cfg, topo);
    if (rc >= 0) return rc;
    if (cfg.num_threads == 0)
    {
        fprintf(stderr, COL_RED "  Error: no threads to test\n" COL_RESET);
        return 2;
    }
    compute_schedule(cfg);
    return -1;
}

// ============================================================================
// Header
// ============================================================================

static void print_banner()
{
    printf(COL_CYAN "\n");
    printf("  +================================================================+\n");
    printf("  |       coreprobe v%-6s - FPU/SIMD Correctness Diagnostic    |\n", COREPROBE_VERSION);
    printf("  +================================================================+\n\n" COL_RESET);
}

static void print_cpu_summary(const RunConfig& cfg, const CPUFeatures& cpu, const TopologyInfo& topo)
{
    printf("  CPU:                 %s\n", cpu.brand);
    printf("  Vendor:              %s\n", cpu.vendor);
    printf("  Logical processors:  %d\n", cfg.max_threads);
    printf("  Physical cores:      %d%s\n",
           topo.core_count,
           topo.valid ? " (OS topology)" : " (heuristic — 1 thread per core)");
    printf("  Instruction sets:    SSE3=%s  AVX2=%s  FMA3=%s  AVX-512F=%s\n",
           cpu.has_sse3 ? COL_GREEN "yes" COL_RESET : COL_RED "no" COL_RESET,
           cpu.has_avx2 ? COL_GREEN "yes" COL_RESET : COL_RED "no" COL_RESET,
           cpu.has_fma3 ? COL_GREEN "yes" COL_RESET : COL_RED "no" COL_RESET,
           cpu.has_avx512f ? COL_GREEN "yes" COL_RESET : COL_GRAY "no" COL_RESET);
    printf("  OS AVX state:        %s\n",
           cpu.os_avx_enabled ? COL_GREEN "enabled (XSAVE/XGETBV)" COL_RESET
                              : COL_YELLOW "disabled - AVX/FMA/XLANE tests will be skipped" COL_RESET);
}

static void print_thread_selection(const RunConfig& cfg)
{
    printf("  Testing threads:     ");
    if (cfg.socket_filter >= 0)
    {
        printf("%d (socket %d only)\n", cfg.num_threads, cfg.socket_filter);
    } else if (cfg.num_threads == cfg.max_threads)
    {
        printf("ALL (%d)\n", cfg.num_threads);
    } else
    {
        for (int i = 0; i < cfg.num_threads; i++)
            printf("%d%s", cfg.thread_list[static_cast<size_t>(i)], i < cfg.num_threads - 1 ? ", " : "\n");
    }
}

static void print_run_summary(const RunConfig& cfg)
{
    print_thread_selection(cfg);
    printf("  Mode:                %s\n", cfg.soak_mode ? "SOAK (extended)" : "standard");
    if (cfg.actual_total != cfg.total_seconds)
    {
        printf("  Duration:            ~%ds total (requested %ds, %.1fs/test, min 2s/test)\n",
               cfg.actual_total,
               cfg.total_seconds,
               cfg.secs_per_test);
    } else
    {
        printf("  Duration:            ~%ds total (%.1fs/thread, %.1fs/test)\n",
               cfg.total_seconds,
               cfg.secs_per_thread,
               cfg.secs_per_test);
    }
    printf("  Tolerance:           %.6f\n", TOLERANCE);
    printf("  Rerun on fail:       %dx (deterministic seed replay)\n", RERUN_COUNT);
    printf("  Compile flags:       -O0, volatile floats (correctness test, not throughput)\n");
    if (cfg.high_priority)
    {
        printf("  Priority:            " COL_GREEN "HIGH" COL_RESET "\n");
    } else
    {
        printf("  Priority:            " COL_YELLOW "normal (needs elevated privileges)" COL_RESET "\n");
    }
}

static void print_feature_warnings(const CPUFeatures& cpu)
{
    if (!cpu.os_avx_enabled)
    {
        printf("  " COL_YELLOW "Warning: OS has not enabled AVX state (XGETBV XCR0 bits 1:2)." COL_RESET "\n");
        printf("  " COL_YELLOW "  AVX2 and FMA3 tests will be skipped. This is unusual on" COL_RESET "\n");
        printf("  " COL_YELLOW "  modern systems -check BIOS settings or OS configuration." COL_RESET "\n");
    } else
    {
        if (!cpu.has_avx2)
            printf("  " COL_YELLOW "Note: AVX2 not supported by CPU, test will be skipped" COL_RESET "\n");
        if (!cpu.has_fma3)
            printf("  " COL_YELLOW "Note: FMA3 not supported by CPU, test will be skipped" COL_RESET "\n");
        if (cpu.cpu_avx512f && !cpu.os_avx512_enabled)
            printf("  " COL_YELLOW "Warning: CPU supports AVX-512F but the OS has not enabled\n"
                   "  AVX-512 state (XCR0 bits 5:7) - AVX512 test will be skipped." COL_RESET "\n");
    }
    if (!cpu.has_sse3) printf("  " COL_YELLOW "Note: SSE3 not supported by CPU, test will be skipped" COL_RESET "\n");
    printf("\n");
}

static void print_header(const RunConfig& cfg, const CPUFeatures& cpu, const TopologyInfo& topo)
{
    print_banner();
    print_cpu_summary(cfg, cpu, topo);
    print_run_summary(cfg);
    print_feature_warnings(cpu);
}

static void print_table_header()
{
    printf(COL_GRAY "  %-5s %-13s %-13s %-13s %-13s %-13s %-13s %s" COL_RESET "\n",
           "THR",
           "SCALAR",
           "SSE3",
           "AVX2",
           "FMA3",
           "AVX512",
           "XLANE",
           "STATUS");
    printf(COL_GRAY "  %-5s %-13s %-13s %-13s %-13s %-13s %-13s %s" COL_RESET "\n",
           "---",
           "--------",
           "--------",
           "--------",
           "--------",
           "--------",
           "--------",
           "------");
}

// ============================================================================
// Failure report
// ============================================================================

static int count_confirmed_fails(const std::vector<CoreResult>& all, const std::vector<int>& fail_indices)
{
    int confirmed = 0;
    for (int fi : fail_indices)
    {
        for (const TestResult& tr : all[static_cast<size_t>(fi)].tests)
        {
            if (!tr.passed && !tr.skipped && tr.confirmed) confirmed++;
        }
    }
    return confirmed;
}

static void print_one_test_failure(const TestResult* tr, int t)
{
    if (tr->skipped) return;
    if (tr->passed)
    {
        printf("    " COL_GREEN "%s OK" COL_RESET "\n", tname[t]);
        return;
    }
    printf("    " COL_RED "%s FAILED" COL_RESET, tname[t]);
    printf(" at iter %llu  deviation=%.10f", static_cast<unsigned long long>(tr->worst_iter), tr->worst_dev);
    if (tr->confirmed)
    {
        printf("  " COL_RED "[confirmed %d/%d reruns]" COL_RESET, tr->rerun_fails, RERUN_COUNT);
    } else
    {
        printf("  " COL_YELLOW "[transient, 0/%d reruns]" COL_RESET, RERUN_COUNT);
    }
    printf("\n");
    printf("      quat(%.8f, %.8f, %.8f, %.8f)\n",
           static_cast<double>(tr->worst_q[0]),
           static_cast<double>(tr->worst_q[1]),
           static_cast<double>(tr->worst_q[2]),
           static_cast<double>(tr->worst_q[3]));
}

static void print_failed_thread(const CoreResult* cr, const TopologyInfo& topo)
{
    int tid  = cr->thread_id;
    int phys = (tid >= 0 && tid < static_cast<int>(topo.physical_core.size()))
                   ? topo.physical_core[static_cast<size_t>(tid)]
                   : -1;
    int pkg =
        (tid >= 0 && tid < static_cast<int>(topo.package_id.size())) ? topo.package_id[static_cast<size_t>(tid)] : -1;
    printf("  Thread %d (physical core %d, package %d%s):\n", tid, phys, pkg, topo.valid ? "" : ", heuristic");
    for (int t = 0; t < NUM_TESTS; t++) print_one_test_failure(&cr->tests[t], t);
    printf("\n");
}

static void
print_affected_cores(const std::vector<CoreResult>& all, const std::vector<int>& fail_indices, const TopologyInfo& topo)
{
    printf("  " COL_MAGENTA "Diagnosis:" COL_RESET "\n");
    printf("  Affected physical core(s): ");
    std::vector<bool> seen_core(static_cast<size_t>(topo.core_count > 0 ? topo.core_count : 0), false);
    for (int fi : fail_indices)
    {
        int tid = all[static_cast<size_t>(fi)].thread_id;
        if (tid < 0 || tid >= static_cast<int>(topo.physical_core.size())) continue;
        int phys = topo.physical_core[static_cast<size_t>(tid)];
        if (phys >= 0 && phys < static_cast<int>(seen_core.size()) && !seen_core[static_cast<size_t>(phys)])
        {
            printf(COL_RED "Core %d " COL_RESET, phys);
            seen_core[static_cast<size_t>(phys)] = true;
        }
    }
    printf("\n");
}

static bool scalar_ok_simd_fail(const CoreResult& cr)
{
    if (!cr.tests[T_SCALAR].passed) return false;
    for (int t = T_SSE3; t < NUM_TESTS; t++)
    {
        if (!cr.tests[t].passed && !cr.tests[t].skipped) return true;
    }
    return false;
}

static bool xlane_only_fail(const CoreResult& cr)
{
    if (cr.tests[T_XLANE].skipped || cr.tests[T_XLANE].passed) return false;
    for (int t = 0; t < NUM_TESTS; t++)
    {
        if (t == T_XLANE) continue;
        if (!cr.tests[t].passed && !cr.tests[t].skipped) return false;
    }
    return true;
}

static void print_pattern_hints(bool simd_only, bool xlane_only, int confirmed_fails)
{
    if (simd_only)
    {
        printf("\n  " COL_YELLOW "Pattern: SCALAR passes but SIMD fails" COL_RESET "\n");
        printf("  This indicates SIMD execution units (SSE/AVX/FMA/lane-crossing)\n");
        printf("  are faulty while the scalar FP pipeline is intact. Common causes:\n");
        printf("    - Silicon defect in SIMD execution unit on affected core\n");
        printf("    - Degraded CPU (age, heat damage, electromigration)\n");
        printf("    - If on OC/PBO: reduce clocks or increase voltage\n");
        printf("    - If on stock: CPU hardware fault, consider RMA or replacement\n");
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

static void
print_fail_report(const std::vector<CoreResult>& all, const std::vector<int>& fail_indices, const TopologyInfo& topo)
{
    printf(COL_RED "  *** FPU ERRORS DETECTED ***" COL_RESET "\n\n");
    bool simd_only  = false;
    bool xlane_only = false;
    for (int fi : fail_indices)
    {
        const CoreResult* cr = &all[static_cast<size_t>(fi)];
        print_failed_thread(cr, topo);
        if (scalar_ok_simd_fail(*cr)) simd_only = true;
        if (xlane_only_fail(*cr)) xlane_only = true;
    }
    print_affected_cores(all, fail_indices, topo);
    print_pattern_hints(simd_only, xlane_only, count_confirmed_fails(all, fail_indices));
}

// ============================================================================
// Test execution
// ============================================================================

static bool should_skip_test(int t, const CPUFeatures& cpu)
{
    if (t == T_SSE3 && !cpu.has_sse3) return true;
    if (t == T_AVX2 && !cpu.has_avx2) return true;
    if (t == T_FMA3 && !cpu.has_fma3) return true;
    if (t == T_AVX512 && !cpu.has_avx512f) return true;
    if (t == T_XLANE && !cpu.has_avx2) return true;
    return false;
}

static void print_test_pass(const TestResult& tr, int t)
{
    char   buf[32];
    double wd = tr.worst_dev;
    if (t != T_XLANE && wd > WARN_THRESHOLD)
    {
        snprintf(buf, sizeof(buf), "WARN %.3f%%", wd * 100.0);
        printf(COL_YELLOW "%-13s" COL_RESET, buf);
    } else
    {
        snprintf(buf, sizeof(buf), "PASS %lluM", static_cast<unsigned long long>(tr.iterations / 1'000'000ULL));
        printf(COL_GREEN "%-13s" COL_RESET, buf);
    }
}

static void confirm_and_print_fail(TestResult& tr, int t, uint32_t seed, double secs_per_test)
{
    tr.rerun_fails   = 0;
    double rerun_dur = secs_per_test > RERUN_DURATION ? secs_per_test : RERUN_DURATION;
    for (int rr = 0; rr < RERUN_COUNT && g_stop == 0; rr++)
    {
        TestResult rerun = rerun_single(test_funcs[t], seed, rerun_dur);
        if (!rerun.passed) tr.rerun_fails++;
    }
    tr.confirmed = tr.rerun_fails > 0;
    char buf[32];
    if (t == T_XLANE)
    {
        snprintf(buf, sizeof(buf), "FAIL mismatch");
    } else
    {
        snprintf(buf, sizeof(buf), "FAIL %.3f%%", tr.worst_dev * 100.0);
    }
    if (tr.confirmed)
    {
        printf(COL_RED "%-13s" COL_RESET, buf);
    } else
    {
        printf(COL_YELLOW "%-13s" COL_RESET, buf);
    }
}

static bool run_one_test(TestResult& tr, int t, uint32_t seed, double secs_per_test)
{
    PRNG rng;
    rng.seed(seed);
    tr           = test_funcs[t](rng, now_sec() + secs_per_test);
    tr.fail_seed = seed;
    if (tr.passed)
    {
        print_test_pass(tr, t);
        return true;
    }
    confirm_and_print_fail(tr, t, seed, secs_per_test);
    return false;
}

static void mark_skipped(TestResult& tr, bool print_cell)
{
    tr.skipped = true;
    tr.passed  = true;
    if (print_cell)
    {
        printf(COL_GRAY "%-13s" COL_RESET, "skip");
        fflush(stdout);
    }
}

static bool run_thread_tests(CoreResult& cr, int tid, int pass, const CPUFeatures& cpu, const RunConfig& cfg)
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
    for (int t = 0; t < NUM_TESTS; t++)
    {
        if (g_stop != 0 || should_skip_test(t, cpu))
        {
            mark_skipped(cr.tests[t], g_stop == 0);
            continue;
        }
        uint32_t seed = 0xF0'0D'00'00U + (static_cast<uint32_t>(pass) << 20) + static_cast<uint32_t>(tid) * NUM_TESTS +
                        static_cast<uint32_t>(t);
        if (!run_one_test(cr.tests[t], t, seed, cfg.secs_per_test)) core_ok = false;
        fflush(stdout);
    }
    if (g_stop != 0) printf(COL_YELLOW " interrupted" COL_RESET "\n");
    else if (core_ok) printf(COL_GREEN " OK" COL_RESET "\n");
    else printf(COL_RED " ** FAIL **" COL_RESET "\n");
    return true;
}

static int tally_failures(const std::vector<CoreResult>& all, std::vector<int>& fail_indices)
{
    int total = 0;
    for (size_t ci = 0; ci < all.size(); ci++)
    {
        if (!all[ci].affinity_ok) continue;
        bool has_fail = false;
        for (const TestResult& tr : all[ci].tests)
        {
            if (!tr.passed && !tr.skipped)
            {
                total++;
                has_fail = true;
            }
        }
        if (has_fail) fail_indices.push_back(static_cast<int>(ci));
    }
    return total;
}

static void print_verdict(const std::vector<CoreResult>& all,
                          const std::vector<int>&        fail_indices,
                          int                            total_fails,
                          int                            affinity_fails,
                          const RunConfig&               cfg,
                          const TopologyInfo&            topo)
{
    if (total_fails > 0)
    {
        print_fail_report(all, fail_indices, topo);
        return;
    }
    if (affinity_fails == cfg.num_threads)
    {
        printf(COL_RED "  *** No tests ran: affinity failed on every requested thread ***" COL_RESET "\n");
        return;
    }
    printf(COL_GREEN "  ALL TESTS PASSED -no FPU/SIMD errors detected." COL_RESET "\n");
    if (affinity_fails > 0)
        printf(COL_YELLOW "  Note: %d thread(s) skipped due to affinity failures." COL_RESET "\n", affinity_fails);
}

static int print_pass_summary(const std::vector<CoreResult>& all,
                              int                            affinity_fails,
                              const RunConfig&               cfg,
                              const CPUFeatures&             cpu,
                              const TopologyInfo&            topo,
                              double                         wall_secs)
{
    printf("\n");
    printf(COL_CYAN "  +================================================================+\n");
    printf("  |                        SUMMARY                                 |\n");
    printf("  +================================================================+\n" COL_RESET);
    printf("\n");
    std::vector<int> fail_indices;
    int              total_fails = tally_failures(all, fail_indices);
    print_verdict(all, fail_indices, total_fails, affinity_fails, cfg, topo);
    printf("\n");
    print_core_map(all.data(), cfg.num_threads, cfg.max_threads, topo);
    printf("\n  Wall time: %.1f seconds\n", wall_secs);
    if (cfg.json_output)
    {
        if (write_json("coreprobe_results.json", cpu, topo, all.data(), cfg.num_threads, wall_secs))
        {
            printf("  Results written to: coreprobe_results.json\n");
        } else
        {
            printf(COL_YELLOW "  Warning: JSON write encountered errors" COL_RESET "\n");
        }
    }
    printf("\n");
    return total_fails;
}

// ============================================================================
// Main
// ============================================================================

static int run_single_pass(int pass, const RunConfig& cfg, const CPUFeatures& cpu, const TopologyInfo& topo)
{
    std::vector<CoreResult> all(static_cast<size_t>(cfg.num_threads));
    int                     affinity_fails = 0;
    double                  wall_start     = now_sec();
    for (int ci = 0; ci < cfg.num_threads && g_stop == 0; ci++)
    {
        int tid                                = cfg.thread_list[static_cast<size_t>(ci)];
        all[static_cast<size_t>(ci)].thread_id = tid;
        if (!run_thread_tests(all[static_cast<size_t>(ci)], tid, pass, cpu, cfg)) affinity_fails++;
    }
    int fails = print_pass_summary(all, affinity_fails, cfg, cpu, topo, now_sec() - wall_start);
    if (affinity_fails == cfg.num_threads && fails == 0) fails = 1;
    return fails;
}

static int run_passes(const RunConfig& cfg, const CPUFeatures& cpu, const TopologyInfo& topo)
{
    int overall_fails = 0;
    int pass_number   = 0;
    for (int pass = 0; pass < cfg.repeat_count && g_stop == 0; pass++)
    {
        pass_number = pass + 1;
        if (cfg.repeat_count > 1)
        {
            printf(COL_CYAN "\n  === Pass %d%s ===" COL_RESET "\n\n",
                   pass_number,
                   cfg.until_fail ? " (until-fail mode)" : "");
            print_table_header();
        }
        int fails      = run_single_pass(pass, cfg, cpu, topo);
        overall_fails += fails;
        if (fails > 0) break;
        if (cfg.repeat_count > 1 && pass + 1 < cfg.repeat_count)
            printf("  Pass %d complete -no errors. Continuing...\n", pass_number);
    }
    if (g_stop != 0)
    {
        printf(COL_YELLOW "\n  Interrupted by user." COL_RESET "\n");
    } else if (cfg.repeat_count > 1 && overall_fails == 0)
    {
        printf(COL_GREEN "\n  All %d passes completed with no failures." COL_RESET "\n", pass_number);
    }
    return overall_fails;
}

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
    RunConfig    cfg;
    cfg.max_threads = max_threads;
    int parse_rc    = parse_args(argc, argv, cfg, topo);
    if (parse_rc >= 0) return parse_rc;
    cfg.high_priority = platform_set_high_priority();
    print_header(cfg, cpu, topo);
    print_table_header();
    int overall_fails = run_passes(cfg, cpu, topo);
    int exit_code     = overall_fails > 0 ? 1 : (g_stop != 0 ? 130 : 0);
    if (cfg.pause_at_end && g_stop == 0)
    {
        g_exit_code    = exit_code;
        g_pause_prompt = 1;
        printf("  Press Enter to exit...");
        fflush(stdout);
        getchar();
    }
    return exit_code;
}
