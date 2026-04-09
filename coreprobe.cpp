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

#define COREPROBE_VERSION "1.0.1"
#define MAX_THREADS 4096

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <immintrin.h>

// Optimization guards - prevent optimization even under -O2/-O3/PGO

#ifdef _MSC_VER
    #define COREPROBE_NO_OPTIMIZE  __pragma(optimize("", off))
    #define COREPROBE_RESTORE_OPT  __pragma(optimize("", on))
#elif defined(__clang__) || defined(__GNUC__)
    #define COREPROBE_NO_OPTIMIZE  _Pragma("GCC push_options") \
                                  _Pragma("GCC optimize(\"O0\")")
    #define COREPROBE_RESTORE_OPT  _Pragma("GCC pop_options")
#else
    #define COREPROBE_NO_OPTIMIZE
    #define COREPROBE_RESTORE_OPT
#endif

// Per-function ISA target attributes (GCC/Clang; MSVC uses global /arch:)
#if defined(__GNUC__) || defined(__clang__)
    #define TARGET_SSE3     __attribute__((target("sse3")))
    #define TARGET_AVX2     __attribute__((target("avx2")))
    #define TARGET_AVX2_FMA __attribute__((target("avx2,fma")))
#else
    #define TARGET_SSE3
    #define TARGET_AVX2
    #define TARGET_AVX2_FMA
#endif

// ============================================================================
// Platform abstraction
// ============================================================================

#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
    #include <intrin.h>

    static HANDLE hConsole;

    static void platform_init() {
        hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
        DWORD mode;
        GetConsoleMode(hConsole, &mode);
        SetConsoleMode(hConsole, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
    }

    static int platform_num_threads() {
        int total = (int)GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
        if (total > 0) return total;
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        return (int)si.dwNumberOfProcessors;
    }

    static bool platform_set_affinity(int thread_id) {
        GROUP_AFFINITY ga = {};
        ga.Group = (WORD)(thread_id / 64);
        ga.Mask  = (KAFFINITY)1 << (thread_id % 64);
        if (!SetThreadGroupAffinity(GetCurrentThread(), &ga, NULL)) return false;
        Sleep(0);
        return true;
    }

    static bool platform_set_high_priority() {
        return SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS) != 0;
    }

    static double now_sec() {
        static LARGE_INTEGER freq = {};
        if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
        LARGE_INTEGER t;
        QueryPerformanceCounter(&t);
        return (double)t.QuadPart / (double)freq.QuadPart;
    }

    static void cpuid(int leaf, int subleaf, uint32_t out[4]) {
        int regs[4];
        __cpuidex(regs, leaf, subleaf);
        out[0]=(uint32_t)regs[0]; out[1]=(uint32_t)regs[1];
        out[2]=(uint32_t)regs[2]; out[3]=(uint32_t)regs[3];
    }

#else // Linux / POSIX
    #include <pthread.h>
    #include <sched.h>
    #include <unistd.h>
    #include <time.h>
    #include <cpuid.h>

    static void platform_init() {}

    static int platform_num_threads() {
        return (int)sysconf(_SC_NPROCESSORS_ONLN);
    }

    static bool platform_set_affinity(int thread_id) {
        int num_cpus = (int)sysconf(_SC_NPROCESSORS_CONF);
        if (num_cpus < thread_id + 1) num_cpus = thread_id + 1;
        size_t size = CPU_ALLOC_SIZE(num_cpus);
        cpu_set_t *cpuset = CPU_ALLOC(num_cpus);
        if (!cpuset) return false;
        CPU_ZERO_S(size, cpuset);
        CPU_SET_S(thread_id, size, cpuset);
        int ret = pthread_setaffinity_np(pthread_self(), size, cpuset);
        CPU_FREE(cpuset);
        if (ret != 0) return false;
        sched_yield();
        return true;
    }

    static bool platform_set_high_priority() {
        errno = 0;
        nice(-20);
        return errno == 0;
    }

    static double now_sec() {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
    }

    static void cpuid(int leaf, int subleaf, uint32_t out[4]) {
        __cpuid_count(leaf, subleaf, out[0], out[1], out[2], out[3]);
    }
#endif

// ============================================================================
// CPU topology detection
// ============================================================================

struct TopologyInfo {
    int  physical_core[MAX_THREADS];
    int  package_id[MAX_THREADS];
    int  core_count;
    bool valid;
};

static TopologyInfo detect_topology(int max_threads) {
    TopologyInfo topo = {};
    int n = max_threads < MAX_THREADS ? max_threads : MAX_THREADS;

    for (int i = 0; i < n; i++) {
        topo.physical_core[i] = i;
        topo.package_id[i] = 0;
    }
    topo.core_count = n;
    topo.valid = false;

#ifdef _WIN32
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, NULL, &len);
    if (GetLastError() == ERROR_INSUFFICIENT_BUFFER && len > 0) {
        uint8_t *buf = new uint8_t[len];
        if (GetLogicalProcessorInformationEx(RelationProcessorCore,
                (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buf, &len)) {
            int core_idx = 0;
            DWORD offset = 0;
            while (offset < len) {
                auto *info = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)(buf + offset);
                if (info->Relationship == RelationProcessorCore) {
                    for (WORD g = 0; g < info->Processor.GroupCount; g++) {
                        KAFFINITY mask = info->Processor.GroupMask[g].Mask;
                        for (int bit = 0; bit < 64; bit++) {
                            if (mask & ((KAFFINITY)1 << bit)) {
                                int tid = (int)(g * 64 + bit);
                                if (tid < MAX_THREADS)
                                    topo.physical_core[tid] = core_idx;
                            }
                        }
                    }
                    core_idx++;
                }
                offset += info->Size;
            }
            topo.core_count = core_idx;
            topo.valid = true;
        }
        delete[] buf;
    }

    if (topo.valid) {
        DWORD pkg_len = 0;
        GetLogicalProcessorInformationEx(RelationProcessorPackage, NULL, &pkg_len);
        if (GetLastError() == ERROR_INSUFFICIENT_BUFFER && pkg_len > 0) {
            uint8_t *pbuf = new uint8_t[pkg_len];
            if (GetLogicalProcessorInformationEx(RelationProcessorPackage,
                    (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)pbuf, &pkg_len)) {
                int pkg_idx = 0;
                DWORD poff = 0;
                while (poff < pkg_len) {
                    auto *pi = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)(pbuf + poff);
                    if (pi->Relationship == RelationProcessorPackage) {
                        for (WORD g = 0; g < pi->Processor.GroupCount; g++) {
                            KAFFINITY mask = pi->Processor.GroupMask[g].Mask;
                            for (int bit = 0; bit < 64; bit++) {
                                if (mask & ((KAFFINITY)1 << bit)) {
                                    int tid = (int)(g * 64 + bit);
                                    if (tid < MAX_THREADS)
                                        topo.package_id[tid] = pkg_idx;
                                }
                            }
                        }
                        pkg_idx++;
                    }
                    poff += pi->Size;
                }
            }
            delete[] pbuf;
        }
    }
#else
    struct PkgCore { int pkg; int core; };
    PkgCore raw[MAX_THREADS] = {};
    bool ok = true;
    for (int i = 0; i < n && ok; i++) {
        char path[256];
        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu%d/topology/core_id", i);
        FILE *f = fopen(path, "r");
        if (f) {
            fscanf(f, "%d", &raw[i].core);
            fclose(f);
        } else { ok = false; }

        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", i);
        f = fopen(path, "r");
        if (f) {
            fscanf(f, "%d", &raw[i].pkg);
            fclose(f);
        } else { ok = false; }
        topo.package_id[i] = raw[i].pkg;
    }
    if (ok) {
        PkgCore unique[MAX_THREADS];
        int num_unique = 0;
        for (int i = 0; i < n; i++) {
            int idx = -1;
            for (int u = 0; u < num_unique; u++) {
                if (unique[u].pkg == raw[i].pkg && unique[u].core == raw[i].core) {
                    idx = u; break;
                }
            }
            if (idx < 0) {
                idx = num_unique;
                unique[num_unique++] = raw[i];
            }
            topo.physical_core[i] = idx;
        }
        topo.core_count = num_unique;
        topo.valid = true;
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
#define COL_WHITE   "\033[1;37m"

// ============================================================================
// CPUID detection
// ============================================================================

struct CPUFeatures {
    bool has_sse;
    bool has_sse3;
    bool has_avx2;
    bool has_fma3;
    bool os_avx_enabled;
    char brand[49];
    char vendor[13];
};

// Check if OS enabled AVX state via XGETBV (required for AVX/FMA instructions)
static uint64_t xgetbv(uint32_t xcr) {
#ifdef _WIN32
    return _xgetbv(xcr);
#else
    uint32_t lo, hi;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(xcr));
    return ((uint64_t)hi << 32) | lo;
#endif
}

static CPUFeatures detect_cpu() {
    CPUFeatures f = {};
    uint32_t r[4];

    cpuid(0, 0, r);
    memcpy(f.vendor + 0, &r[1], 4);
    memcpy(f.vendor + 4, &r[3], 4);
    memcpy(f.vendor + 8, &r[2], 4);
    f.vendor[12] = 0;

    uint32_t max_ext;
    cpuid(0x80000000, 0, r);
    max_ext = r[0];
    if (max_ext >= 0x80000004) {
        cpuid(0x80000002, 0, r);
        memcpy(f.brand + 0, r, 16);
        cpuid(0x80000003, 0, r);
        memcpy(f.brand + 16, r, 16);
        cpuid(0x80000004, 0, r);
        memcpy(f.brand + 32, r, 16);
        f.brand[48] = 0;
        char *p = f.brand;
        while (*p == ' ') p++;
        if (p != f.brand) memmove(f.brand, p, strlen(p) + 1);
    }

    cpuid(1, 0, r);
    f.has_sse        = (r[3] & (1 << 25)) != 0;
    f.has_sse3       = (r[2] & (1 << 0)) != 0;
    bool has_osxsave = (r[2] & (1 << 27)) != 0;
    bool cpu_fma3    = (r[2] & (1 << 12)) != 0;
    bool cpu_avx     = (r[2] & (1 << 28)) != 0;

    cpuid(7, 0, r);
    bool cpu_avx2 = (r[1] & (1 << 5)) != 0;

    f.os_avx_enabled = false;
    if (has_osxsave && cpu_avx) {
        uint64_t xcr0 = xgetbv(0);
        f.os_avx_enabled = ((xcr0 & 0x6) == 0x6);
    }

    f.has_avx2 = cpu_avx2 && f.os_avx_enabled;
    f.has_fma3 = cpu_fma3 && f.os_avx_enabled;

    return f;
}

// ============================================================================
// PRNG -xoshiro128**
// ============================================================================

struct PRNG {
    uint32_t s[4];

    void seed(uint32_t v) {
        for (int i = 0; i < 4; i++) {
            v += 0x9E3779B9u;
            uint32_t z = v;
            z = (z ^ (z >> 16)) * 0x85EBCA6Bu;
            z = (z ^ (z >> 13)) * 0xC2B2AE35u;
            s[i] = z ^ (z >> 16);
        }
    }

    uint32_t next() {
        uint32_t r = ((s[1] * 5) << 7 | (s[1] * 5) >> 25) * 9;
        uint32_t t = s[1] << 9;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
        s[2] ^= t; s[3] = (s[3] << 11) | (s[3] >> 21);
        return r;
    }

    float randf() { return ((float)(int32_t)next()) / (float)INT32_MAX; }
};

// ============================================================================
// Test infrastructure
// ============================================================================

static const double TOLERANCE = 0.001;
static const double WARN_THRESHOLD = 0.0001;
static const int    RERUN_COUNT = 3;
static const double RERUN_DURATION = 1.0;
static const uint64_t ITER_CHECK_FREQ = 0xFFFFF;

enum TestType { T_SCALAR = 0, T_SSE3, T_AVX2, T_FMA3, T_XLANE, NUM_TESTS };
static const char *tname[NUM_TESTS] = { "SCALAR", "SSE3", "AVX2", "FMA3", "XLANE" };

struct TestResult {
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

static TestResult run_scalar(PRNG &rng, double deadline) {
    TestResult r = {}; r.passed = true;
    for (uint64_t i = 1; ; i++) {
        volatile float qx = rng.randf(), qy = rng.randf();
        volatile float qz = rng.randf(), qw = rng.randf();
        volatile float x = qx, y = qy, z = qz, w = qw;
        volatile float ls = x*x + y*y + z*z + w*w;
        if (ls < 1e-10f) continue;
        volatile float il = 1.0f / sqrtf((float)ls);
        volatile float nx = x*il, ny = y*il, nz = z*il, nw = w*il;
        volatile float ck = nx*nx + ny*ny + nz*nz + nw*nw;
        double dev = fabs((double)(float)ck - 1.0);
        if (dev > r.worst_dev) r.worst_dev = dev;
        if (dev > TOLERANCE) {
            r.passed = false; r.worst_dev = dev; r.worst_iter = i;
            r.worst_q[0]=nx; r.worst_q[1]=ny; r.worst_q[2]=nz; r.worst_q[3]=nw;
            r.iterations = i; return r;
        }
        r.iterations = i;
        if ((i & ITER_CHECK_FREQ) == 0 && now_sec() >= deadline) break;
    }
    return r;
}

// SSE3 helper - dot product for verification
TARGET_SSE3
static float sse3_dot4(__m128 v) {
    __m128 sq2=_mm_mul_ps(v,v);
    __m128 h1=_mm_hadd_ps(sq2,sq2), h2=_mm_hadd_ps(h1,h1);
    volatile float c; _mm_store_ss((float*)&c,h2); return c;
}

// SSE3 -128-bit SIMD

TARGET_SSE3
static TestResult run_sse(PRNG &rng, double deadline) {
    TestResult r = {}; r.passed = true;
    for (uint64_t i = 1; ; i++) {
        volatile float qx=rng.randf(), qy=rng.randf(), qz=rng.randf(), qw=rng.randf();
        __m128 q = _mm_set_ps((float)qw,(float)qz,(float)qy,(float)qx);
        __m128 sq = _mm_mul_ps(q,q);
        __m128 s1 = _mm_hadd_ps(sq,sq), s2 = _mm_hadd_ps(s1,s1);
        volatile float ls; _mm_store_ss((float*)&ls, s2);
        if ((float)ls < 1e-10f) continue;

        __m128 lv = _mm_shuffle_ps(s2,s2,0);
        __m128 inv = _mm_rsqrt_ps(lv);
        __m128 nr = _mm_mul_ps(_mm_mul_ps(_mm_set1_ps(0.5f),lv),_mm_mul_ps(inv,inv));
        inv = _mm_mul_ps(inv, _mm_sub_ps(_mm_set1_ps(1.5f), nr));
        __m128 na = _mm_mul_ps(q, inv);
        __m128 nb = _mm_div_ps(q, _mm_sqrt_ps(lv));

        double da = fabs((double)sse3_dot4(na)-1.0), db = fabs((double)sse3_dot4(nb)-1.0);
        double worst = da>db?da:db;
        if (worst > r.worst_dev) r.worst_dev = worst;
        if (worst > TOLERANCE) {
            r.passed = false; r.worst_dev = worst; r.worst_iter = i;
            volatile float t[4]; _mm_storeu_ps((float*)t, da>db?na:nb);
            r.worst_q[0]=t[0]; r.worst_q[1]=t[1]; r.worst_q[2]=t[2]; r.worst_q[3]=t[3];
            r.iterations = i; return r;
        }
        r.iterations = i;
        if ((i & ITER_CHECK_FREQ) == 0 && now_sec() >= deadline) break;
    }
    return r;
}

// AVX2 -256-bit SIMD (two quaternions simultaneously)

TARGET_AVX2
static TestResult run_avx2(PRNG &rng, double deadline) {
    TestResult r = {}; r.passed = true;
    for (uint64_t i = 1; ; i++) {
        volatile float a=rng.randf(),b=rng.randf(),c=rng.randf(),d=rng.randf();
        volatile float e=rng.randf(),f=rng.randf(),g=rng.randf(),h=rng.randf();
        __m256 q = _mm256_set_ps((float)h,(float)g,(float)f,(float)e,
                                  (float)d,(float)c,(float)b,(float)a);
        __m256 sq = _mm256_mul_ps(q,q);
        __m256 h1 = _mm256_hadd_ps(sq,sq), h2 = _mm256_hadd_ps(h1,h1);
        volatile float l1,l2;
        _mm_store_ss((float*)&l1, _mm256_castps256_ps128(h2));
        _mm_store_ss((float*)&l2, _mm256_extractf128_ps(h2,1));
        if ((float)l1<1e-10f||(float)l2<1e-10f) continue;

        __m256 lb = _mm256_hadd_ps(_mm256_hadd_ps(sq,sq),_mm256_hadd_ps(sq,sq));
        __m256 norm = _mm256_div_ps(q, _mm256_sqrt_ps(lb));
        __m256 ns = _mm256_mul_ps(norm,norm);
        __m256 nh1=_mm256_hadd_ps(ns,ns), nh2=_mm256_hadd_ps(nh1,nh1);
        volatile float c1,c2;
        _mm_store_ss((float*)&c1, _mm256_castps256_ps128(nh2));
        _mm_store_ss((float*)&c2, _mm256_extractf128_ps(nh2,1));
        double d1=fabs((double)(float)c1-1.0), d2=fabs((double)(float)c2-1.0);
        { double w = d1>d2?d1:d2; if (w > r.worst_dev) r.worst_dev = w; }

        if (d1>TOLERANCE||d2>TOLERANCE) {
            r.passed = false; r.worst_dev = d1>d2?d1:d2; r.worst_iter = i;
            volatile float t[8]; _mm256_storeu_ps((float*)t, norm);
            int w = d1>d2?0:4;
            r.worst_q[0]=t[w]; r.worst_q[1]=t[w+1]; r.worst_q[2]=t[w+2]; r.worst_q[3]=t[w+3];
            r.iterations = i; _mm256_zeroupper(); return r;
        }
        r.iterations = i;
        if ((i & ITER_CHECK_FREQ) == 0 && now_sec() >= deadline) break;
    }
    _mm256_zeroupper();
    return r;
}

// FMA3 helper - dot product using fused multiply-add
TARGET_AVX2_FMA
static float fma3_check(__m128 n) {
    volatile float v[4]; _mm_storeu_ps((float*)v,n);
    __m128 a2=_mm_set1_ps((float)v[0]),b2=_mm_set1_ps((float)v[1]);
    __m128 c2=_mm_set1_ps((float)v[2]),d2=_mm_set1_ps((float)v[3]);
    __m128 r2=_mm_mul_ss(a2,a2);
    r2=_mm_fmadd_ss(b2,b2,r2); r2=_mm_fmadd_ss(c2,c2,r2); r2=_mm_fmadd_ss(d2,d2,r2);
    volatile float o; _mm_store_ss((float*)&o,r2); return o;
}

// FMA3 -fused multiply-add pipeline

TARGET_AVX2_FMA
static TestResult run_fma3(PRNG &rng, double deadline) {
    TestResult r = {}; r.passed = true;
    for (uint64_t i = 1; ; i++) {
        volatile float qx=rng.randf(),qy=rng.randf(),qz=rng.randf(),qw=rng.randf();
        __m128 q = _mm_set_ps((float)qw,(float)qz,(float)qy,(float)qx);
        __m128 xx=_mm_set1_ps((float)qx), yy=_mm_set1_ps((float)qy);
        __m128 zz=_mm_set1_ps((float)qz), ww=_mm_set1_ps((float)qw);
        __m128 dot=_mm_mul_ss(xx,xx);
        dot=_mm_fmadd_ss(yy,yy,dot);
        dot=_mm_fmadd_ss(zz,zz,dot);
        dot=_mm_fmadd_ss(ww,ww,dot);
        volatile float ls; _mm_store_ss((float*)&ls,dot);
        if ((float)ls<1e-10f) continue;

        __m128 inv=_mm_rsqrt_ss(dot);
        __m128 nrf=_mm_fnmadd_ss(_mm_mul_ss(dot,_mm_set_ss(0.5f)),_mm_mul_ss(inv,inv),_mm_set_ss(1.5f));
        inv=_mm_mul_ss(inv,nrf);
        __m128 na=_mm_mul_ps(q, _mm_shuffle_ps(inv,inv,0));
        __m128 len=_mm_sqrt_ss(dot);
        __m128 nb=_mm_div_ps(q, _mm_shuffle_ps(len,len,0));

        double da=fabs((double)fma3_check(na)-1.0), db=fabs((double)fma3_check(nb)-1.0);
        double worst=da>db?da:db;
        if (worst > r.worst_dev) r.worst_dev = worst;
        if (worst>TOLERANCE) {
            r.passed=false; r.worst_dev=worst; r.worst_iter=i;
            volatile float t[4]; _mm_storeu_ps((float*)t, da>db?na:nb);
            r.worst_q[0]=t[0]; r.worst_q[1]=t[1]; r.worst_q[2]=t[2]; r.worst_q[3]=t[3];
            r.iterations=i; return r;
        }
        r.iterations = i;
        if ((i & ITER_CHECK_FREQ) == 0 && now_sec() >= deadline) break;
    }
    return r;
}

// XLANE - AVX2 cross-lane data integrity (bit-exact)

TARGET_AVX2
static TestResult run_xlane(PRNG &rng, double deadline) {
    TestResult r = {}; r.passed = true;
    for (uint64_t i = 1; ; i++) {
        volatile float f0=rng.randf(), f1=rng.randf(), f2=rng.randf(), f3=rng.randf();
        volatile float f4=rng.randf(), f5=rng.randf(), f6=rng.randf(), f7=rng.randf();

        __m256 src = _mm256_set_ps((float)f7,(float)f6,(float)f5,(float)f4,
                                    (float)f3,(float)f2,(float)f1,(float)f0);

        __m256 swapped = _mm256_permute2f128_ps(src, src, 0x01);
        volatile float sw[8]; _mm256_storeu_ps((float*)sw, swapped);
        volatile float expected_sw[8] = {f4,f5,f6,f7, f0,f1,f2,f3};
        for (int j = 0; j < 8; j++) {
            if (sw[j] != expected_sw[j]) {
                r.passed = false; r.worst_dev = 1.0; r.worst_iter = i;
                r.worst_q[0]=sw[j]; r.worst_q[1]=expected_sw[j];
                r.worst_q[2]=(float)j; r.worst_q[3]=0.0f;
                r.iterations = i; _mm256_zeroupper(); return r;
            }
        }

        __m256i idx_rev = _mm256_set_epi32(0,1,2,3,4,5,6,7);
        __m256  rev = _mm256_permutevar8x32_ps(src, idx_rev);
        volatile float rv[8]; _mm256_storeu_ps((float*)rv, rev);

        volatile float expected_rv[8] = {f7,f6,f5,f4, f3,f2,f1,f0};
        for (int j = 0; j < 8; j++) {
            if (rv[j] != expected_rv[j]) {
                r.passed = false; r.worst_dev = 1.0; r.worst_iter = i;
                r.worst_q[0]=rv[j]; r.worst_q[1]=expected_rv[j];
                r.worst_q[2]=(float)j; r.worst_q[3]=1.0f;
                r.iterations = i; _mm256_zeroupper(); return r;
            }
        }

        __m256i idx_rot = _mm256_set_epi32(2,1,0,7,6,5,4,3);
        __m256  rot = _mm256_permutevar8x32_ps(src, idx_rot);
        volatile float rt[8]; _mm256_storeu_ps((float*)rt, rot);

        volatile float expected_rt[8] = {f3,f4,f5,f6, f7,f0,f1,f2};
        for (int j = 0; j < 8; j++) {
            if (rt[j] != expected_rt[j]) {
                r.passed = false; r.worst_dev = 1.0; r.worst_iter = i;
                r.worst_q[0]=rt[j]; r.worst_q[1]=expected_rt[j];
                r.worst_q[2]=(float)j; r.worst_q[3]=2.0f;
                r.iterations = i; _mm256_zeroupper(); return r;
            }
        }

        r.iterations = i;
        if ((i & ITER_CHECK_FREQ) == 0 && now_sec() >= deadline) break;
    }
    _mm256_zeroupper();
    return r;
}

// ============================================================================
// Test dispatch
// ============================================================================

typedef TestResult (*TestFunc)(PRNG&, double);
static TestFunc test_funcs[NUM_TESTS] = {
    run_scalar, run_sse, run_avx2, run_fma3, run_xlane
};

// Deterministic rerun to confirm failures

static TestResult rerun_single(TestFunc fn, uint32_t seed, double duration) {
    PRNG rng;
    rng.seed(seed);
    return fn(rng, now_sec() + duration);
}

COREPROBE_RESTORE_OPT

// ============================================================================
// Argument parsing
// ============================================================================

static int parse_threads(int argc, char **argv, int *out, int max_threads) {
    bool seen[MAX_THREADS] = {};
    int count = 0;
    for (int a = 2; a < argc && count < MAX_THREADS; a++) {
        if (argv[a][0] == '-' && argv[a][1] == '-') {
            if (strcmp(argv[a], "--socket") == 0 || strcmp(argv[a], "--repeat") == 0) a++;
            continue;
        }
        char *dash = strchr(argv[a], '-');
        if (dash && dash != argv[a]) {
            int lo = atoi(argv[a]);
            int hi = atoi(dash + 1);
            for (int t = lo; t <= hi && count < MAX_THREADS; t++) {
                if (t >= 0 && t < max_threads && !seen[t]) {
                    seen[t] = true;
                    out[count++] = t;
                }
            }
        } else {
            int t = atoi(argv[a]);
            if (t >= 0 && t < max_threads && !seen[t]) {
                seen[t] = true;
                out[count++] = t;
            }
        }
    }
    if (count == 0) {
        for (int i = 0; i < max_threads && i < MAX_THREADS; i++) out[i] = i;
        return max_threads < MAX_THREADS ? max_threads : MAX_THREADS;
    }
    return count;
}

// ============================================================================
// JSON output
// ============================================================================

struct CoreResult {
    int        thread_id;
    bool       affinity_ok;
    TestResult tests[NUM_TESTS];
};

static void write_json(const char *path, CPUFeatures &cpu, TopologyInfo &topo,
                       CoreResult *all, int num_results, double wall_time) {
    FILE *fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "  Warning: could not write %s\n", path);
        return;
    }

    fprintf(fp, "{\n");
    fprintf(fp, "  \"version\": \"%s\",\n", COREPROBE_VERSION);
    fprintf(fp, "  \"cpu\": {\n");
    fprintf(fp, "    \"brand\": \"%s\",\n", cpu.brand);
    fprintf(fp, "    \"vendor\": \"%s\",\n", cpu.vendor);
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

    for (int i = 0; i < num_results; i++) {
        CoreResult *cr = &all[i];
        fprintf(fp, "    {\n");
        fprintf(fp, "      \"thread\": %d,\n", cr->thread_id);
        fprintf(fp, "      \"physical_core\": %d,\n", topo.physical_core[cr->thread_id]);
        fprintf(fp, "      \"package\": %d,\n", topo.package_id[cr->thread_id]);
        fprintf(fp, "      \"affinity_ok\": %s,\n", cr->affinity_ok ? "true" : "false");
        fprintf(fp, "      \"tests\": {\n");
        for (int t = 0; t < NUM_TESTS; t++) {
            TestResult *tr = &cr->tests[t];
            fprintf(fp, "        \"%s\": {\n", tname[t]);
            if (tr->skipped) {
                fprintf(fp, "          \"status\": \"skipped\"\n");
            } else if (tr->passed) {
                fprintf(fp, "          \"status\": \"pass\",\n");
                fprintf(fp, "          \"iterations\": %llu\n",
                        (unsigned long long)tr->iterations);
            } else {
                fprintf(fp, "          \"status\": \"FAIL\",\n");
                fprintf(fp, "          \"iterations\": %llu,\n",
                        (unsigned long long)tr->iterations);
                fprintf(fp, "          \"fail_iteration\": %llu,\n",
                        (unsigned long long)tr->worst_iter);
                fprintf(fp, "          \"deviation\": %.10f,\n", tr->worst_dev);
                fprintf(fp, "          \"quaternion\": [%.8f, %.8f, %.8f, %.8f],\n",
                        tr->worst_q[0], tr->worst_q[1], tr->worst_q[2], tr->worst_q[3]);
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
    fclose(fp);
}

// ============================================================================
// Help
// ============================================================================

static void print_help(const char *argv0) {
    printf("\n");
    printf("  coreprobe v%s - Per-Core FPU/SIMD Correctness Diagnostic\n\n", COREPROBE_VERSION);
    printf("  Stress-tests quaternion normalization across SCALAR, SSE3, AVX2, FMA3\n");
    printf("  plus cross-lane AVX2 data integrity (XLANE) on each logical processor\n");
    printf("  to detect faulty floating-point and SIMD execution units.\n");
    printf("  Catches defects that memtest86+, Prime95, and WHEA reporting miss.\n\n");
    printf("  Usage:\n");
    printf("    %s [seconds] [threads...] [flags]\n\n", argv0);
    printf("  Examples:\n");
    printf("    %s                  test all threads, 120s default\n", argv0);
    printf("    %s 60               test all threads, 60s\n", argv0);
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

static void print_core_map(CoreResult *all, int num_results, int max_threads,
                           TopologyInfo &topo) {
    int *thread_status = new int[max_threads]();
    for (int i = 0; i < num_results; i++) {
        if (!all[i].affinity_ok) continue;
        bool ok = true;
        for (int t = 0; t < NUM_TESTS; t++) {
            if (!all[i].tests[t].passed && !all[i].tests[t].skipped) ok = false;
        }
        thread_status[all[i].thread_id] = ok ? 1 : 2;
    }

    int *core_status = new int[topo.core_count]();
    for (int tid = 0; tid < max_threads; tid++) {
        int phys = topo.physical_core[tid];
        if (phys >= 0 && phys < topo.core_count) {
            if (thread_status[tid] > core_status[phys])
                core_status[phys] = thread_status[tid];
        }
    }

    printf("  Core Map (%d physical cores%s):\n\n  ",
           topo.core_count, topo.valid ? ", OS topology" : ", heuristic");

    for (int phys = 0; phys < topo.core_count; phys++) {
        if (core_status[phys] == 2)      printf(COL_RED);
        else if (core_status[phys] == 1) printf(COL_GREEN);
        else                             printf(COL_GRAY);

        printf("[%2d]", phys);
        printf(COL_RESET);

        if ((phys + 1) % 8 == 0 && phys + 1 < topo.core_count) {
            printf("  |  ");
        } else {
            printf(" ");
        }
    }
    printf("\n");

    printf("\n  ");
    printf(COL_GREEN "[OK]" COL_RESET " = pass  ");
    printf(COL_RED "[XX]" COL_RESET " = FAIL  ");
    printf(COL_GRAY "[--]" COL_RESET " = not tested\n");

    delete[] thread_status;
    delete[] core_status;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv) {
    platform_init();

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_help(argv[0]);
            return 0;
        }
    }

    bool json_output = false;
    bool soak_mode = false;
    bool pause_at_end = false;
    bool until_fail = false;
    int  socket_filter = -1;
    int  repeat_count = 1;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--json") == 0) json_output = true;
        if (strcmp(argv[i], "--soak") == 0) soak_mode = true;
        if (strcmp(argv[i], "--pause") == 0) pause_at_end = true;
        if (strcmp(argv[i], "--until-fail") == 0) until_fail = true;
        if (strcmp(argv[i], "--socket") == 0 && i + 1 < argc)
            socket_filter = atoi(argv[++i]);
        if (strcmp(argv[i], "--repeat") == 0 && i + 1 < argc)
            repeat_count = atoi(argv[++i]);
    }
    if (repeat_count < 1) repeat_count = 1;
    if (until_fail) repeat_count = INT32_MAX;

    int max_threads = platform_num_threads();
    CPUFeatures cpu = detect_cpu();
    TopologyInfo topo = detect_topology(max_threads);

    int total_seconds = 120;
    if (soak_mode) total_seconds = 600;
    if (argc > 1 && argv[1][0] != '-') {
        total_seconds = atoi(argv[1]);
        if (total_seconds <= 0) total_seconds = 120;
    }

    int thread_list[MAX_THREADS];
    int num_threads = parse_threads(argc, argv, thread_list, max_threads);

    if (socket_filter >= 0) {
        if (!topo.valid) {
            printf(COL_YELLOW "  Warning: --socket %d requested but topology detection failed, "
                   "testing all threads\n" COL_RESET, socket_filter);
        } else {
            int filtered[MAX_THREADS];
            int nf = 0;
            for (int i = 0; i < num_threads; i++) {
                if (topo.package_id[thread_list[i]] == socket_filter)
                    filtered[nf++] = thread_list[i];
            }
            if (nf > 0) {
                memcpy(thread_list, filtered, nf * sizeof(int));
                num_threads = nf;
            } else {
                printf(COL_RED "  Error: no threads found for socket %d\n" COL_RESET, socket_filter);
                return 1;
            }
        }
    }

    if (num_threads == 0) {
        printf(COL_RED "  Error: no threads to test\n" COL_RESET);
        return 1;
    }

    double secs_per_thread = (double)total_seconds / num_threads;
    double secs_per_test = secs_per_thread / NUM_TESTS;
    if (secs_per_test < 2.0) secs_per_test = 2.0;

    printf(COL_CYAN "\n");
    printf("  +================================================================+\n");
    printf("  |       coreprobe v%-6s - FPU/SIMD Correctness Diagnostic    |\n", COREPROBE_VERSION);
    printf("  +================================================================+\n\n" COL_RESET);

    printf("  CPU:                 %s\n", cpu.brand);
    printf("  Vendor:              %s\n", cpu.vendor);
    printf("  Logical processors:  %d\n", max_threads);
    printf("  Physical cores:      %d%s\n", topo.core_count,
           topo.valid ? " (OS topology)" : " (heuristic — 1 thread per core)");
    printf("  Instruction sets:    SSE3=%s  AVX2=%s  FMA3=%s\n",
           cpu.has_sse3 ? COL_GREEN "yes" COL_RESET : COL_RED "no" COL_RESET,
           cpu.has_avx2 ? COL_GREEN "yes" COL_RESET : COL_RED "no" COL_RESET,
           cpu.has_fma3 ? COL_GREEN "yes" COL_RESET : COL_RED "no" COL_RESET);
    printf("  OS AVX state:        %s\n",
           cpu.os_avx_enabled ? COL_GREEN "enabled (XSAVE/XGETBV)" COL_RESET
                              : COL_YELLOW "disabled - AVX/FMA/XLANE tests will be skipped" COL_RESET);
    printf("  Testing threads:     ");
    if (socket_filter >= 0) {
        printf("%d (socket %d only)\n", num_threads, socket_filter);
    } else if (num_threads == max_threads) {
        printf("ALL (%d)\n", num_threads);
    } else {
        for (int i = 0; i < num_threads; i++)
            printf("%d%s", thread_list[i], i < num_threads-1 ? ", " : "\n");
    }
    printf("  Mode:                %s\n", soak_mode ? "SOAK (extended)" : "standard");
    printf("  Duration:            ~%ds total (%.1fs/thread, %.1fs/test)\n",
           total_seconds, secs_per_thread, secs_per_test);
    printf("  Tolerance:           %.6f\n", TOLERANCE);
    printf("  Rerun on fail:       %dx (deterministic seed replay)\n", RERUN_COUNT);
    printf("  Compile flags:       -O0, volatile floats (correctness test, not throughput)\n");

    if (platform_set_high_priority())
        printf("  Priority:            " COL_GREEN "HIGH" COL_RESET "\n");
    else
        printf("  Priority:            " COL_YELLOW "normal (needs elevated privileges)" COL_RESET "\n");

    if (!cpu.os_avx_enabled) {
        printf("  " COL_YELLOW "Warning: OS has not enabled AVX state (XGETBV XCR0 bits 1:2)." COL_RESET "\n");
        printf("  " COL_YELLOW "  AVX2 and FMA3 tests will be skipped. This is unusual on" COL_RESET "\n");
        printf("  " COL_YELLOW "  modern systems -check BIOS settings or OS configuration." COL_RESET "\n");
    } else {
        if (!cpu.has_avx2) printf("  " COL_YELLOW "Note: AVX2 not supported by CPU, test will be skipped" COL_RESET "\n");
        if (!cpu.has_fma3) printf("  " COL_YELLOW "Note: FMA3 not supported by CPU, test will be skipped" COL_RESET "\n");
    }
    if (!cpu.has_sse3) printf("  " COL_YELLOW "Note: SSE3 not supported by CPU, test will be skipped" COL_RESET "\n");

    printf("\n");

    printf(COL_GRAY "  %-5s %-13s %-13s %-13s %-13s %-13s %s" COL_RESET "\n",
           "THR", "SCALAR", "SSE3", "AVX2", "FMA3", "XLANE", "STATUS");
    printf(COL_GRAY "  %-5s %-13s %-13s %-13s %-13s %-13s %s" COL_RESET "\n",
           "---", "--------", "--------", "--------", "--------", "--------", "------");

    int overall_fails = 0;
    int pass_number = 0;

    for (int pass = 0; pass < repeat_count; pass++) {
        pass_number = pass + 1;

        if (repeat_count > 1) {
            printf(COL_CYAN "\n  === Pass %d%s ===" COL_RESET "\n\n",
                   pass_number, until_fail ? " (until-fail mode)" : "");
            printf(COL_GRAY "  %-5s %-13s %-13s %-13s %-13s %-13s %s" COL_RESET "\n",
                   "THR", "SCALAR", "SSE3", "AVX2", "FMA3", "XLANE", "STATUS");
            printf(COL_GRAY "  %-5s %-13s %-13s %-13s %-13s %-13s %s" COL_RESET "\n",
                   "---", "--------", "--------", "--------", "--------", "--------", "------");
        }

    CoreResult *all = new CoreResult[num_threads]();
    double wall_start = now_sec();

    for (int ci = 0; ci < num_threads; ci++) {
        int tid = thread_list[ci];
        CoreResult *cr = &all[ci];
        cr->thread_id = tid;

        if (!platform_set_affinity(tid)) {
            cr->affinity_ok = false;
            printf("  " COL_RED "T%-3d  affinity FAILED, skipping" COL_RESET "\n", tid);
            continue;
        }
        cr->affinity_ok = true;

        printf("  " COL_YELLOW "T%-3d" COL_RESET "  ", tid);
        fflush(stdout);

        bool core_ok = true;
        PRNG rng;

        for (int t = 0; t < NUM_TESTS; t++) {
            if (t == T_SSE3 && !cpu.has_sse3) {
                cr->tests[t].skipped = true;
                cr->tests[t].passed = true;
                printf(COL_GRAY "%-13s" COL_RESET, "skip");
                fflush(stdout);
                continue;
            }
            if (t == T_AVX2 && !cpu.has_avx2) {
                cr->tests[t].skipped = true;
                cr->tests[t].passed = true;
                printf(COL_GRAY "%-13s" COL_RESET, "skip");
                fflush(stdout);
                continue;
            }
            if (t == T_FMA3 && !cpu.has_fma3) {
                cr->tests[t].skipped = true;
                cr->tests[t].passed = true;
                printf(COL_GRAY "%-13s" COL_RESET, "skip");
                fflush(stdout);
                continue;
            }
            if (t == T_XLANE && !cpu.has_avx2) {
                cr->tests[t].skipped = true;
                cr->tests[t].passed = true;
                printf(COL_GRAY "%-13s" COL_RESET, "skip");
                fflush(stdout);
                continue;
            }

            uint32_t seed = 0xF00D0000u + ((uint32_t)pass << 20)
                          + (uint32_t)tid * NUM_TESTS + (uint32_t)t;
            rng.seed(seed);
            double deadline = now_sec() + secs_per_test;
            cr->tests[t] = test_funcs[t](rng, deadline);
            cr->tests[t].fail_seed = seed;

            if (cr->tests[t].passed) {
                char buf[32];
                double wd = cr->tests[t].worst_dev;
                if (t != T_XLANE && wd > WARN_THRESHOLD) {
                    // WARN: passed but near threshold
                    snprintf(buf, sizeof(buf), "WARN %.3f%%", wd * 100.0);
                    printf(COL_YELLOW "%-13s" COL_RESET, buf);
                } else {
                    snprintf(buf, sizeof(buf), "PASS %lluM",
                             (unsigned long long)(cr->tests[t].iterations / 1000000ULL));
                    printf(COL_GREEN "%-13s" COL_RESET, buf);
                }
            } else {
                core_ok = false;

                // Deterministic rerun to confirm
                cr->tests[t].rerun_fails = 0;
                double rerun_dur = secs_per_test > RERUN_DURATION ? secs_per_test : RERUN_DURATION;
                for (int rr = 0; rr < RERUN_COUNT; rr++) {
                    TestResult rerun = rerun_single(test_funcs[t], seed, rerun_dur);
                    if (!rerun.passed) cr->tests[t].rerun_fails++;
                }
                cr->tests[t].confirmed = (cr->tests[t].rerun_fails > 0);

                char buf[32];
                if (t == T_XLANE) {
                    snprintf(buf, sizeof(buf), "FAIL mismatch");
                } else {
                    snprintf(buf, sizeof(buf), "FAIL %.3f%%",
                             cr->tests[t].worst_dev * 100.0);
                }

                if (cr->tests[t].confirmed)
                    printf(COL_RED "%-13s" COL_RESET, buf);
                else
                    printf(COL_YELLOW "%-13s" COL_RESET, buf);
            }
            fflush(stdout);
        }

        if (core_ok) {
            printf(COL_GREEN " OK" COL_RESET);
        } else {
            printf(COL_RED " ** FAIL **" COL_RESET);
        }
        printf("\n");
    }

    double wall_end = now_sec();

    printf("\n");
    printf(COL_CYAN "  +================================================================+\n");
    printf("  |                        SUMMARY                                 |\n");
    printf("  +================================================================+\n" COL_RESET);
    printf("\n");

    int total_fails = 0;
    int confirmed_fails = 0;
    int fail_indices[MAX_THREADS];
    int num_fail = 0;

    for (int ci = 0; ci < num_threads; ci++) {
        if (!all[ci].affinity_ok) continue;
        bool has_fail = false;
        for (int t = 0; t < NUM_TESTS; t++) {
            if (!all[ci].tests[t].passed && !all[ci].tests[t].skipped) {
                total_fails++;
                if (all[ci].tests[t].confirmed) confirmed_fails++;
                has_fail = true;
            }
        }
        if (has_fail) fail_indices[num_fail++] = ci;
    }

    if (total_fails > 0) {
        printf(COL_RED "  *** FPU ERRORS DETECTED ***" COL_RESET "\n\n");

        for (int fi = 0; fi < num_fail; fi++) {
            CoreResult *cr = &all[fail_indices[fi]];
            int phys = topo.physical_core[cr->thread_id];
            int pkg  = topo.package_id[cr->thread_id];

            printf("  Thread %d (physical core %d, package %d%s):\n",
                   cr->thread_id, phys, pkg,
                   topo.valid ? "" : ", heuristic");

            for (int t = 0; t < NUM_TESTS; t++) {
                TestResult *tr = &cr->tests[t];
                if (tr->skipped) continue;
                if (!tr->passed) {
                    printf("    " COL_RED "%s FAILED" COL_RESET, tname[t]);
                    printf(" at iter %llu  deviation=%.10f",
                           (unsigned long long)tr->worst_iter, tr->worst_dev);
                    if (tr->confirmed)
                        printf("  " COL_RED "[confirmed %d/%d reruns]" COL_RESET,
                               tr->rerun_fails, RERUN_COUNT);
                    else
                        printf("  " COL_YELLOW "[transient, 0/%d reruns]" COL_RESET,
                               RERUN_COUNT);
                    printf("\n");
                    printf("      quat(%.8f, %.8f, %.8f, %.8f)\n",
                           tr->worst_q[0], tr->worst_q[1], tr->worst_q[2], tr->worst_q[3]);
                } else {
                    printf("    " COL_GREEN "%s OK" COL_RESET "\n", tname[t]);
                }
            }
            printf("\n");
        }

        printf("  " COL_MAGENTA "Diagnosis:" COL_RESET "\n");
        printf("  Affected physical core(s): ");
        bool seen[MAX_THREADS] = {};
        for (int fi = 0; fi < num_fail; fi++) {
            int phys = topo.physical_core[all[fail_indices[fi]].thread_id];
            if (!seen[phys]) {
                printf(COL_RED "Core %d " COL_RESET, phys);
                seen[phys] = true;
            }
        }
        printf("\n");

        bool scalar_ok_simd_fail = false;
        for (int fi = 0; fi < num_fail; fi++) {
            CoreResult *cr = &all[fail_indices[fi]];
            if (cr->tests[T_SCALAR].passed) {
                for (int t = T_SSE3; t < NUM_TESTS; t++) {
                    if (!cr->tests[t].passed && !cr->tests[t].skipped)
                        scalar_ok_simd_fail = true;
                }
            }
        }

        if (scalar_ok_simd_fail) {
            printf("\n  " COL_YELLOW "Pattern: SCALAR passes but SIMD fails" COL_RESET "\n");
            printf("  This indicates SIMD execution units (SSE/AVX/FMA/lane-crossing)\n");
            printf("  are faulty while the scalar FP pipeline is intact. Common causes:\n");
            printf("    - Silicon defect in SIMD execution unit on affected core\n");
            printf("    - Degraded CPU (age, heat damage, electromigration)\n");
            printf("    - If on OC/PBO: reduce clocks or increase voltage\n");
            printf("    - If on stock: CPU hardware fault, consider RMA or replacement\n");
        }

        bool xlane_only = false;
        for (int fi = 0; fi < num_fail; fi++) {
            CoreResult *cr = &all[fail_indices[fi]];
            if (!cr->tests[T_XLANE].skipped && !cr->tests[T_XLANE].passed &&
                cr->tests[T_SCALAR].passed && cr->tests[T_SSE3].passed &&
                (cr->tests[T_AVX2].passed || cr->tests[T_AVX2].skipped) &&
                (cr->tests[T_FMA3].passed || cr->tests[T_FMA3].skipped))
                xlane_only = true;
        }
        if (xlane_only) {
            printf("\n  " COL_YELLOW "Pattern: only XLANE fails" COL_RESET "\n");
            printf("  Arithmetic is correct but cross-lane data movement is corrupted.\n");
            printf("  This points to the AVX2 lane-crossing interconnect specifically.\n");
        }

        if (confirmed_fails > 0) {
            printf("\n  " COL_RED "Confirmed failures reproduce with identical seeds." COL_RESET "\n");
            printf("  " COL_RED "This is a hardware defect, not a transient error." COL_RESET "\n");
        }
    } else {
        printf(COL_GREEN "  ALL TESTS PASSED -no FPU/SIMD errors detected." COL_RESET "\n");
    }

    printf("\n");

    print_core_map(all, num_threads, max_threads, topo);

    printf("\n  Wall time: %.1f seconds\n", wall_end - wall_start);

    if (json_output) {
        write_json("coreprobe_results.json", cpu, topo, all, num_threads, wall_end - wall_start);
        printf("  Results written to: coreprobe_results.json\n");
    }

    printf("\n");

    delete[] all;
    overall_fails += total_fails;

    if (total_fails > 0) break;
    if (repeat_count > 1 && pass + 1 < repeat_count) {
        printf("  Pass %d complete -no errors. Continuing...\n", pass_number);
    }

    }

    if (repeat_count > 1 && overall_fails == 0) {
        printf(COL_GREEN "\n  All %d passes completed with no failures." COL_RESET "\n", pass_number);
    }

    int exit_code = overall_fails > 0 ? 1 : 0;

    if (pause_at_end) {
        printf("  Press Enter to exit...");
        fflush(stdout);
        getchar();
    }

    return exit_code;
}