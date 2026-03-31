# coreprobe

Per-core FPU/SIMD correctness diagnostic for x86-64. Catches silicon defects that memtest86+, Prime95, and WHEA miss.

Developed after discovering a Ryzen 9 5950X with a single core whose SIMD units produced incorrect results under quaternion math, causing Unreal Engine 5 `IsRotationNormalized()` assertion crashes that only appeared when the OS scheduler placed work on that specific core.

## Tests

| Test | Instruction Path | What It Catches | Comparison |
|------|-----------------|-----------------|------------|
| `SCALAR` | Scalar `1/sqrt` | Basic FPU errors | Tolerance |
| `SSE` | 128-bit `rsqrt+NR` / `sqrt+div` | SSE pipeline faults | Tolerance |
| `AVX2` | 256-bit `sqrt+div` | AVX2 execution unit errors | Tolerance |
| `FMA3` | FMA `rsqrt+NR` / `sqrt+div` | Fused multiply-add faults | Tolerance |
| `XLANE` | `vperm2f128`, `vpermd` | Cross-lane data corruption | Bit-exact |

The **XLANE** test specifically targets the 128-bit lane boundary in AVX2 registers -- a known weak point where data corruption can occur even when lane-local arithmetic appears healthy.

## Build

MinGW:
```
g++ -O0 -mavx2 -mfma -std=c++17 -o coreprobe.exe coreprobe.cpp
```

MSVC:
```
cl /Od /arch:AVX2 /std:c++17 /EHsc coreprobe.cpp /Fe:coreprobe.exe
```

Linux:
```
g++ -O0 -mavx2 -mfma -std=c++17 -lpthread -o coreprobe coreprobe.cpp
```

> **`-O0` / `/Od` is intentional.** This is an arithmetic correctness test, not a throughput benchmark. Optimizations can mask hardware faults by reordering or eliminating floating-point operations.

## Usage

```
coreprobe [seconds] [threads...] [flags]
```

| Example | Description |
|---------|-------------|
| `coreprobe` | All threads, 120 seconds |
| `coreprobe 60` | All threads, 60 seconds |
| `coreprobe 20 4` | Thread 4 only, 20 seconds |
| `coreprobe 60 0-15` | Threads 0-15, 60 seconds |
| `coreprobe --soak` | Extended 10-minute soak test |
| `coreprobe --socket 0` | Only threads on socket 0 |
| `coreprobe --repeat 5` | Run 5 full passes |
| `coreprobe --until-fail` | Repeat until failure detected |
| `coreprobe 30 --json` | Write results to `coreprobe_results.json` |
| `coreprobe --pause` | Wait for Enter before exiting |

## Output

```
T0   SCALAR PASS 12M  SSE PASS 8M   AVX2 PASS 6M   FMA3 PASS 10M  XLANE PASS 9M   OK
T1   SCALAR PASS 11M  SSE PASS 8M   AVX2 PASS 6M   FMA3 PASS 9M   XLANE PASS 9M   OK
```

- `T0` -- logical thread 0
- `12M` -- millions of iterations completed
- `OK` / `FAIL` -- per-thread verdict

On failure, coreprobe automatically re-runs with the same deterministic seed to confirm reproducibility:
- **Confirmed** -- re-run also failed. Hardware defect.
- **Transient** -- not reproduced. Marginal stability.

## Core Map

After testing, a visual core map shows pass/fail status per physical core:

```
Core Map (8 physical cores, OS topology):

[00] [01] [02] [03] [04] [05] [06] [07]

[OK] = pass  [XX] = FAIL  [--] = not tested
```

Physical core topology is detected via OS APIs (Windows `GetLogicalProcessorInformationEx`, Linux sysfs) -- no hardcoded SMT assumptions.

## Design

- **Correctness, not throughput.** `-O0` and `volatile` barriers ensure every FP operation executes through hardware, not optimized away by the compiler. Pragma-based optimization guards protect test functions even under `-O2`/`-O3`/PGO.
- **Sequential per-core testing.** A faulty execution unit produces wrong results regardless of SMT contention. Sequential testing gives clean per-core fault attribution.
- **Deterministic PRNG.** xoshiro128** seeded per-test enables exact failure replay for confirmation.
- **Dual-path verification.** SSE and FMA3 tests run both `rsqrt+Newton-Raphson` (approximate) and `sqrt+div` (precise) paths on the same input, each judged against its own tolerance.
- **Cross-platform.** Windows (Win32 API) and Linux (pthreads/sysfs). Supports >64 logical processors via processor groups (Windows) and CPU_ALLOC (Linux).

## When to Use

- Validating CPU overclock / PBO stability
- Testing new hardware for silicon defects
- Diagnosing intermittent calculation failures in production
- Per-core verification after BIOS/firmware updates
- Checking silicon quality (binning)

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | All tests passed |
| `1` | Failures detected |

## License

MIT
