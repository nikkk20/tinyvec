#include <cassert>
#include <cmath>

#include "kernels.hpp"
#include "tinyvec/core/vector_ops.hpp"

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif

namespace tinyvec {

namespace {

enum class SimdTier { Scalar, Avx2, Avx512 };

#if defined(_MSC_VER)

// MSVC has no __builtin_cpu_supports, so do CPUID + XGETBV by hand.
// XGETBV is required in addition to CPUID feature bits: CPUID can report a
// feature is present in silicon while the OS hasn't opted into saving its
// register state (XCR0), in which case using it is a fault waiting to
// happen on real hardware/VMs with the state trimmed.
bool os_supports_avx() {
    int regs[4];
    __cpuid(regs, 1);
    const bool osxsave = (regs[2] & (1 << 27)) != 0;
    const bool avx = (regs[2] & (1 << 28)) != 0;
    if (!osxsave || !avx) return false;
    const unsigned long long xcr0 = _xgetbv(0);
    return (xcr0 & 0x6) == 0x6;  // XMM + YMM state
}

bool os_supports_avx512() {
    if (!os_supports_avx()) return false;
    const unsigned long long xcr0 = _xgetbv(0);
    return (xcr0 & 0xE6) == 0xE6;  // + opmask, ZMM_Hi256, Hi16_ZMM state
}

SimdTier detect_tier() {
    int regs[4];
    __cpuid(regs, 0);
    const int max_leaf = regs[0];
    if (max_leaf < 7) return SimdTier::Scalar;

    __cpuid(regs, 1);
    const bool fma = (regs[2] & (1 << 12)) != 0;

    int regs7[4];
    __cpuidex(regs7, 7, 0);
    const bool avx2 = (regs7[1] & (1 << 5)) != 0;
    const bool avx512f = (regs7[1] & (1 << 16)) != 0;

    if (avx512f && os_supports_avx512()) return SimdTier::Avx512;
    if (avx2 && fma && os_supports_avx()) return SimdTier::Avx2;
    return SimdTier::Scalar;
}

#else  // GCC / Clang: __builtin_cpu_supports already accounts for OSXSAVE/XGETBV.

SimdTier detect_tier() {
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512f")) return SimdTier::Avx512;
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) return SimdTier::Avx2;
    return SimdTier::Scalar;
}

#endif

using L2Fn = float (*)(const float*, const float*, unsigned);
using DotFn = float (*)(const float*, const float*, unsigned);

struct DispatchTable {
    SimdTier tier;
    L2Fn l2;
    DotFn dot;

    DispatchTable() : tier(detect_tier()) {
        switch (tier) {
            case SimdTier::Avx512:
                l2 = kernels::l2_avx512;
                dot = kernels::dot_avx512;
                break;
            case SimdTier::Avx2:
                l2 = kernels::l2_avx2;
                dot = kernels::dot_avx2;
                break;
            case SimdTier::Scalar:
            default:
                l2 = kernels::l2_scalar;
                dot = kernels::dot_scalar;
                break;
        }
    }
};

const DispatchTable& table() {
    static const DispatchTable instance;
    return instance;
}

}  // namespace

float l2_distance_squared(VectorView a, VectorView b) {
    assert(a.size() == b.size());
    return table().l2(a.data(), b.data(), static_cast<unsigned>(a.size()));
}

float dot_product(VectorView a, VectorView b) {
    assert(a.size() == b.size());
    return table().dot(a.data(), b.data(), static_cast<unsigned>(a.size()));
}

float cosine_similarity(VectorView a, VectorView b) {
    assert(a.size() == b.size());
    const float na = table().dot(a.data(), a.data(), static_cast<unsigned>(a.size()));
    const float nb = table().dot(b.data(), b.data(), static_cast<unsigned>(b.size()));
    if (na == 0.0f || nb == 0.0f) return 0.0f;
    const float d = table().dot(a.data(), b.data(), static_cast<unsigned>(a.size()));
    return d / std::sqrt(na * nb);
}

const char* active_simd_tier() {
    switch (table().tier) {
        case SimdTier::Avx512: return "avx512";
        case SimdTier::Avx2: return "avx2";
        default: return "scalar";
    }
}

}  // namespace tinyvec
