#include <immintrin.h>

#include "kernels.hpp"

namespace {

float hsum256(__m256 v) {
    const __m128 lo = _mm256_castps256_ps128(v);
    const __m128 hi = _mm256_extractf128_ps(v, 1);
    const __m128 sum4 = _mm_add_ps(lo, hi);
    const __m128 sum2 = _mm_add_ps(sum4, _mm_movehl_ps(sum4, sum4));
    const __m128 sum1 = _mm_add_ss(sum2, _mm_shuffle_ps(sum2, sum2, 0x1));
    return _mm_cvtss_f32(sum1);
}

}  // namespace

namespace tinyvec::kernels {

float l2_avx2(const float* a, const float* b, unsigned n) {
    __m256 acc = _mm256_setzero_ps();
    unsigned i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 va = _mm256_loadu_ps(a + i);
        const __m256 vb = _mm256_loadu_ps(b + i);
        const __m256 d = _mm256_sub_ps(va, vb);
        acc = _mm256_fmadd_ps(d, d, acc);
    }
    float total = hsum256(acc);
    for (; i < n; ++i) {
        const float d = a[i] - b[i];
        total += d * d;
    }
    return total;
}

float dot_avx2(const float* a, const float* b, unsigned n) {
    __m256 acc = _mm256_setzero_ps();
    unsigned i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 va = _mm256_loadu_ps(a + i);
        const __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    float total = hsum256(acc);
    for (; i < n; ++i) {
        total += a[i] * b[i];
    }
    return total;
}

}  // namespace tinyvec::kernels
