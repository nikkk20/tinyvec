#include <immintrin.h>

#include "kernels.hpp"

namespace tinyvec::kernels {

float l2_avx512(const float* a, const float* b, unsigned n) {
    __m512 acc = _mm512_setzero_ps();
    unsigned i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m512 va = _mm512_loadu_ps(a + i);
        const __m512 vb = _mm512_loadu_ps(b + i);
        const __m512 d = _mm512_sub_ps(va, vb);
        acc = _mm512_fmadd_ps(d, d, acc);
    }
    float total = _mm512_reduce_add_ps(acc);
    for (; i < n; ++i) {
        const float d = a[i] - b[i];
        total += d * d;
    }
    return total;
}

float dot_avx512(const float* a, const float* b, unsigned n) {
    __m512 acc = _mm512_setzero_ps();
    unsigned i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m512 va = _mm512_loadu_ps(a + i);
        const __m512 vb = _mm512_loadu_ps(b + i);
        acc = _mm512_fmadd_ps(va, vb, acc);
    }
    float total = _mm512_reduce_add_ps(acc);
    for (; i < n; ++i) {
        total += a[i] * b[i];
    }
    return total;
}

}  // namespace tinyvec::kernels
