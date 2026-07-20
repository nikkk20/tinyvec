#include "kernels.hpp"

namespace tinyvec::kernels {

float l2_scalar(const float* a, const float* b, unsigned n) {
    float acc = 0.0f;
    for (unsigned i = 0; i < n; ++i) {
        const float d = a[i] - b[i];
        acc += d * d;
    }
    return acc;
}

float dot_scalar(const float* a, const float* b, unsigned n) {
    float acc = 0.0f;
    for (unsigned i = 0; i < n; ++i) {
        acc += a[i] * b[i];
    }
    return acc;
}

}  // namespace tinyvec::kernels
