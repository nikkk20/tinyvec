#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "tinyvec/core/vector_ops.hpp"

// Not a plain assert(): asserts compile out under NDEBUG/Release, and this
// suite must still catch regressions in optimized builds.
#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                      \
            std::fprintf(stderr, "CHECK FAILED: %s (%s:%d)\n", #cond,        \
                          __FILE__, __LINE__);                               \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

namespace {

// Independent double-precision reference, deliberately not sharing code with
// any shipped kernel (scalar included) so a bug common to both isn't masked.
float ref_l2(const std::vector<float>& a, const std::vector<float>& b) {
    double acc = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        acc += d * d;
    }
    return static_cast<float>(acc);
}

float ref_dot(const std::vector<float>& a, const std::vector<float>& b) {
    double acc = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        acc += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return static_cast<float>(acc);
}

float ref_cosine(const std::vector<float>& a, const std::vector<float>& b) {
    const double na = ref_dot(a, a);
    const double nb = ref_dot(b, b);
    if (na == 0.0 || nb == 0.0) return 0.0f;
    return static_cast<float>(ref_dot(a, b) / std::sqrt(na * nb));
}

std::vector<float> random_vec(std::mt19937& rng, unsigned dim) {
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    std::vector<float> v(dim);
    for (auto& x : v) x = dist(rng);
    return v;
}

}  // namespace

int main() {
    std::printf("active SIMD tier: %s\n", tinyvec::active_simd_tier());

    std::mt19937 rng(42);
    const unsigned dims[] = {1, 3, 7, 8, 15, 16, 33, 128, 384, 1536};
    // Relative tolerance: SIMD FMA accumulation order differs from the
    // scalar double reference, so exact equality isn't expected.
    const float kTol = 1e-2f;

    for (unsigned dim : dims) {
        const auto a = random_vec(rng, dim);
        const auto b = random_vec(rng, dim);
        tinyvec::VectorView va(a), vb(b);

        const float got_l2 = tinyvec::l2_distance_squared(va, vb);
        const float want_l2 = ref_l2(a, b);
        CHECK(std::fabs(got_l2 - want_l2) <= kTol * std::max(1.0f, want_l2));

        const float got_dot = tinyvec::dot_product(va, vb);
        const float want_dot = ref_dot(a, b);
        CHECK(std::fabs(got_dot - want_dot) <= kTol * std::max(1.0f, std::fabs(want_dot)));

        const float got_cos = tinyvec::cosine_similarity(va, vb);
        const float want_cos = ref_cosine(a, b);
        CHECK(std::fabs(got_cos - want_cos) <= kTol);
    }

    // Zero-vector cosine edge case must not divide by zero / produce NaN.
    {
        const std::vector<float> zero(128, 0.0f);
        const auto other = random_vec(rng, 128);
        tinyvec::VectorView vzero(zero), vother(other);
        CHECK(tinyvec::cosine_similarity(vzero, vother) == 0.0f);
        CHECK(tinyvec::cosine_similarity(vzero, vzero) == 0.0f);
    }

    // Identical vectors: L2 == 0, cosine == 1, dot == squared norm.
    {
        const auto v = random_vec(rng, 384);
        tinyvec::VectorView vv(v);
        CHECK(std::fabs(tinyvec::l2_distance_squared(vv, vv)) <= kTol);
        CHECK(std::fabs(tinyvec::cosine_similarity(vv, vv) - 1.0f) <= kTol);
    }

    std::printf("all checks passed\n");
    return 0;
}
