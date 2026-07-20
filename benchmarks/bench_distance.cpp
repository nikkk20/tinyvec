// Compares scalar vs AVX2 vs AVX-512 kernel throughput directly (bypassing
// runtime dispatch) to verify the SIMD work actually pays off. This reaches
// into src/core/ internals on purpose - a benchmark, unlike the library
// itself, is allowed to assume it's run on the machine it was built for.
#include <chrono>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "kernels.hpp"
#include "tinyvec/core/vector_ops.hpp"

namespace {

using Clock = std::chrono::steady_clock;

template <typename Fn>
double bench_ns_per_call(Fn&& fn, int iters) {
    const auto start = Clock::now();
    for (int i = 0; i < iters; ++i) fn();
    const auto end = Clock::now();
    return std::chrono::duration<double, std::nano>(end - start).count() /
           static_cast<double>(iters);
}

}  // namespace

int main() {
    constexpr unsigned kDim = 384;  // matches all-MiniLM-L6-v2 output dim
    constexpr int kIters = 200000;

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    std::vector<float> a(kDim), b(kDim);
    for (auto& x : a) x = dist(rng);
    for (auto& x : b) x = dist(rng);

    const std::string tier = tinyvec::active_simd_tier();
    std::printf("dim=%u iters=%d, detected tier=%s\n\n", kDim, kIters, tier.c_str());

    volatile float sink = 0.0f;  // prevents the optimizer from deleting the loop

    const double scalar_ns = bench_ns_per_call(
        [&] { sink = tinyvec::kernels::l2_scalar(a.data(), b.data(), kDim); }, kIters);
    std::printf("l2_scalar:  %8.2f ns/call\n", scalar_ns);

    if (tier == "avx2" || tier == "avx512") {
        const double avx2_ns = bench_ns_per_call(
            [&] { sink = tinyvec::kernels::l2_avx2(a.data(), b.data(), kDim); }, kIters);
        std::printf("l2_avx2:    %8.2f ns/call  (%.2fx vs scalar)\n", avx2_ns,
                     scalar_ns / avx2_ns);
    }

    if (tier == "avx512") {
        const double avx512_ns = bench_ns_per_call(
            [&] { sink = tinyvec::kernels::l2_avx512(a.data(), b.data(), kDim); }, kIters);
        std::printf("l2_avx512:  %8.2f ns/call  (%.2fx vs scalar)\n", avx512_ns,
                     scalar_ns / avx512_ns);
    }

    return 0;
}
