// End-to-end search throughput for FlatIndex, on top of the SIMD kernels
// verified separately in bench_distance.cpp.
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

#include "tinyvec/core/vector_ops.hpp"
#include "tinyvec/index/flat_index.hpp"

int main() {
    constexpr unsigned kDim = 384;  // matches all-MiniLM-L6-v2 output dim
    constexpr int kN = 100000;
    constexpr std::size_t kK = 10;
    constexpr int kQueries = 200;

    std::mt19937 rng(3);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    tinyvec::FlatIndex index(kDim, tinyvec::Metric::L2);
    index.reserve(kN);
    std::vector<float> row(kDim);
    for (int i = 0; i < kN; ++i) {
        for (auto& x : row) x = dist(rng);
        index.add(static_cast<std::uint64_t>(i), row);
    }

    std::vector<std::vector<float>> queries(kQueries, std::vector<float>(kDim));
    for (auto& q : queries)
        for (auto& x : q) x = dist(rng);

    const auto start = std::chrono::steady_clock::now();
    std::size_t sink = 0;
    for (const auto& q : queries) {
        sink += index.search(q, kK).size();
    }
    const auto end = std::chrono::steady_clock::now();

    const double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    const double per_query_ms = total_ms / kQueries;

    std::printf("N=%d dim=%u k=%zu, tier=%s\n", kN, kDim, kK, tinyvec::active_simd_tier());
    std::printf("%.3f ms/query  (%.1f queries/sec)  [sink=%zu]\n", per_query_ms,
                1000.0 / per_query_ms, sink);
    return 0;
}
