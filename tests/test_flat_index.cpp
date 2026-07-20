#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <utility>
#include <vector>

#include "tinyvec/index/flat_index.hpp"

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

// Independent brute-force reference, not sharing code with FlatIndex's
// internal heap logic, so a shared bug isn't masked.
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
    for (size_t i = 0; i < a.size(); ++i) acc += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    return static_cast<float>(acc);
}

float ref_cosine(const std::vector<float>& a, const std::vector<float>& b) {
    const double na = ref_dot(a, a);
    const double nb = ref_dot(b, b);
    if (na == 0.0 || nb == 0.0) return 0.0f;
    return static_cast<float>(ref_dot(a, b) / std::sqrt(na * nb));
}

}  // namespace

int main() {
    constexpr unsigned kDim = 8;

    // --- L2 nearest-neighbor ordering, with non-sequential external ids ---
    {
        tinyvec::FlatIndex index(kDim, tinyvec::Metric::L2);
        const std::vector<float> query(kDim, 0.0f);
        const std::vector<float> near(kDim, 1.0f);   // dist^2 = 8
        const std::vector<float> mid(kDim, 2.0f);    // dist^2 = 32
        const std::vector<float> far(kDim, 5.0f);    // dist^2 = 200

        index.add(100, far);
        index.add(5, near);
        index.add(42, mid);

        const auto results = index.search(query, 3);
        CHECK(results.size() == 3);
        CHECK(results[0].id == 5 && std::fabs(results[0].score - ref_l2(query, near)) < 1e-3f);
        CHECK(results[1].id == 42 && std::fabs(results[1].score - ref_l2(query, mid)) < 1e-3f);
        CHECK(results[2].id == 100 && std::fabs(results[2].score - ref_l2(query, far)) < 1e-3f);
    }

    // --- Cosine similarity ordering (descending similarity) ---
    {
        tinyvec::FlatIndex index(kDim, tinyvec::Metric::Cosine);
        const std::vector<float> query = {1, 0, 0, 0, 0, 0, 0, 0};
        const std::vector<float> same = {2, 0, 0, 0, 0, 0, 0, 0};       // cos = 1
        const std::vector<float> orth = {0, 1, 0, 0, 0, 0, 0, 0};       // cos = 0
        const std::vector<float> opp = {-1, 0, 0, 0, 0, 0, 0, 0};       // cos = -1

        index.add(1, opp);
        index.add(2, orth);
        index.add(3, same);

        const auto results = index.search(query, 3);
        CHECK(results.size() == 3);
        CHECK(results[0].id == 3);
        CHECK(std::fabs(results[0].score - 1.0f) < 1e-3f);
        CHECK(results[1].id == 2);
        CHECK(std::fabs(results[1].score - 0.0f) < 1e-3f);
        CHECK(results[2].id == 1);
        CHECK(std::fabs(results[2].score - (-1.0f)) < 1e-3f);
    }

    // --- k larger than size() returns everything, sorted ---
    {
        tinyvec::FlatIndex index(kDim, tinyvec::Metric::L2);
        const std::vector<float> query(kDim, 0.0f);
        index.add(1, std::vector<float>(kDim, 1.0f));
        index.add(2, std::vector<float>(kDim, 2.0f));
        const auto results = index.search(query, 100);
        CHECK(results.size() == 2);
        CHECK(results[0].id == 1);
        CHECK(results[1].id == 2);
    }

    // --- Empty index / k == 0 ---
    {
        tinyvec::FlatIndex index(kDim, tinyvec::Metric::L2);
        const std::vector<float> query(kDim, 0.0f);
        CHECK(index.search(query, 5).empty());
        index.add(1, std::vector<float>(kDim, 1.0f));
        CHECK(index.search(query, 0).empty());
    }

    // --- size()/dim() and reserve() don't disturb behavior ---
    {
        tinyvec::FlatIndex index(kDim, tinyvec::Metric::L2);
        index.reserve(10);
        CHECK(index.size() == 0);
        CHECK(index.dim() == kDim);
        index.add(7, std::vector<float>(kDim, 3.0f));
        CHECK(index.size() == 1);
    }

    // --- Larger randomized cross-check against brute-force reference ---
    {
        constexpr unsigned dim = 64;
        constexpr int n = 200;
        std::mt19937 rng(11);
        std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

        tinyvec::FlatIndex index(dim, tinyvec::Metric::L2);
        std::vector<std::vector<float>> vecs(n);
        for (int i = 0; i < n; ++i) {
            vecs[i].resize(dim);
            for (auto& x : vecs[i]) x = dist(rng);
            index.add(static_cast<std::uint64_t>(i), vecs[i]);
        }

        std::vector<float> query(dim);
        for (auto& x : query) x = dist(rng);

        const auto results = index.search(query, 5);
        CHECK(results.size() == 5);

        // Verify against a full independent brute-force top-5.
        std::vector<std::pair<float, int>> all;
        for (int i = 0; i < n; ++i) all.push_back({ref_l2(query, vecs[i]), i});
        std::sort(all.begin(), all.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        for (int i = 0; i < 5; ++i) {
            CHECK(results[i].id == static_cast<std::uint64_t>(all[i].second));
            CHECK(std::fabs(results[i].score - all[i].first) < 1e-1f);
        }
    }

    std::printf("all flat_index checks passed\n");
    return 0;
}
