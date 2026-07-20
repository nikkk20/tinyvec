#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "tinyvec/core/types.hpp"

namespace tinyvec {

struct SearchResult {
    std::uint64_t id;
    // L2: squared distance (smaller = closer). Cosine/Dot: similarity
    // (larger = closer). Results are always ordered nearest/most-similar
    // first regardless of metric.
    float score;
};

// Exact brute-force nearest-neighbor index. Stores vectors contiguously and
// scans them in full on every query using the SIMD-dispatched kernels from
// vector_ops.hpp.
class FlatIndex {
public:
    FlatIndex(dim_t dim, Metric metric);

    // Capacity hint to avoid reallocation churn when the final count is
    // known ahead of time.
    void reserve(std::size_t n);

    // Precondition: vec.size() == dim().
    void add(std::uint64_t id, VectorView vec);

    // Returns up to k nearest neighbors, nearest first. Precondition:
    // query.size() == dim().
    std::vector<SearchResult> search(VectorView query, std::size_t k) const;

    std::size_t size() const noexcept;
    dim_t dim() const noexcept;

private:
    dim_t dim_;
    Metric metric_;
    std::vector<float> data_;         // flat buffer, row i at [i*dim_, (i+1)*dim_)
    std::vector<std::uint64_t> ids_;  // ids_[i] is the external id for row i
};

}  // namespace tinyvec
