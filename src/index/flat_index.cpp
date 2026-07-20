#include "tinyvec/index/flat_index.hpp"

#include <algorithm>
#include <cassert>
#include <queue>

#include "tinyvec/core/vector_ops.hpp"

namespace tinyvec {

FlatIndex::FlatIndex(dim_t dim, Metric metric) : dim_(dim), metric_(metric) {}

void FlatIndex::reserve(std::size_t n) {
    data_.reserve(n * dim_);
    ids_.reserve(n);
}

void FlatIndex::add(std::uint64_t id, VectorView vec) {
    assert(vec.size() == dim_);
    data_.insert(data_.end(), vec.begin(), vec.end());
    ids_.push_back(id);
}

std::size_t FlatIndex::size() const noexcept { return ids_.size(); }

dim_t FlatIndex::dim() const noexcept { return dim_; }

namespace {

struct Candidate {
    float comparable;  // higher is always better, regardless of metric
    float score;       // metric-native value returned to the caller
    std::uint64_t id;
};

// std::priority_queue is a max-heap by default; inverting the comparison
// turns it into a min-heap so heap.top() is always the weakest of the
// current top-k, the one to evict when a better candidate shows up.
struct MinByComparable {
    bool operator()(const Candidate& a, const Candidate& b) const {
        return a.comparable > b.comparable;
    }
};

}  // namespace

std::vector<SearchResult> FlatIndex::search(VectorView query, std::size_t k) const {
    assert(query.size() == dim_);
    if (k == 0 || ids_.empty()) return {};

    std::priority_queue<Candidate, std::vector<Candidate>, MinByComparable> heap;

    for (std::size_t i = 0; i < ids_.size(); ++i) {
        const VectorView row(data_.data() + i * dim_, dim_);

        float score;
        float comparable;
        switch (metric_) {
            case Metric::L2:
                score = l2_distance_squared(query, row);
                comparable = -score;  // smaller distance -> larger comparable
                break;
            case Metric::Cosine:
                score = cosine_similarity(query, row);
                comparable = score;
                break;
            case Metric::Dot:
                score = dot_product(query, row);
                comparable = score;
                break;
        }

        if (heap.size() < k) {
            heap.push({comparable, score, ids_[i]});
        } else if (comparable > heap.top().comparable) {
            heap.pop();
            heap.push({comparable, score, ids_[i]});
        }
    }

    std::vector<SearchResult> results;
    results.reserve(heap.size());
    while (!heap.empty()) {
        results.push_back({heap.top().id, heap.top().score});
        heap.pop();
    }
    std::reverse(results.begin(), results.end());  // heap yields weakest-first; flip to best-first
    return results;
}

}  // namespace tinyvec
