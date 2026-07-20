#pragma once

#include "tinyvec/core/types.hpp"

namespace tinyvec {

// Squared L2 distance. Precondition: a.size() == b.size().
float l2_distance_squared(VectorView a, VectorView b);

// Inner product. Precondition: a.size() == b.size().
float dot_product(VectorView a, VectorView b);

// Cosine similarity in [-1, 1]. Returns 0 if either vector is all-zero.
// Precondition: a.size() == b.size().
float cosine_similarity(VectorView a, VectorView b);

// Name of the SIMD tier selected at startup ("avx512", "avx2", or "scalar").
// Exposed for benchmarking/diagnostics.
const char* active_simd_tier();

}  // namespace tinyvec
