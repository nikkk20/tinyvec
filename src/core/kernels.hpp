#pragma once

// Internal per-tier kernel declarations. Not part of the public API in
// include/ — callers use tinyvec::l2_distance_squared etc. from
// vector_ops.hpp, which dispatch to whichever of these is selected at
// startup based on detected CPU features.

namespace tinyvec::kernels {

float l2_scalar(const float* a, const float* b, unsigned n);
float dot_scalar(const float* a, const float* b, unsigned n);

float l2_avx2(const float* a, const float* b, unsigned n);
float dot_avx2(const float* a, const float* b, unsigned n);

float l2_avx512(const float* a, const float* b, unsigned n);
float dot_avx512(const float* a, const float* b, unsigned n);

}  // namespace tinyvec::kernels
