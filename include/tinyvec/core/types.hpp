#pragma once

#include <cstdint>
#include <span>

namespace tinyvec {

using dim_t = std::uint32_t;
using VectorView = std::span<const float>;

enum class Metric { L2, Cosine, Dot };

}  // namespace tinyvec
