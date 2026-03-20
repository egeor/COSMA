#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <ostream>

namespace cosma {

/// Minimal BFloat16 type for COSMA.
/// Stores 16-bit brain floating point; arithmetic promotes to float.
struct bfloat16 {
    uint16_t raw;

    constexpr bfloat16() : raw(0) {}

    /// Construct from float with round-to-nearest-even.
    bfloat16(float f) { // NOLINT(google-explicit-constructor)
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(bits));
        // round-to-nearest-even
        uint32_t rounding_bias = ((bits >> 16) & 1) + 0x7FFFu;
        raw = static_cast<uint16_t>((bits + rounding_bias) >> 16);
    }

    bfloat16(double d) : bfloat16(static_cast<float>(d)) {} // NOLINT
    bfloat16(int i) : bfloat16(static_cast<float>(i)) {}    // NOLINT

    /// Convert to float (exact, zero-extends lower 16 bits).
    operator float() const { // NOLINT(google-explicit-constructor)
        uint32_t bits = static_cast<uint32_t>(raw) << 16;
        float f;
        std::memcpy(&f, &bits, sizeof(f));
        return f;
    }

    operator double() const { return static_cast<double>(float(*this)); } // NOLINT

    // Arithmetic (promote to float, then truncate back)
    friend bfloat16 operator+(bfloat16 a, bfloat16 b) { return bfloat16(float(a) + float(b)); }
    friend bfloat16 operator-(bfloat16 a, bfloat16 b) { return bfloat16(float(a) - float(b)); }
    friend bfloat16 operator*(bfloat16 a, bfloat16 b) { return bfloat16(float(a) * float(b)); }
    friend bfloat16 operator/(bfloat16 a, bfloat16 b) { return bfloat16(float(a) / float(b)); }

    bfloat16 &operator+=(bfloat16 o) { *this = *this + o; return *this; }
    bfloat16 &operator-=(bfloat16 o) { *this = *this - o; return *this; }
    bfloat16 &operator*=(bfloat16 o) { *this = *this * o; return *this; }

    // Comparisons (bfloat16 vs bfloat16)
    friend bool operator==(bfloat16 a, bfloat16 b) { return float(a) == float(b); }
    friend bool operator!=(bfloat16 a, bfloat16 b) { return float(a) != float(b); }
    friend bool operator<(bfloat16 a, bfloat16 b)  { return float(a) <  float(b); }
    friend bool operator>(bfloat16 a, bfloat16 b)  { return float(a) >  float(b); }
    friend bool operator<=(bfloat16 a, bfloat16 b) { return float(a) <= float(b); }
    friend bool operator>=(bfloat16 a, bfloat16 b) { return float(a) >= float(b); }

    // Mixed comparisons to resolve ambiguity with int/float literals
    friend bool operator>(bfloat16 a, int b)  { return float(a) >  float(b); }
    friend bool operator<(bfloat16 a, int b)  { return float(a) <  float(b); }
    friend bool operator>=(bfloat16 a, int b) { return float(a) >= float(b); }
    friend bool operator<=(bfloat16 a, int b) { return float(a) <= float(b); }
    friend bool operator==(bfloat16 a, int b) { return float(a) == float(b); }
    friend bool operator!=(bfloat16 a, int b) { return float(a) != float(b); }

    friend bfloat16 abs(bfloat16 v) { return bfloat16(std::fabs(float(v))); }
    friend bfloat16 conj(bfloat16 v) { return v; }

    friend std::ostream &operator<<(std::ostream &os, bfloat16 v) {
        return os << float(v);
    }
};

} // namespace cosma

// Allow std::abs to work for bfloat16
namespace std {
inline cosma::bfloat16 abs(cosma::bfloat16 v) {
    return cosma::bfloat16(std::fabs(float(v)));
}

template <>
struct numeric_limits<cosma::bfloat16> {
    static constexpr bool is_specialized = true;
    static constexpr cosma::bfloat16 max() noexcept {
        cosma::bfloat16 v{};
        v.raw = 0x7F7F;
        return v; // ~3.39e38
    }
    static constexpr cosma::bfloat16 min() noexcept {
        cosma::bfloat16 v{};
        v.raw = 0x0080;
        return v; // ~1.17e-38 (smallest normal)
    }
    static constexpr cosma::bfloat16 lowest() noexcept {
        cosma::bfloat16 v{};
        v.raw = 0xFF7F;
        return v; // ~-3.39e38
    }
    static constexpr cosma::bfloat16 epsilon() noexcept {
        cosma::bfloat16 v{};
        v.raw = 0x3C00;
        return v; // 2^-7 ~ 0.0078125
    }
    static constexpr bool is_integer = false;
    static constexpr bool is_signed = true;
    static constexpr int digits = 8; // 7 mantissa + 1 implicit
};
} // namespace std
