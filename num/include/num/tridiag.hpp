#pragma once

#include <cassert>
#include <cstddef>
#include <ranges>
#include <span>
#include <vector>

namespace num {

/// Use the Thomas algorithm to solve the linear system \f$Ax=r\f$, where
/// \f$A\f$ is a diagonally dominant tridiagonal matrix. The system should match
/// the form:
/// \f[b_ix_{i-1}+d_ix_i+a_ix_{i+1}=r_i\f]
/// All four spans—`b`, `d`, `a`, and `r`—must be of equal length `n`
/// (\f$n\ge2\f$). Elements `b[0]` and `a[n-1]` are ignored. Function runs in
/// \f$O(n)\f$.
///
/// @param  b The sub-diagonal.
/// @param  d The main diagonal.
/// @param  a The super-diagonal.
/// @param  r The right-hand side vector.
/// @return   The solution vector.
inline std::vector<double> tridiag(const std::span<const double> b,
                                   const std::span<const double> d,
                                   const std::span<const double> a,
                                   const std::span<const double> r) {
  assert((b.size() == d.size()) && (d.size() == a.size()) &&
         (a.size() == r.size()));
  const std::size_t n = b.size();
  assert(n >= 2);

  std::vector<double> ap(std::from_range, a);
  std::vector<double> rp(std::from_range, r);

  assert(d[0] != 0.0);
  ap[0] /= d[0];
  rp[0] /= d[0];

  for (std::size_t i = 1; i < n; ++i) {
    const double denom = d[i] - b[i] * ap[i - 1];
    assert(denom != 0.0);
    ap[i] /= denom;
    rp[i] -= b[i] * rp[i - 1];
    rp[i] /= denom;
  }

  for (std::size_t i = n - 1; i-- > 0;)
    rp[i] -= ap[i] * rp[i + 1];

  return rp;
}

} // namespace num
