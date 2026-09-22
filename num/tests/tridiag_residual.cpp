#include "num/tridiag.hpp"
#include <cstddef>
#include <cstdlib>
#include <print>
#include <random>
#include <vector>

int main() {
  const std::size_t n = 6;
  const double tol = 1e-12;
  std::mt19937 g(1);
  std::uniform_real_distribution<double> u(-1, 1);
  std::vector<double> b(n), d(n), a(n), r(n);
  for (std::size_t i = 0; i < n; ++i) {
    b[i] = u(g);
    a[i] = u(g);
    r[i] = u(g);
    d[i] = 4 + u(g); // necessary for diagonal dominance
  }
  b[0] = 0;     // not necessary, but convenient for subsequent loop
  a[n - 1] = 0; // also not necessary for same
  auto x = num::tridiag(b, d, a, r);

  bool ok = true;
  for (std::size_t i = 0; i < n; ++i) {
    double res = (i > 0 ? b[i] * x[i - 1] : 0.0) + d[i] * x[i] +
                 (i < n - 1 ? a[i] * x[i + 1] : 0.0) - r[i];
    if (std::abs(res) > tol) {
      ok = false;
      std::println("row {}: residual {:.3e}", i, res);
    }
  }
  return ok ? 0 : 1;
}
