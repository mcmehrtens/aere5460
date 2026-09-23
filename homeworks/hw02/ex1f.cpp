/// @file
///
/// AERE 5460, Homework 2, Exercise 1f. Solves the heat fin equation
/// \f$\frac{\mathrm{d}^2T}{\mathrm{d}x^2}=MT-100x^2(1-x)^2\f$ for \f$M=1,5,9\f$
/// with boundary conditions \f$T(0)=1\f$ and \f$T'(1)=0\f$. Generates a
/// solution CSV (`ex1f_M<M>_N<N>.csv`) for each run and puts it in the `data`
/// folder relative to this source file.

#include "num/csv.hpp"
#include "num/tridiag.hpp"
#include <array>
#include <cstddef>
#include <filesystem>
#include <format>
#include <print>
#include <span>
#include <string_view>
#include <vector>

int main() {
  const auto base_output_path = std::filesystem::path{HW_SOURCE_DIR} / "data";

  const std::array<double, 3> M_array = {1.0, 5.0, 9.0};
  const std::array<std::size_t, 6> N_array = {11, 21, 41, 81, 161, 321};

  constexpr std::array<std::string_view, 2> column_names{"x", "T"};

  for (double M : M_array) {
    for (std::size_t N : N_array) {
      std::vector<double> x(N), b(N), d(N), a(N), r(N);

      const double h = 1.0 / (static_cast<double>(N) - 1.0);
      const double d_val = -2.0 - M * h * h;
      const double r_val = -100.0 * h * h;

      // i = 0
      d[0] = 1.0;
      r[0] = 1.0;

      // i = 1, 2, ... , N - 2
      for (std::size_t i = 1; i < N - 1; ++i) {
        x[i] = h * static_cast<double>(i);
        b[i] = 1.0;
        d[i] = d_val;
        a[i] = 1.0;
        r[i] = r_val * x[i] * x[i] * (1 - x[i]) * (1 - x[i]);
      }

      // i = N - 1
      x[N - 1] = 1.0;
      b[N - 1] = 2.0;
      d[N - 1] = d_val;

      const auto T = num::tridiag(b, d, a, r);

      const auto output_path =
          base_output_path / std::format("ex1f_M{}_N{}.csv", M, N);
      const auto comment =
          std::format("M={} N={} h={} Tright={}", M, N, h, T[N - 1]);

      std::println("=== RUN SUMMARY ===");
      std::println("output_path = {}", output_path.string());
      std::println("M = {}", M);
      std::println("N = {}", N);
      std::println("h = {}", h);
      std::println("T(1) = {}", T[N - 1]);
      std::println();

      const std::array<std::span<const double>, 2> columns = {x, T};
      num::write_output(output_path, comment, column_names, columns);
    }
  }
}
