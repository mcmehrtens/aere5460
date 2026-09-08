/// @file
///
/// AERE 5460, Homework 1, Exercise 3. Solves a damped oscillator Y'' + σY' +
/// ω²Y = 0 with 2nd-order Adams–Bashforth using 2nd-order Runge–Kutta for
/// startup. Solves with sigma 0.0, 0.5 and N 21, 101, 301. Generates a solution
/// CSV (`ex3_ab2_sigma<σ>_N<N>.csv`) for each run and puts it in the `data`
/// folder relative to this source file.

#include "num/csv.hpp"
#include "num/ode.hpp"
#include <array>
#include <cstddef>
#include <filesystem>
#include <format>
#include <print>
#include <string_view>

int main() {
  const auto base_output_path = std::filesystem::path{HW01_SOURCE_DIR} / "data";

  const std::array<double, 2> sigma_array = {0.0, 0.5};
  const double omega = 1.0;
  const double t0 = 0.0;
  const double t1 = 32.0;
  const num::State<2> u0 = {1.0, 0.0};
  const std::array<std::size_t, 3> N_array = {21, 101, 301};

  std::println("=== FIXED PARAMS ===");
  std::println("𝜔 = {}", omega);
  std::println("(Y₀, Ẏ₀) = ({}, {})", u0[0], u0[1]);
  std::println("t₀ = {}", t0);
  std::println("tₑ = {}", t1);
  std::println();

  constexpr std::array<std::string_view, 3> cols{"t", "Y", "Ydot"};

  for (double sigma : sigma_array) {
    for (std::size_t N : N_array) {
      const auto rhs = [sigma, omega](double,
                                      const num::State<2> &u) -> num::State<2> {
        const auto [y, ydot] = u;
        return {ydot, -sigma * ydot - omega * omega * y};
      };

      const auto sol = num::ab2(rhs, u0, t0, t1, N);

      const double dt = (t1 - t0) / (static_cast<double>(N) - 1);
      const auto output_path =
          base_output_path / std::format("ex3_ab2_sigma{}_N{}.csv", sigma, N);
      const auto comment =
          std::format("method={} sigma={} omega={} t0={} t1={} N={} dt={}",
                      "ab2", sigma, omega, t0, t1, N, dt);

      std::println("=== RUN SUMMARY ===");
      std::println("output_path = {}", output_path.string());
      std::println("𝜎 = {}", sigma);
      std::println("N = {}", N);
      std::println("𝛥t = {}", dt);
      std::println("(Y(tₑ), Ẏ(tₑ)) = ({}, {})", sol.u[N - 1][0],
                   sol.u[N - 1][1]);
      std::println();

      num::write_output(output_path, comment, cols, sol);
    }
  }
}
