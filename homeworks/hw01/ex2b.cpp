/// @file
///
/// AERE 5460, Homework 1, Exercise 2b. Solves a nonlinear spring (without
/// damping) modeled by Y'' + Y - BY^3 = 0 with 2nd-order Runge–Kutta (midpoint
/// variant). Solves with B 0.2, 0.6, 0.9, 0.999 and N sufficiently large,
/// dependent on B. Generates a solution CSV (`ex2b_rk2_B<B>_N<N>.csv`) for each
/// run and puts it in the `data` folder relative to this source file.

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

  const std::array<double, 4> B_array = {0.2, 0.6, 0.9, 0.999};
  const double t0 = 0.0;
  const double t1 = 32.0;
  const num::State<2> u0 = {1.0, 0.0};
  const std::array<std::size_t, 4> N_array = {100, 1000, 10000, 100000};

  const auto rk2 = [](auto rhs, double t, const auto &u, double dt) {
    return num::rk2_step(rhs, t, u, dt);
  };

  std::println("=== FIXED PARAMS ===");
  std::println("(Y₀, Ẏ₀) = ({}, {})", u0[0], u0[1]);
  std::println("t₀ = {}", t0);
  std::println("tₑ = {}", t1);
  std::println();

  constexpr std::array<std::string_view, 3> cols{"t", "Y", "Ydot"};

  for (double B : B_array) {
    for (std::size_t N : N_array) {
      const auto rhs = [B](double, const num::State<2> &u) -> num::State<2> {
        const auto [y, ydot] = u;
        return {ydot, y * (B * y * y - 1)};
      };

      const auto sol = num::integrate(rk2, rhs, u0, t0, t1, N);

      const double dt = (t1 - t0) / (static_cast<double>(N) - 1);
      const auto output_path =
          base_output_path / std::format("ex2b_rk2_B{}_N{}.csv", B, N);
      const auto comment = std::format("method={} B={} t0={} t1={} N={} dt={}",
                                       "rk2", B, t0, t1, N, dt);

      std::println("=== RUN SUMMARY ===");
      std::println("output_path = {}", output_path.string());
      std::println("B = {}", B);
      std::println("N = {}", N);
      std::println("𝛥t = {}", dt);
      std::println("(Y(tₑ), Ẏ(tₑ)) = ({}, {})", sol.u[N - 1][0],
                   sol.u[N - 1][1]);
      std::println();

      num::write_output(output_path, comment, cols, sol);
    }
  }
}
