#include "num/ode.hpp"
#include <array>
#include <cstddef>
#include <print>

int main() {
  const std::array<double, 2> sigma_array = {0.0, 0.5};
  const double omega = 1.0;
  const double t0 = 0.0;
  const double t1 = 32.0;
  const num::State<2> u0 = {1.0, 0.0};
  const std::array<std::size_t, 3> N_array = {21, 101, 301};

  const auto rk2 = [](auto rhs, double t, const auto &u, double dt) {
    return num::rk2_step(rhs, t, u, dt);
  };

  std::println("=== FIXED PARAMS ===");
  std::println("𝜔 = {}", omega);
  std::println("(Y₀, Ẏ₀) = ({}, {})", u0[0], u0[1]);
  std::println("t₀ = {}", t0);
  std::println("tₑ = {}", t1);
  std::println();

  for (double sigma : sigma_array) {
    for (std::size_t N : N_array) {
      const auto rhs = [sigma, omega](double,
                                      const num::State<2> &u) -> num::State<2> {
        const auto [y, ydot] = u;
        return {ydot, -sigma * ydot - omega * omega * y};
      };

      const auto sol = num::integrate(rk2, rhs, u0, t0, t1, N);

      std::println("=== RUN SUMMARY ===");
      std::println("𝜎 = {}", sigma);
      std::println("N = {}", N);
      std::println("𝛥t = {}", (t1 - t0) / (static_cast<double>(N) - 1));
      std::println("(Y(tₑ), Ẏ(tₑ)) = ({}, {})", sol.u[N - 1][0],
                   sol.u[N - 1][1]);
      std::println();
    }
  }
}
