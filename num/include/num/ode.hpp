#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <vector>

namespace num {

template <std::size_t Dim> using State = std::array<double, Dim>;

template <std::size_t Dim>
State<Dim> axpy(double a, const State<Dim> &x, const State<Dim> &y) {
  State<Dim> result{};
  for (std::size_t i = 0; i < Dim; ++i)
    result[i] = a * x[i] + y[i];
  return result;
}

template <std::size_t Dim, typename RHS>
State<Dim> rk2_step(RHS rhs, double t, const State<Dim> &u, double dt) {
  const double dt_half = 0.5 * dt;
  const State<Dim> u_mid = axpy(dt_half, rhs(t, u), u);
  return axpy(dt, rhs(t + dt_half, u_mid), u);
}

template <std::size_t Dim> struct Solution {
  std::vector<double> t;
  std::vector<State<Dim>> u;
};

template <std::size_t Dim, typename Stepper, typename RHS>
Solution<Dim> integrate(Stepper step, RHS rhs, const State<Dim> &u0, double t0,
                        double t1, std::size_t n_points) {
  assert(n_points >= 2);

  const double dt = (t1 - t0) / static_cast<double>(n_points - 1);
  Solution<Dim> sol{std::vector<double>(n_points),
                    std::vector<State<Dim>>(n_points)};
  sol.t[0] = t0;
  sol.u[0] = u0;
  for (std::size_t n = 1; n < n_points; ++n) {
    sol.t[n] = t0 + static_cast<double>(n) * dt;
    sol.u[n] = step(rhs, sol.t[n - 1], sol.u[n - 1], dt);
  }
  return sol;
}

} // namespace num
