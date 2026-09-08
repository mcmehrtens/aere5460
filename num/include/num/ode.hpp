#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <vector>

namespace num {

/// A fixed-size vector of `Dim` doubles representing the ODE state.
template <std::size_t Dim> using State = std::array<double, Dim>;

/// Compute ax + y where x and y have size Dim. Useful for performing operations
/// over all state variables.
///
/// @tparam Dim The number of state variables.
/// @param  a   The scalar multiplier.
/// @param  x   The x state variables.
/// @param  y   The y state variables.
/// @return     The result of ax + y.
template <std::size_t Dim>
State<Dim> axpy(double a, const State<Dim> &x, const State<Dim> &y) {
  State<Dim> result{};
  for (std::size_t i = 0; i < Dim; ++i)
    result[i] = a * x[i] + y[i];
  return result;
}

/// Integrate a single ODE step with 2nd-order Runge–Kutta (midpoint variant).
/// This method requires two evaluations of RHS per step.
///
/// @tparam Dim The number of state variables.
/// @tparam RHS The RHS function with expected signature `State<Dim> RHS(double
///             t, const State<Dim> &u)`.
/// @param  rhs The right-hand-side callable evaluating du/dt.
/// @param  t   The current simulation time.
/// @param  u   The current state variables.
/// @param  dt  The size of step to integrate over.
/// @return     The computed state variables at time t = t + dt.
template <std::size_t Dim, typename RHS>
State<Dim> rk2_step(RHS rhs, double t, const State<Dim> &u, double dt) {
  const double dt_half = 0.5 * dt;
  const State<Dim> u_mid = axpy(dt_half, rhs(t, u), u);
  return axpy(dt, rhs(t + dt_half, u_mid), u);
}

/// The solution data structure. `t` and `u` are parallel vectors of the same
/// length.
///
/// @tparam Dim The number of state variables.
template <std::size_t Dim> struct Solution {
  std::vector<double> t;     ///< The time vector.
  std::vector<State<Dim>> u; ///< The state variable vector.
};

/// Integrate `Dim` ODEs with RHS `RHS` from time `t0` to time `t1` with
/// `n_points - 1` steps of stepper `Stepper` and initial conditions `u0`. dt
/// for each step is computed as `(t1 - t0) / (n_points - 1)`.
///
/// @tparam Dim       The number of state variables.
/// @tparam Stepper   The stepping function with expected signature `State<Dim>
///                   Stepper(RHS rhs, double t, const State<Dim> &u, double
///                   dt)`.
/// @tparam RHS       The RHS function with expected signature `State<Dim>
///                   RHS(double t, const State<Dim> &u)`.
/// @param  step      The stepper algorithm callable evaluating u(t + dt) from
///                   u(t).
/// @param  rhs       The right-hand-side callable evaluating du/dt.
/// @param  u0        The initial conditions, i.e., u(t0).
/// @param  t0        The initial simulation time.
/// @param  t1        The final simulation time.
/// @param  n_points  The number of time points, including `t0` and `t1`; at
///                   least 2.
/// @return           The parallel solution vectors for the time `t` and the
///                   state variables `u` at each time `t`.
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
