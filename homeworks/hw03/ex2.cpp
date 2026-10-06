/// @file
///
/// AERE 5460, Homework 3, Exercise 2. Solves the 1D diffusion equation with a
/// temperature-dependent diffusivity, \f$\frac{\partial T}{\partial
/// t}=-\frac{\partial F}{\partial x}\f$ with \f$F=-\kappa\frac{\partial
/// T}{\partial x}\f$ and \f$\kappa=0.1+0.1e^T\f$, on an insulated slab. Uses a
/// cell-centered finite-volume discretization in space and RK2 in time.

#include "num/csv.hpp"
#include "num/ode.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <format>
#include <numbers>
#include <numeric>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <vector>

int main() {
  const auto base_output_path = std::filesystem::path{HW_SOURCE_DIR} / "data";

  // again, I do calculate these time bases to get clean floats in my output
  constexpr std::size_t bases_per_unit_time = 100;
  constexpr double t_base = 1.0 / static_cast<double>(bases_per_unit_time);
  constexpr std::size_t bases_final = 5; // tf = 0.05

  // Nx faces bound Nx - 1 cells; cell i spans faces i and i + 1
  constexpr std::size_t Nx = 201;
  constexpr std::size_t Ncell = Nx - 1;
  constexpr double dx = 1.0 / static_cast<double>(Ncell);
  constexpr double sigma = 0.1;

  constexpr double alpha_target = 0.25;

  constexpr std::size_t n_hist = 1000;
  constexpr std::array<std::string_view, 2> hist_column_names{"t", "I"};

  const auto x = [] {
    num::State<Ncell> g{};
    for (std::size_t i = 0; i < Ncell; ++i)
      g[i] = (static_cast<double>(i) + 0.5) * dx;
    return g;
  }();

  const auto T0 = [&x] {
    num::State<Ncell> T{};
    for (std::size_t i = 0; i < Ncell; ++i) {
      const double s = (x[i] - 0.5) / sigma;
      T[i] = std::exp(-s * s) * std::numbers::inv_sqrtpi / sigma;
    }
    return T;
  }();

  const auto kappa = [](double T) { return 0.1 * (1.0 + std::exp(T)); };
  const double kappa_max = kappa(std::ranges::max(T0));

  // dT_i/dt = -(F_{i+1} - F_i) / dx, with F_0 = F_{Nx-1} = 0 at the
  // insulated ends
  const auto rhs = [&kappa](double, const num::State<Ncell> &T) {
    num::State<Ncell> k{};
    for (std::size_t i = 0; i < Ncell; ++i)
      k[i] = kappa(T[i]);

    std::array<double, Nx> F{};
    for (std::size_t i = 1; i < Nx - 1; ++i) {
      const double kf = 0.5 * (k[i - 1] + k[i]);
      F[i] = -kf * (T[i] - T[i - 1]) / dx;
    }

    num::State<Ncell> dTdt{};
    for (std::size_t i = 0; i < Ncell; ++i)
      dTdt[i] = -(F[i + 1] - F[i]) / dx;
    return dTdt;
  };

  const auto integral = [](const num::State<Ncell> &T) {
    return std::accumulate(T.begin(), T.end(), 0.0) * dx;
  };

  // the views passed to write_output need owning storage that outlives them
  const auto write_profiles =
      [&x](const std::filesystem::path &path, std::string_view comment,
           const std::vector<std::string> &names,
           const std::vector<num::State<Ncell>> &profiles) {
        std::vector<std::string_view> column_names{"x"};
        std::vector<std::span<const double>> columns{x};
        for (std::size_t k = 0; k < names.size(); ++k) {
          column_names.emplace_back(names[k]);
          columns.emplace_back(profiles[k]);
        }
        num::write_output(path, comment, column_names, columns);
      };

  const auto steps_per_base = static_cast<std::size_t>(
      std::ceil(t_base * kappa_max / (alpha_target * dx * dx)));
  const std::size_t steps_per_unit_time = bases_per_unit_time * steps_per_base;
  const double dt = 1.0 / static_cast<double>(steps_per_unit_time);
  const double alpha = kappa_max * dt / (dx * dx);

  const std::size_t n_steps = bases_final * steps_per_base;
  const std::size_t steps_per_hist = std::max<std::size_t>(1, n_steps / n_hist);

  std::vector<double> t_snap{0.0};
  std::vector<num::State<Ncell>> T_snap{T0};
  t_snap.reserve(bases_final + 1);
  T_snap.reserve(bases_final + 1);

  std::vector<double> t_hist{0.0};
  std::vector<double> I_hist{integral(T0)};
  t_hist.reserve(n_hist + 2);
  I_hist.reserve(n_hist + 2);

  num::State<Ncell> T = T0;
  for (std::size_t n = 1; n <= n_steps; ++n) {
    const double t_prev =
        static_cast<double>(n - 1) / static_cast<double>(steps_per_unit_time);
    T = num::rk2_step(rhs, t_prev, T, dt);
    const double t =
        static_cast<double>(n) / static_cast<double>(steps_per_unit_time);

    if (n % steps_per_hist == 0 || n == n_steps) {
      t_hist.push_back(t);
      I_hist.push_back(integral(T));
    }
    if (n % steps_per_base == 0) {
      t_snap.push_back(t);
      T_snap.push_back(T);
    }
  }

  const auto output_path_snap = base_output_path / "ex2_profiles.csv";
  const auto output_path_hist = base_output_path / "ex2_history.csv";
  const auto comment = std::format(
      "alpha={} alpha_eff={} kappa_max={} dt={} dx={} Nx={} Ncell={} "
      "steps={} t_final={}",
      alpha_target, alpha, kappa_max, dt, dx, Nx, Ncell, n_steps,
      t_hist.back());

  std::println("=== RUN SUMMARY ===");
  std::println("profiles_path = {}", output_path_snap.string());
  std::println("history_path = {}", output_path_hist.string());
  std::println("alpha (target) = {}", alpha_target);
  std::println("alpha (effective) = {}", alpha);
  std::println("kappa_max = {}", kappa_max);
  std::println("dt = {}", dt);
  std::println("steps = {}", n_steps);
  for (std::size_t k = 0; k < t_snap.size(); ++k)
    std::println("max T(t={}) = {}", t_snap[k], std::ranges::max(T_snap[k]));
  std::println("T(x={}, t_final) = {}", x[0], T[0]);
  std::println("I(0) = {}", I_hist.front());
  std::println("I(t_final) - I(0) = {}", I_hist.back() - I_hist.front());

  std::vector<std::string> snap_names;
  snap_names.reserve(t_snap.size());
  for (const double t : t_snap)
    snap_names.push_back(std::format("T_t{}", t));
  write_profiles(output_path_snap, comment, snap_names, T_snap);

  const std::array<std::span<const double>, 2> hist_columns{t_hist, I_hist};
  num::write_output(output_path_hist, comment, hist_column_names, hist_columns);
}
