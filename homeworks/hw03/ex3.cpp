/// @file
///
/// AERE 5460, Homework 3, Exercise 3. Solves the 1D diffusion equation
/// \f$\frac{\partial T}{\partial t}=\frac{\partial^2T}{\partial x^2}\f$ with
/// an insulated end at x = 0 and T = 1 at x = 1 using Crank–Nicolson, for
/// several values of \f$\alpha=\Delta t/\Delta x^2\f$.

#include "num/csv.hpp"
#include "num/ode.hpp"
#include "num/tridiag.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <format>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <vector>

int main() {
  const auto base_output_path = std::filesystem::path{HW_SOURCE_DIR} / "data";

  constexpr std::size_t bases_per_unit_time = 25;
  constexpr double t_base = 1.0 / static_cast<double>(bases_per_unit_time);
  constexpr std::size_t bases_final = 10; // tf = 0.4

  constexpr std::size_t Nx = 201;
  constexpr double dx = 1.0 / static_cast<double>(Nx - 1);
  // T_{Nx-1} = 1 is known, so the system solves for T_0, ..., T_{Nx-2}
  constexpr std::size_t N = Nx - 1;
  constexpr std::array<double, 4> alpha_targets = {1.0, 10.0, 100.0, 1600.0};

  constexpr std::array<std::string_view, 3> hist_column_names{"t", "Q", "hT"};

  const auto x = [] {
    num::State<Nx> g{};
    for (std::size_t i = 0; i < Nx; ++i)
      g[i] = static_cast<double>(i) * dx;
    return g;
  }();

  // the views passed to write_output need owning storage that outlives them
  const auto write_profiles =
      [&x](const std::filesystem::path &path, std::string_view comment,
           const std::vector<std::string> &names,
           const std::vector<num::State<Nx>> &profiles) {
        std::vector<std::string_view> column_names{"x"};
        std::vector<std::span<const double>> columns{x};
        for (std::size_t k = 0; k < names.size(); ++k) {
          column_names.emplace_back(names[k]);
          columns.emplace_back(profiles[k]);
        }
        num::write_output(path, comment, column_names, columns);
      };

  for (const auto alpha_target : alpha_targets) {
    const auto steps_per_base = static_cast<std::size_t>(
        std::ceil(t_base / (alpha_target * dx * dx) - 1e-9));
    const std::size_t steps_per_unit_time =
        bases_per_unit_time * steps_per_base;
    const double dt = 1.0 / static_cast<double>(steps_per_unit_time);
    const double alpha = dt / (dx * dx);
    const std::size_t n_steps = bases_final * steps_per_base;

    // A T^{n+1} = B T^n + g; A is constant, so build it once
    std::vector<double> b(N, -0.5 * alpha);
    std::vector<double> d(N, 1.0 + alpha);
    std::vector<double> a(N, -0.5 * alpha);
    a[0] = -alpha; // ghost node T_{-1} = T_1
    std::vector<double> r(N);

    num::State<Nx> T{};
    T[Nx - 1] = 1.0;

    std::vector<double> t_snap{0.0};
    std::vector<num::State<Nx>> T_snap{T};
    t_snap.reserve(bases_final + 1);
    T_snap.reserve(bases_final + 1);

    std::vector<double> t_hist, Q_hist, hT_hist;
    t_hist.reserve(n_steps);
    Q_hist.reserve(n_steps);
    hT_hist.reserve(n_steps);

    double max_abs_T = 1.0;
    for (std::size_t n = 1; n <= n_steps; ++n) {
      const double t =
          static_cast<double>(n) / static_cast<double>(steps_per_unit_time);

      r[0] = (1.0 - alpha) * T[0] + alpha * T[1];
      for (std::size_t i = 1; i < N - 1; ++i)
        r[i] = 0.5 * alpha * T[i - 1] + (1.0 - alpha) * T[i] +
               0.5 * alpha * T[i + 1];
      r[N - 1] = 0.5 * alpha * T[N - 2] + (1.0 - alpha) * T[N - 1] + alpha;

      std::ranges::copy(num::tridiag(b, d, a, r), T.begin());

      for (const double v : T)
        max_abs_T = std::max(max_abs_T, std::abs(v));

      const double Q =
          (T[Nx - 3] - 4.0 * T[Nx - 2] + 3.0 * T[Nx - 1]) / (2.0 * dx);
      const double hT = Q / (T[Nx - 1] - T[0]);

      t_hist.push_back(t);
      Q_hist.push_back(Q);
      hT_hist.push_back(hT);

      if (n % steps_per_base == 0) {
        t_snap.push_back(t);
        T_snap.push_back(T);
      }
    }

    const auto output_path_snap =
        base_output_path /
        std::format("ex3_alpha{:g}_profiles.csv", alpha_target);
    const auto output_path_hist =
        base_output_path /
        std::format("ex3_alpha{:g}_history.csv", alpha_target);
    const auto comment = std::format(
        "alpha={} alpha_eff={} dt={} dx={} Nx={} steps={} "
        "t_final={} max_abs_T={}",
        alpha_target, alpha, dt, dx, Nx, n_steps, t_hist.back(), max_abs_T);

    std::println("=== RUN SUMMARY ===");
    std::println("profiles_path = {}", output_path_snap.string());
    std::println("history_path = {}", output_path_hist.string());
    std::println("alpha (target) = {}", alpha_target);
    std::println("alpha (effective) = {}", alpha);
    std::println("dt = {}", dt);
    std::println("steps = {}", n_steps);
    std::println("max|T| = {}", max_abs_T);
    std::println("T(0) = {}", T[0]);
    std::println("Q(1) = {}", Q_hist.back());
    std::println("hT = {}", hT_hist.back());
    std::println();

    std::vector<std::string> snap_names;
    snap_names.reserve(t_snap.size());
    for (const double t : t_snap)
      snap_names.push_back(std::format("T_t{}", t));
    write_profiles(output_path_snap, comment, snap_names, T_snap);

    const std::array<std::span<const double>, 3> hist_columns{t_hist, Q_hist,
                                                              hT_hist};
    num::write_output(output_path_hist, comment, hist_column_names,
                      hist_columns);
  }
}
