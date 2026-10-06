/// @file
///
/// AERE 5460, Homework 3, Exercise 1. Solves the 1D diffusion equation with an
/// Euler explicit method. Specifically, it solves the non-dimensionalized
/// equation \f$\frac{\partial T}{\partial t}=\frac{\partial^2T}{\partial
/// x^2}\f$ for several values of \f$\alpha=\Delta t/\Delta x^2\f$, both
/// stable and unstable. Generates a profile CSV
/// (`ex1_alpha<alpha>_profiles.csv`) and a heat transfer history CSV
/// (`ex1_alpha<alpha>_history.csv`) for each run and puts them in the `data`
/// folder relative to this source file.

#include "num/csv.hpp"
#include "num/ode.hpp"
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
#include <utility>
#include <vector>

int main() {
  const auto base_output_path = std::filesystem::path{HW_SOURCE_DIR} / "data";

  // I'm doing all this business with bases to ensure that my output times all
  // end up as whole multiples of 0.02 (which is the largest common denominator
  // of both 0.04 and 0.3)
  constexpr std::size_t bases_per_unit_time = 50;
  constexpr double t_base = 1.0 / static_cast<double>(bases_per_unit_time);
  constexpr std::size_t bases_per_snap = 2; // dt_snap = 0.04
  constexpr std::size_t bases_final = 15;   // tf = 0.3

  constexpr std::size_t Nx = 201;
  constexpr double dx = 1.0 / static_cast<double>(Nx - 1);
  constexpr std::array<double, 8> alpha_targets = {0.6, 0.51, 0.5,       0.49,
                                                   0.4, 0.25, 1.0 / 6.0, 0.1};

  // early profiles show how the step at x = 1 evolves near the stability limit
  constexpr std::size_t steps_per_early_snap = 5;
  constexpr std::size_t n_early = 300;

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

    const std::size_t steps_per_snap = bases_per_snap * steps_per_base;
    const std::size_t n_steps = bases_final * steps_per_base;
    const std::size_t n_snaps = n_steps / steps_per_snap + 2;

    std::vector<double> t_snap;
    t_snap.reserve(n_snaps);
    t_snap.push_back(0.0);

    std::vector<num::State<Nx>> T_snap;
    T_snap.reserve(n_snaps);
    num::State<Nx> T{};
    num::State<Nx> T_new{};
    T[Nx - 1] = T_new[Nx - 1] = 1.0;
    T_snap.push_back(T);

    std::vector<std::size_t> n_early_snap{0};
    std::vector<num::State<Nx>> T_early_snap{T};

    std::vector<double> t_hist, Q_hist, hT_hist;
    t_hist.reserve(n_steps);
    Q_hist.reserve(n_steps);
    hT_hist.reserve(n_steps);

    std::size_t n_final = 0;
    bool diverged = false;
    for (std::size_t n = 1; n <= n_steps; ++n) {
      n_final = n;
      const double t =
          static_cast<double>(n) / static_cast<double>(steps_per_unit_time);

      for (std::size_t i = 1; i < Nx - 1; ++i) {
        T_new[i] = T[i] + alpha * (T[i - 1] - 2.0 * T[i] + T[i + 1]);
      }
      std::swap(T_new, T);

      const bool bounded =
          std::ranges::all_of(T, [](double v) { return std::abs(v) <= 10.0; });
      if (!bounded) {
        diverged = true;
        n_early_snap.push_back(n);
        T_early_snap.push_back(T);
        break;
      }

      const double Q =
          (T[Nx - 3] - 4.0 * T[Nx - 2] + 3.0 * T[Nx - 1]) / (2.0 * dx);
      const double hT = Q / (T[Nx - 1] - T[0]);

      t_hist.push_back(t);
      Q_hist.push_back(Q);
      hT_hist.push_back(hT);

      if (n % steps_per_snap == 0 || n == n_steps) {
        t_snap.push_back(t);
        T_snap.push_back(T);
      }
      if (n % steps_per_early_snap == 0 && n <= n_early) {
        n_early_snap.push_back(n);
        T_early_snap.push_back(T);
      }
    }
    const double t_final =
        static_cast<double>(n_final) / static_cast<double>(steps_per_unit_time);

    const auto output_path_snap =
        base_output_path /
        std::format("ex1_alpha{:.3g}_profiles.csv", alpha_target);
    const auto output_path_early =
        base_output_path /
        std::format("ex1_alpha{:.3g}_early.csv", alpha_target);
    const auto output_path_hist =
        base_output_path /
        std::format("ex1_alpha{:.3g}_history.csv", alpha_target);
    const auto comment = std::format(
        "alpha={} alpha_eff={} dt={} dx={} Nx={} steps={} t_final={} "
        "diverged={}",
        alpha_target, alpha, dt, dx, Nx, n_final, t_final, diverged ? 1 : 0);

    std::println("=== RUN SUMMARY ===");
    std::println("profiles_path = {}", output_path_snap.string());
    std::println("early_path = {}", output_path_early.string());
    std::println("history_path = {}", output_path_hist.string());
    std::println("alpha (target) = {}", alpha_target);
    std::println("alpha (effective) = {}", alpha);
    std::println("dt = {}", dt);
    std::println("steps = {}", n_final);
    std::println("t_final = {}", t_final);
    if (diverged) {
      std::println("DIVERGED: max|T| > 10 at step {}", n_final);
    } else {
      std::println("T(0.5) = {}", T[(Nx - 1) / 2]);
      std::println("Q(1) = {}", Q_hist.back());
      std::println("hT = {}", hT_hist.back());
    }
    std::println();

    std::vector<std::string> snap_names;
    snap_names.reserve(t_snap.size());
    for (const double t : t_snap)
      snap_names.push_back(std::format("T_t{}", t));
    write_profiles(output_path_snap, comment, snap_names, T_snap);

    std::vector<std::string> early_names;
    early_names.reserve(n_early_snap.size());
    for (const std::size_t n : n_early_snap)
      early_names.push_back(std::format("T_n{}", n));
    write_profiles(output_path_early, comment, early_names, T_early_snap);

    const std::array<std::span<const double>, 3> hist_columns{t_hist, Q_hist,
                                                              hT_hist};
    num::write_output(output_path_hist, comment, hist_column_names,
                      hist_columns);
  }
}
