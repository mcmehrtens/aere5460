/// @file
///
/// AERE 5460, Homework 4, Exercise 3. Solves the non-dimensional diffusion
/// equation \f$\frac{\partial T}{\partial t}=\nabla^2T\f$ with finite
/// differences, using ADI (Crank-Nicolson, delta form). T = sin^2(2 pi x) on
/// y = 0 and T = 0 on the other walls.

#include "num/csv.hpp"
#include "num/tridiag.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <format>
#include <mdspan>
#include <numbers>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Tridiag {
  std::vector<double> b, d, a;
};

constexpr std::size_t Nx = 91;
constexpr std::size_t Ny = 191;

using GridExtents = std::extents<std::size_t, Nx, Ny>;
using Grid = std::mdspan<double, GridExtents, std::layout_left>;
using ConstGrid = std::mdspan<const double, GridExtents, std::layout_left>;

// R = (rx dxx + ry dyy) T on interior nodes; wall entries stay 0
void residual(ConstGrid T, Grid out, double rx, double ry) {
  for (std::size_t j = 1; j < Ny - 1; ++j)
    for (std::size_t i = 1; i < Nx - 1; ++i)
      out[i, j] = rx * (T[i - 1, j] - 2.0 * T[i, j] + T[i + 1, j]) +
                  ry * (T[i, j - 1] - 2.0 * T[i, j] + T[i, j + 1]);
}

} // namespace

int main() {
  const auto base_output_path = std::filesystem::path{HW_SOURCE_DIR} / "data";

  constexpr double dx = 1.0 / static_cast<double>(Nx - 1);
  constexpr double dy = 1.0 / static_cast<double>(Ny - 1);

  // (1/2, 1/2) is exactly a node
  static_assert((Nx - 1) % 2 == 0 && (Ny - 1) % 2 == 0);
  constexpr std::size_t i_c = (Nx - 1) / 2;
  constexpr std::size_t j_c = (Ny - 1) / 2;

  // all output times are multiples of t_base; dt divides t_base evenly
  constexpr double t_base = 0.002;
  constexpr std::size_t bases_final = 150; // t_end = 0.3
  constexpr std::array<std::size_t, 5> snap_bases{1, 5, 20, 100, 150};

  // dt is set for accuracy, not stability; ry > rx since dy < dx
  constexpr double ry_max = 1.0;
  constexpr double dt_max = ry_max * dy * dy;
  const auto steps_per_base =
      static_cast<std::size_t>(std::ceil(t_base / dt_max));
  const double dt = t_base / static_cast<double>(steps_per_base);
  const std::size_t n_steps = bases_final * steps_per_base;

  // effective values after shrinking dt
  const double rx = dt / (dx * dx);
  const double ry = dt / (dy * dy);

  // constant LHS factors (1 - r/2 d2); wall rows force dT = 0
  Tridiag x_sweep{
      .b = std::vector<double>(Nx, -0.5 * rx),
      .d = std::vector<double>(Nx, 1.0 + rx),
      .a = std::vector<double>(Nx, -0.5 * rx),
  };
  x_sweep.b[0] = 0.0;
  x_sweep.d[0] = 1.0;
  x_sweep.a[0] = 0.0;
  x_sweep.b[Nx - 1] = 0.0;
  x_sweep.d[Nx - 1] = 1.0;
  x_sweep.a[Nx - 1] = 0.0;

  Tridiag y_sweep{
      .b = std::vector<double>(Ny, -0.5 * ry),
      .d = std::vector<double>(Ny, 1.0 + ry),
      .a = std::vector<double>(Ny, -0.5 * ry),
  };
  y_sweep.b[0] = 0.0;
  y_sweep.d[0] = 1.0;
  y_sweep.a[0] = 0.0;
  y_sweep.b[Ny - 1] = 0.0;
  y_sweep.d[Ny - 1] = 1.0;
  y_sweep.a[Ny - 1] = 0.0;

  std::vector<double> T(Nx * Ny), R(Nx * Ny), dTstar(Nx * Ny), dT(Nx * Ny),
      column(Ny);
  Grid T_grid{T.data()};
  Grid R_grid{R.data()};
  Grid dTstar_grid{dTstar.data()};
  Grid dT_grid{dT.data()};

  // IC: T = 0 except T = sin^2(2 pi x) on the lower wall
  for (std::size_t i = 0; i < Nx; ++i) {
    const double s =
        std::sin(2.0 * std::numbers::pi * static_cast<double>(i) * dx);
    T_grid[i, 0] = s * s;
  }

  // snapshot times come from the base index so column names stay clean
  std::vector<double> t_snap;
  std::vector<std::vector<double>> T_snap;

  std::vector<double> t_hist{0.0};
  std::vector<double> T_center_hist{T_grid[i_c, j_c]};

  for (std::size_t n = 1; n <= n_steps; ++n) {
    // residual from T^n
    residual(ConstGrid{T_grid}, R_grid, rx, ry);

    // sweep in x
    for (std::size_t j = 1; j < Ny - 1; ++j) {
      const auto row =
          num::tridiag(x_sweep.b, x_sweep.d, x_sweep.a,
                       std::span<const double>{R}.subspan(Nx * j, Nx));
      std::ranges::copy(row, std::span{dTstar}.subspan(Nx * j, Nx).begin());
    }

    // sweep in y
    for (std::size_t i = 1; i < Nx - 1; ++i) {
      for (std::size_t j = 0; j < Ny; ++j)
        column[j] = dTstar_grid[i, j];
      const auto col = num::tridiag(y_sweep.b, y_sweep.d, y_sweep.a, column);
      for (std::size_t j = 0; j < Ny; ++j)
        dT_grid[i, j] = col[j];
    }

    // update
    for (std::size_t j = 0; j < Ny; ++j)
      for (std::size_t i = 0; i < Nx; ++i)
        T_grid[i, j] += dT_grid[i, j];

    // T is now at step n
    t_hist.push_back(static_cast<double>(n) * dt);
    T_center_hist.push_back(T_grid[i_c, j_c]);

    const std::size_t base = n / steps_per_base;
    if (n % steps_per_base == 0 && std::ranges::contains(snap_bases, base)) {
      t_snap.push_back(t_base * static_cast<double>(base));
      T_snap.push_back(T);
    }
  }

  // node coordinates in the same flat order as T
  std::vector<double> x_nodes(Nx * Ny), y_nodes(Nx * Ny);
  Grid x_grid{x_nodes.data()};
  Grid y_grid{y_nodes.data()};
  for (std::size_t j = 0; j < Ny; ++j)
    for (std::size_t i = 0; i < Nx; ++i) {
      x_grid[i, j] = static_cast<double>(i) * dx;
      y_grid[i, j] = static_cast<double>(j) * dy;
    }

  const auto profiles_path = base_output_path / "ex3_profiles.csv";
  const auto history_path = base_output_path / "ex3_history.csv";
  const auto comment = std::format(
      "Nx={} Ny={} dx={} dy={} ry_max={} dt={} rx={} ry={} steps={} "
      "steps_per_base={} t_final={}",
      Nx, Ny, dx, dy, ry_max, dt, rx, ry, n_steps, steps_per_base,
      t_hist.back());

  std::println("=== RUN SUMMARY ===");
  std::println("profiles_path = {}", profiles_path.string());
  std::println("history_path = {}", history_path.string());
  std::println("dt = {}", dt);
  std::println("rx = {}", rx);
  std::println("ry = {}", ry);
  std::println("steps = {}", n_steps);
  std::println("steps_per_base = {}", steps_per_base);
  for (std::size_t k = 0; k < t_snap.size(); ++k) {
    // max over y > 0, i.e. off the lower wall
    const ConstGrid snap{T_snap[k].data()};
    double interior_max = 0.0;
    for (std::size_t j = 1; j < Ny; ++j)
      for (std::size_t i = 0; i < Nx; ++i)
        interior_max = std::max(interior_max, snap[i, j]);
    std::println("t={}: max T = {}, max T (y > 0) = {}, T(1/2, 1/2) = {}",
                 t_snap[k], std::ranges::max(T_snap[k]), interior_max,
                 snap[i_c, j_c]);
  }

  // owning storage for the column name views
  std::vector<std::string> snap_names;
  snap_names.reserve(t_snap.size());
  for (const double t : t_snap)
    snap_names.push_back(std::format("T_t{}", t));

  std::vector<std::string_view> column_names{"x", "y"};
  std::vector<std::span<const double>> columns{x_nodes, y_nodes};
  for (std::size_t k = 0; k < t_snap.size(); ++k) {
    column_names.emplace_back(snap_names[k]);
    columns.emplace_back(T_snap[k]);
  }
  num::write_output(profiles_path, comment, column_names, columns);

  constexpr std::array<std::string_view, 2> history_names{"t", "T_center"};
  const std::array<std::span<const double>, 2> history_columns{t_hist,
                                                               T_center_hist};
  num::write_output(history_path, comment, history_names, history_columns);
}
