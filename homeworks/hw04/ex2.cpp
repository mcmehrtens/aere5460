/// @file
///
/// AERE 5460, Homework 4, Exercise 2. Solves the non-dimensional
/// convection-diffusion equation \f$\frac{\partial T}{\partial
/// t}=-\frac{\partial F_x}{\partial x}-\frac{\partial F_y}{\partial y}\f$ with
/// \f$F_x=-\frac{\partial T}{\partial x}+UT\f$ and \f$F_y=-\frac{\partial
/// T}{\partial y}+VT\f$, using a vertex-centered finite-volume discretization
/// in space and RK2 in time. The walls at x = 0 and y = 0 are insulated; x = 1
/// and y = 1 use a convective outflow condition.

#include "num/csv.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <format>
#include <mdspan>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

// grid: Nx = Ny = 161 nodes, dx = dy = 1/(N-1), 0-based i, j = 0..N-1. Node k
// in your notes is index k-1 here, so the ghost node T_0 is index -1.
//
// reuse ex1's Grid/ConstGrid mdspan aliases (layout_left, T[i, j])

constexpr std::size_t Nx = 161;
constexpr std::size_t Ny = 161;
constexpr double U = 50.0;
constexpr double V = 50.0;

using GridExtents = std::extents<std::size_t, Nx, Ny>;
using Grid = std::mdspan<double, GridExtents, std::layout_left>;
using ConstGrid = std::mdspan<const double, GridExtents, std::layout_left>;

void rhs(ConstGrid T, Grid out, double dx, double dy) {
  for (std::size_t j = 0; j < Ny - 1; ++j)
    for (std::size_t i = 0; i < Nx - 1; ++i) {
      // this accounts for the Neumann condition on the left and lower wall
      const std::size_t im = (i == 0) ? 1 : i - 1;
      const std::size_t jm = (j == 0) ? 1 : j - 1;

      out[i, j] = (T[im, j] - 2.0 * T[i, j] + T[i + 1, j]) / (dx * dx) +
                  (T[i, jm] - 2.0 * T[i, j] + T[i, j + 1]) / (dy * dy) -
                  U * (T[i + 1, j] - T[im, j]) / (2.0 * dx) -
                  V * (T[i, j + 1] - T[i, jm]) / (2.0 * dy);
    }
  for (std::size_t j = 0; j < Ny - 1; ++j)
    out[Nx - 1, j] = -U * (T[Nx - 1, j] - T[Nx - 2, j]) / dx;
  for (std::size_t i = 0; i < Nx - 1; ++i)
    out[i, Ny - 1] = -V * (T[i, Ny - 1] - T[i, Ny - 2]) / dy;
  out[Nx - 1, Ny - 1] = -U * (T[Nx - 1, Ny - 1] - T[Nx - 2, Ny - 1]) / dx -
                        V * (T[Nx - 1, Ny - 1] - T[Nx - 1, Ny - 2]) / dy;
}

} // namespace

int main() {
  const auto base_output_path = std::filesystem::path{HW_SOURCE_DIR} / "data";

  constexpr double sigma = 0.3;
  constexpr double dx = 1.0 / static_cast<double>(Nx - 1);
  constexpr double dy = 1.0 / static_cast<double>(Ny - 1);

  // every output time (0, 0.004, 0.01, 0.012, 0.02) is a whole number of
  // t_base intervals, so dt is shrunk until t_base is a whole number of steps
  constexpr double t_base = 0.002;
  constexpr std::size_t bases_final = 10; // t_end = 0.02
  constexpr std::array<std::size_t, 4> snap_bases{2, 5, 6, 10};

  constexpr double dt_max = 0.5 / (1.0 / (dx * dx) + 1.0 / (dy * dy));
  const auto steps_per_base =
      static_cast<std::size_t>(std::ceil(t_base / dt_max));
  const double dt = t_base / static_cast<double>(steps_per_base);
  const std::size_t n_steps = bases_final * steps_per_base;

  std::vector<double> T(Nx * Ny), T_half(Nx * Ny), R(Nx * Ny);
  Grid T_grid{T.data()};
  Grid T_half_grid{T_half.data()};
  Grid R_grid{R.data()};

  // IC
  for (std::size_t j = 0; j < Ny; ++j)
    for (std::size_t i = 0; i < Nx; ++i) {
      const double x = static_cast<double>(i) * dx;
      const double y = static_cast<double>(j) * dy;
      const double r2 = (x - 0.25) * (x - 0.25) + (y - 0.25) * (y - 0.25);
      T_grid[i, j] = std::exp(-r2 / (sigma * sigma));
    }

  // snapshot times are computed from the base index, not n * dt, so the
  // column names come out clean (T_t0.004, not T_t0.0040000000000000001)
  std::vector<double> t_snap{0.0};
  std::vector<std::vector<double>> T_snap{T};

  // RK-2
  for (std::size_t n = 1; n <= n_steps; ++n) {
    // Evaluate T^(1/2)
    rhs(ConstGrid{T_grid}, R_grid, dx, dy);
    for (std::size_t j = 0; j < Ny; ++j)
      for (std::size_t i = 0; i < Nx; ++i)
        T_half_grid[i, j] = T_grid[i, j] + dt / 2.0 * R_grid[i, j];

    // Evaluate T^(n+1)
    rhs(ConstGrid{T_half_grid}, R_grid, dx, dy);
    for (std::size_t j = 0; j < Ny; ++j)
      for (std::size_t i = 0; i < Nx; ++i)
        T_grid[i, j] += dt * R_grid[i, j];

    // T is now at step n
    const std::size_t base = n / steps_per_base;
    if (n % steps_per_base == 0 && std::ranges::contains(snap_bases, base)) {
      t_snap.push_back(t_base * static_cast<double>(base));
      T_snap.push_back(T);
    }
  }

  // node coordinates in the same flat order as T, so each CSV row is one node
  std::vector<double> x_nodes(Nx * Ny), y_nodes(Nx * Ny);
  Grid x_grid{x_nodes.data()};
  Grid y_grid{y_nodes.data()};
  for (std::size_t j = 0; j < Ny; ++j)
    for (std::size_t i = 0; i < Nx; ++i) {
      x_grid[i, j] = static_cast<double>(i) * dx;
      y_grid[i, j] = static_cast<double>(j) * dy;
    }

  // total heat by the trapezoid rule: edge nodes own half a cell, corners a
  // quarter
  const auto heat = [](const std::vector<double> &field) {
    const ConstGrid f{field.data()};
    double sum = 0.0;
    for (std::size_t j = 0; j < Ny; ++j)
      for (std::size_t i = 0; i < Nx; ++i) {
        const double wx = (i % (Nx - 1) == 0) ? 0.5 : 1.0;
        const double wy = (j % (Ny - 1) == 0) ? 0.5 : 1.0;
        sum += wx * wy * f[i, j];
      }
    return sum * dx * dy;
  };

  const auto output_path = base_output_path / "ex2_profiles.csv";
  const auto comment = std::format(
      "Nx={} Ny={} dx={} dy={} U={} V={} sigma={} dt_max={} dt={} steps={} "
      "steps_per_base={} t_final={}",
      Nx, Ny, dx, dy, U, V, sigma, dt_max, dt, n_steps, steps_per_base,
      t_snap.back());

  std::println("=== RUN SUMMARY ===");
  std::println("profiles_path = {}", output_path.string());
  std::println("dt_max = {}", dt_max);
  std::println("dt = {}", dt);
  std::println("steps = {}", n_steps);
  std::println("steps_per_base = {}", steps_per_base);
  for (std::size_t k = 0; k < t_snap.size(); ++k) {
    const auto peak = std::ranges::max_element(T_snap[k]);
    const auto idx = static_cast<std::size_t>(peak - T_snap[k].begin());
    std::println("max T(t={}) = {} at (x, y) = ({}, {})", t_snap[k], *peak,
                 x_nodes[idx], y_nodes[idx]);
    std::println("heat(t={}) = {}", t_snap[k], heat(T_snap[k]));
  }

  // the views passed to write_output need owning storage that outlives them
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
  num::write_output(output_path, comment, column_names, columns);
}
