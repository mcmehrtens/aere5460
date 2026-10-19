/// @file
///
/// AERE 5460, Homework 4, Exercise 1. Solve the non-dimensional diffusion
/// equation with finite differences, using 2nd order Adams–Bashforth.
/// Specifically, it solves \f$\frac{\partial T}{\partial
/// t}=\frac{\partial^2T}{\partial x^2}+\frac{\partial^2T}{\partial y^2}\f$.

#include "num/csv.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <format>
#include <mdspan>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t Nx = 161;
constexpr std::size_t Ny = 161;

using GridExtents = std::extents<std::size_t, Nx, Ny>;
using Grid = std::mdspan<double, GridExtents, std::layout_left>;
using ConstGrid = std::mdspan<const double, GridExtents, std::layout_left>;

void rhs(ConstGrid T, Grid out, double dx, double dy) {
  for (std::size_t j = 1; j < Ny - 1; ++j)
    for (std::size_t i = 1; i < Nx - 1; ++i)
      out[i, j] = (T[i - 1, j] - 2.0 * T[i, j] + T[i + 1, j]) / (dx * dx) +
                  (T[i, j - 1] - 2.0 * T[i, j] + T[i, j + 1]) / (dy * dy);
}
} // namespace

int main() {
  const auto base_output_path = std::filesystem::path{HW_SOURCE_DIR} / "data";

  constexpr double sigma = 0.1;
  constexpr double dx = 1.0 / static_cast<double>(Nx - 1);
  constexpr double dy = 1.0 / static_cast<double>(Ny - 1);

  constexpr double t_end = 0.1;
  constexpr double t_out = 0.05;
  constexpr double dt_max = 0.25 / (1.0 / (dx * dx) + 1.0 / (dy * dy));
  const auto steps_per_out =
      static_cast<std::size_t>(std::ceil(t_out / dt_max));
  const double dt = t_out / static_cast<double>(steps_per_out);
  const auto n_steps =
      static_cast<std::size_t>(std::round(t_end / t_out)) * steps_per_out;

  constexpr double T_boundary = 0.0;

  std::vector<double> T(Nx * Ny);
  Grid T_grid{T.data()};
  for (std::size_t j = 0; j < Ny; ++j)
    for (std::size_t i = 0; i < Nx; ++i) {
      if (i % (Nx - 1) == 0 || j % (Ny - 1) == 0) {
        T_grid[i, j] = T_boundary;
        continue;
      }
      const double x = static_cast<double>(i) * dx;
      const double y = static_cast<double>(j) * dy;
      const double r2 = (x - 0.25) * (x - 0.25) + (y - 0.25) * (y - 0.25);
      T_grid[i, j] = std::exp(-r2 / (sigma * sigma));
    }

  // snapshot times are computed from the output index, not (n + 1) * dt, so
  // the column names come out clean (T_t0.05, not T_t0.05000000000000001)
  std::vector<double> t_snap{0.0};
  std::vector<std::vector<double>> T_snap{T};

  std::vector<double> rhs_prev(Nx * Ny), rhs_curr(Nx * Ny);
  Grid rhs_prev_grid{rhs_prev.data()};
  Grid rhs_curr_grid{rhs_curr.data()};

  rhs(ConstGrid{T_grid}, rhs_prev_grid, dx, dy);
  for (std::size_t j = 1; j < Ny - 1; ++j)
    for (std::size_t i = 1; i < Nx - 1; ++i)
      T_grid[i, j] += dt * rhs_prev_grid[i, j];

  for (std::size_t n = 1; n < n_steps; ++n) {
    rhs(ConstGrid{T.data()}, rhs_curr_grid, dx, dy);
    for (std::size_t j = 1; j < Ny - 1; ++j)
      for (std::size_t i = 1; i < Nx - 1; ++i)
        T_grid[i, j] +=
            dt / 2.0 * (3.0 * rhs_curr_grid[i, j] - rhs_prev_grid[i, j]);
    std::swap(rhs_prev_grid, rhs_curr_grid);

    // T is now at step n + 1
    if ((n + 1) % steps_per_out == 0) {
      t_snap.push_back(t_out * static_cast<double>((n + 1) / steps_per_out));
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

  const auto output_path = base_output_path / "ex1_profiles.csv";
  const auto comment = std::format(
      "Nx={} Ny={} dx={} dy={} sigma={} dt_max={} dt={} steps={} "
      "steps_per_out={} t_final={}",
      Nx, Ny, dx, dy, sigma, dt_max, dt, n_steps, steps_per_out,
      t_snap.back());

  std::println("=== RUN SUMMARY ===");
  std::println("profiles_path = {}", output_path.string());
  std::println("dt_max = {}", dt_max);
  std::println("dt = {}", dt);
  std::println("steps = {}", n_steps);
  std::println("steps_per_out = {}", steps_per_out);
  for (std::size_t k = 0; k < t_snap.size(); ++k)
    std::println("max T(t={}) = {}", t_snap[k], std::ranges::max(T_snap[k]));

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
