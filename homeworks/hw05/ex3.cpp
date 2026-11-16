/// @file
///
/// AERE 5460, Homework 5, Exercise 3. Solves the Poisson equation
/// \f$\nabla^2\psi=2000[\sin(4\pi x)\sin(4\pi y)]^5\f$ on the unit square
/// with finite differences on a 151 x 151 grid, using the conjugate gradient
/// method with symmetric Gauss-Seidel preconditioning.
/// \f$\psi(x,0)=\sin^2(\pi x)+5\sin^2(4\pi x)+10\sin^2(8\pi x)\f$,
/// \f$\psi(0,y)=\sin^2(4\pi y)\f$, \f$\psi=0\f$ on x = 1 and y = 1.

#include "num/csv.hpp"
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <format>
#include <mdspan>
#include <numbers>
#include <numeric>
#include <print>
#include <span>
#include <string_view>
#include <vector>

namespace {

constexpr std::size_t Nx = 151, Ny = 151;

using GridExtents = std::extents<std::size_t, Nx, Ny>;
using Grid = std::mdspan<double, GridExtents, std::layout_left>;
using ConstGrid = std::mdspan<const double, GridExtents, std::layout_left>;

} // namespace

int main() {
  const auto base_output_path = std::filesystem::path{HW_SOURCE_DIR} / "data";

  constexpr std::size_t max_iter = 10'000;
  constexpr double tol = 1.0e-4;
  constexpr double dx = 1.0 / (static_cast<double>(Nx) - 1.0);
  constexpr double dy = 1.0 / (static_cast<double>(Ny) - 1.0);

  // build grid
  std::vector<double> x(Nx), y(Ny);
  for (std::size_t j = 0; j < Nx; ++j)
    x[j] = static_cast<double>(j) * dx;
  for (std::size_t k = 0; k < Ny; ++k)
    y[k] = static_cast<double>(k) * dy;

  // initialize solution vector
  std::vector<double> psi(Nx * Ny);
  Grid psi_grid{psi.data()};

  // apply BCs
  for (std::size_t j = 0; j < Nx; ++j) {
    const double xj = x[j];
    psi_grid[j, 0] = std::pow(std::sin(std::numbers::pi * xj), 2) +
                     5.0 * std::pow(std::sin(4.0 * std::numbers::pi * xj), 2) +
                     10.0 * std::pow(std::sin(8.0 * std::numbers::pi * xj), 2);
    psi_grid[j, Ny - 1] = 0.0;
  }
  for (std::size_t k = 0; k < Ny; ++k) {
    const double yk = y[k];
    psi_grid[0, k] = std::pow(std::sin(4.0 * std::numbers::pi * yk), 2);
    psi_grid[Nx - 1, k] = 0.0;
  }

  // initialize vectors
  std::vector<double> errors;
  std::vector<double> R(Nx * Ny), z(Nx * Ny), d(Nx * Ny), Ad(Nx * Ny);
  Grid R_grid{R.data()}, z_grid{z.data()}, d_grid{d.data()}, Ad_grid{Ad.data()};

  // 5-point Laplacian stencil coefficients
  constexpr double A1 = 1.0 / (dy * dy), A2 = 1.0 / (dx * dx),
                   A4 = 1.0 / (dx * dx), A5 = 1.0 / (dy * dy),
                   A3 = -(A1 + A2 + A4 + A5);

  // initialize R
  for (std::size_t k = 1; k < Ny - 1; ++k)
    for (std::size_t j = 1; j < Nx - 1; ++j) {
      const double xj = x[j], yk = y[k];
      const double omega =
          2000.0 * std::pow(std::sin(4.0 * std::numbers::pi * xj) *
                                std::sin(4.0 * std::numbers::pi * yk),
                            5);
      R_grid[j, k] = omega - A1 * psi_grid[j, k - 1] - A2 * psi_grid[j - 1, k] -
                     A3 * psi_grid[j, k] - A4 * psi_grid[j + 1, k] -
                     A5 * psi_grid[j, k + 1];
    }

  // main CG loop
  double s = 1.0e15, err0 = 0.0;
  for (std::size_t iter = 0; iter < max_iter; ++iter) {
    // precondition with SGS
    // forward sweep
    for (std::size_t k = 1; k < Ny - 1; ++k)
      for (std::size_t j = 1; j < Nx - 1; ++j)
        z_grid[j, k] =
            (R_grid[j, k] - (A2 * z_grid[j - 1, k] + A1 * z_grid[j, k - 1])) /
            A3;
    // backward sweep
    for (std::size_t k = Ny - 2; k > 0; --k)
      for (std::size_t j = Nx - 2; j > 0; --j)
        z_grid[j, k] -= (A4 * z_grid[j + 1, k] + A5 * z_grid[j, k + 1]) / A3;

    const double s_old = s;
    s = 0.0;

    // calculate s
    for (std::size_t k = 0; k < Ny; ++k)
      for (std::size_t j = 0; j < Nx; ++j)
        s += R_grid[j, k] * z_grid[j, k];

    const double beta = s / s_old;

    // calculate d
    for (std::size_t k = 0; k < Ny; ++k)
      for (std::size_t j = 0; j < Nx; ++j)
        d_grid[j, k] = z_grid[j, k] + beta * d_grid[j, k];

    // calculate Ad
    for (std::size_t k = 1; k < Ny - 1; ++k)
      for (std::size_t j = 1; j < Nx - 1; ++j)
        Ad_grid[j, k] = A1 * d_grid[j, k - 1] + A2 * d_grid[j - 1, k] +
                        A3 * d_grid[j, k] + A4 * d_grid[j + 1, k] +
                        A5 * d_grid[j, k + 1];

    // calculate alpha denominator
    double den = 0.0;
    for (std::size_t k = 0; k < Ny; ++k)
      for (std::size_t j = 0; j < Nx; ++j)
        den += d_grid[j, k] * Ad_grid[j, k];

    const double alpha = s / den;

    // update solution and residual
    for (std::size_t k = 0; k < Ny; ++k)
      for (std::size_t j = 0; j < Nx; ++j) {
        psi_grid[j, k] += alpha * d_grid[j, k];
        R_grid[j, k] -= alpha * Ad_grid[j, k];
      }

    double err = 0.0;
    for (std::size_t k = 0; k < Ny; ++k)
      for (std::size_t j = 0; j < Nx; ++j) {
        err += d_grid[j, k] * d_grid[j, k];
      }

    if (iter == 0)
      err0 = alpha * alpha * err;

    errors.push_back(std::sqrt(alpha * alpha * err / err0));

    if (errors.back() < tol)
      break;
  }

  // one row per node, x varying fastest
  std::vector<double> x_nodes(Nx * Ny), y_nodes(Nx * Ny);
  Grid x_nodes_grid{x_nodes.data()}, y_nodes_grid{y_nodes.data()};
  for (std::size_t k = 0; k < Ny; ++k)
    for (std::size_t j = 0; j < Nx; ++j) {
      x_nodes_grid[j, k] = x[j];
      y_nodes_grid[j, k] = y[k];
    }

  std::vector<double> iteration(errors.size());
  std::iota(iteration.begin(), iteration.end(), 1.0);

  const bool converged = errors.back() < tol;
  const auto comment =
      std::format("Nx={} Ny={} tol={} max_iter={} iterations={} converged={}",
                  Nx, Ny, tol, max_iter, errors.size(), converged);

  const auto psi_path = base_output_path / "ex3_psi.csv";
  constexpr std::array<std::string_view, 3> psi_names = {"x", "y", "psi"};
  const std::array<std::span<const double>, 3> psi_columns = {x_nodes, y_nodes,
                                                              psi};
  num::write_output(psi_path, comment, psi_names, psi_columns);

  const auto residual_path = base_output_path / "ex3_residual.csv";
  constexpr std::array<std::string_view, 2> residual_names = {"iteration",
                                                              "residual"};
  const std::array<std::span<const double>, 2> residual_columns = {iteration,
                                                                   errors};
  num::write_output(residual_path, comment, residual_names, residual_columns);

  std::println("=== RUN SUMMARY ===");
  std::println("psi_path = {}", psi_path.string());
  std::println("residual_path = {}", residual_path.string());
  std::println("{}", comment);
}
