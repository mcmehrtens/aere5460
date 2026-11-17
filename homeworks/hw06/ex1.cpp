/// @file
///
/// AERE 5460, Homework 6, Exercise 1. Potential flow through a contraction:
/// solves \f$\nabla^2\Psi=0\f$ on a 161 x 131 grid with the step represented
/// by i-blanking, using Gauss-Seidel with SOR.

#include "num/csv.hpp"
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <format>
#include <mdspan>
#include <print>
#include <span>
#include <string_view>
#include <vector>

namespace {

constexpr double Lx = 2.0;
constexpr double Ly = 1.0;
constexpr std::size_t Nx = 161;
constexpr std::size_t Ny = 131;
constexpr double dx = Lx / (static_cast<double>(Nx) - 1.0);
constexpr double dy = Ly / (static_cast<double>(Ny) - 1.0);

// j_step occurs at first grid cell >= 1.0
constexpr std::size_t j_step = ((Nx - 1) + 1) / 2; // ceil((Nx - 1) / 2)
// k_step occurs at last grid cell <= 0.25
constexpr std::size_t k_step = (Ny - 1) / 4; // floor((Ny - 1) / 4)

using GridExtents = std::extents<std::size_t, Nx, Ny>;
using Grid = std::mdspan<double, GridExtents, std::layout_left>;
using ConstGrid = std::mdspan<const double, GridExtents, std::layout_left>;

struct SorResult {
  bool converged = false;
  std::size_t iterations = 0;
  double eps_ratio = 1.0;
};

SorResult solve_sor(Grid psi, ConstGrid IB, double lambda, double tol,
                    std::size_t max_iter) {
  const double beta = dx / dy;
  const double beta2 = beta * beta;
  const double denom = 2.0 * (1.0 + beta2);

  // calculate number of fluid points
  double n_fluid = 0.0;
  for (std::size_t k = 1; k < Ny - 1; ++k)
    for (std::size_t j = 1; j < Nx - 1; ++j)
      n_fluid += IB[j, k];

  double eps0 = 0.0;
  SorResult result;

  for (std::size_t iter = 0; iter < max_iter; ++iter) {
    // Gauss–Seidel sweep
    double eps = 0.0;
    for (std::size_t k = 1; k < Ny - 1; ++k) {
      for (std::size_t j = 1; j < Nx - 1; ++j) {
        const double psi_new = (psi[j + 1, k] + psi[j - 1, k] +
                                beta2 * (psi[j, k + 1] + psi[j, k - 1])) /
                               denom;
        const double Deltapsi = (psi_new - psi[j, k]) * IB[j, k];
        eps += Deltapsi * Deltapsi;
        psi[j, k] = psi[j, k] + lambda * Deltapsi;
      }
    }
    eps = std::sqrt(eps / n_fluid);

    if (iter == 0)
      eps0 = eps;
    result.iterations = iter + 1;
    result.eps_ratio = eps / eps0;
    if (result.eps_ratio < tol) {
      result.converged = true;
      break;
    }
  }
  return result;
}

void write_solution(const std::filesystem::path &path, std::string_view comment,
                    ConstGrid psi, ConstGrid IB) {
  std::vector<double> x(psi.size()), y(psi.size());
  Grid x_grid{x.data()};
  Grid y_grid{y.data()};
  for (std::size_t k = 0; k < Ny; ++k)
    for (std::size_t j = 0; j < Nx; ++j) {
      x_grid[j, k] = static_cast<double>(j) * dx;
      y_grid[j, k] = static_cast<double>(k) * dy;
    }

  constexpr std::array<std::string_view, 4> names = {"x", "y", "psi", "iblank"};
  const std::array<std::span<const double>, 4> columns = {
      x, y, std::span<const double>{psi.data_handle(), psi.size()},
      std::span<const double>{IB.data_handle(), IB.size()}};
  num::write_output(path, comment, names, columns);
}

void apply_bc(Grid psi) {
  for (std::size_t k = 0; k < Ny; ++k) {
    const double y = static_cast<double>(k) * dy;
    psi[0, k] = -y;
    if (k > k_step)
      psi[Nx - 1, k] = -4.0 / 3.0 * (y - 0.25);
  }

  for (std::size_t j = 0; j < Nx; ++j) {
    psi[j, 0] = 0.0;
    psi[j, Ny - 1] = -1.0;
  }

  // blank out the inside of the step
  for (std::size_t k = 0; k <= k_step; ++k)
    for (std::size_t j = j_step; j < Nx; ++j)
      psi[j, k] = 0.0;
}
} // namespace

int main() {
  constexpr double lambda = 1.9;
  constexpr std::size_t max_iter = 2000;
  constexpr double tol = 1.0e-4;

  std::vector<double> psi(Nx * Ny);
  Grid psi_grid{psi.data()};
  apply_bc(psi_grid);

  // build I-blanking vector
  std::vector<double> IB(Nx * Ny, 1.0);
  Grid IB_grid{IB.data()};
  for (std::size_t k = 0; k <= k_step; ++k)
    for (std::size_t j = j_step; j < Nx; ++j)
      IB_grid[j, k] = 0.0;

  const SorResult result = solve_sor(psi_grid, IB_grid, lambda, tol, max_iter);

  const auto path =
      std::filesystem::path{HW_SOURCE_DIR} / "data" / "ex1_psi.csv";
  const auto comment = std::format(
      "Nx={} Ny={} lambda={} tol={} max_iter={} iterations={} eps_ratio={} "
      "converged={}",
      Nx, Ny, lambda, tol, max_iter, result.iterations, result.eps_ratio,
      result.converged);
  write_solution(path, comment, psi_grid, IB_grid);

  std::println("=== RUN SUMMARY ===");
  std::println("path = {}", path.string());
  std::println("{}", comment);
}
