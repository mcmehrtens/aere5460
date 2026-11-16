/// @file
///
/// AERE 5460, Homework 5, Exercise 1. Solves the Laplace equation
/// \f$\frac{\partial^2\psi}{\partial x^2}+\frac{\partial^2\psi}{\partial
/// y^2}=0\f$ on the unit square with finite differences, using Gauss-Seidel
/// with successive over-relaxation (SOR). \f$\psi=\sin^2(3n\pi y)\f$ on x = 0,
/// \f$\psi=\sin^2(3n\pi x)\f$ on y = 0 and \f$\psi=0\f$ on x = 1 and y = 1.

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

using GridExtents = std::dextents<std::size_t, 2>;
using Grid = std::mdspan<double, GridExtents, std::layout_left>;

struct GridSize {
  std::size_t Nx;
  std::size_t Ny;
};

struct SorResult {
  std::vector<double> residual;
};

SorResult solve_sor(Grid psi, double lambda, double tol, std::size_t max_iter) {
  const std::size_t Nx = psi.extent(0), Ny = psi.extent(1);
  const double dx = 1.0 / (static_cast<double>(Nx) - 1.0),
               dy = 1.0 / (static_cast<double>(Ny) - 1.0);
  const double beta = dx / dy;
  const double beta2 = beta * beta;
  const double denom = 2.0 * (1.0 + beta2);

  SorResult result;
  result.residual.reserve(max_iter);

  for (std::size_t iter = 0; iter < max_iter; ++iter) {
    // Gauss–Seidel sweep
    double eps = 0.0;
    for (std::size_t j = 1; j < Ny - 1; ++j) {
      for (std::size_t i = 1; i < Nx - 1; ++i) {
        const double psi_new = (psi[i + 1, j] + psi[i - 1, j] +
                                beta2 * (psi[i, j + 1] + psi[i, j - 1])) /
                               denom;
        const double Deltapsi = psi_new - psi[i, j];
        // it's not clear to me whether eps should take into account the
        // relaxation factor, but it divides out when calculating the ratio so
        // it shouldn't affect anything
        eps += Deltapsi * Deltapsi;
        psi[i, j] = psi[i, j] + lambda * Deltapsi;
      }
    }
    // -2 in each direction since we don't sum over the boundary points (and the
    // boundary points don't change)
    eps = std::sqrt(eps / (static_cast<double>((Nx - 2) * (Ny - 2))));
    result.residual.push_back(eps);

    if (eps / result.residual.front() < tol)
      break;
  }
  return result;
}

void apply_bc(Grid psi, const double n) {
  const std::size_t Nx = psi.extent(0), Ny = psi.extent(1);
  const double dx = 1.0 / (static_cast<double>(Nx) - 1.0),
               dy = 1.0 / (static_cast<double>(Ny) - 1.0);

  for (std::size_t j = 0; j < Ny; ++j) {
    const double y = static_cast<double>(j) * dy;
    const double sin_val = std::sin(3.0 * n * std::numbers::pi * y);
    psi[0, j] = sin_val * sin_val;
  }

  for (std::size_t i = 0; i < Nx; ++i) {
    const double x = static_cast<double>(i) * dx;
    const double sin_val = std::sin(3.0 * n * std::numbers::pi * x);
    psi[i, 0] = sin_val * sin_val;
  }
}

// one row per node, x varying fastest
void write_solution(const std::filesystem::path &path, std::string_view comment,
                    Grid psi) {
  const std::size_t Nx = psi.extent(0), Ny = psi.extent(1);
  const double dx = 1.0 / (static_cast<double>(Nx) - 1.0),
               dy = 1.0 / (static_cast<double>(Ny) - 1.0);

  std::vector<double> x(psi.size()), y(psi.size());
  Grid x_grid{x.data(), Nx, Ny};
  Grid y_grid{y.data(), Nx, Ny};
  for (std::size_t j = 0; j < Ny; ++j)
    for (std::size_t i = 0; i < Nx; ++i) {
      x_grid[i, j] = static_cast<double>(i) * dx;
      y_grid[i, j] = static_cast<double>(j) * dy;
    }

  constexpr std::array<std::string_view, 3> names = {"x", "y", "psi"};
  const std::array<std::span<const double>, 3> columns = {
      x, y, std::span<const double>{psi.data_handle(), psi.size()}};
  num::write_output(path, comment, names, columns);
}

void write_residual(const std::filesystem::path &path, std::string_view comment,
                    const SorResult &result) {
  std::vector<double> iteration(result.residual.size());
  std::iota(iteration.begin(), iteration.end(), 1.0);

  constexpr std::array<std::string_view, 2> names = {"iteration", "eps"};
  const std::array<std::span<const double>, 2> columns = {iteration,
                                                          result.residual};
  num::write_output(path, comment, names, columns);
}

} // namespace

int main() {
  const auto base_output_path = std::filesystem::path{HW_SOURCE_DIR} / "data";

  constexpr std::array n_array = {1.0, 3.0};
  constexpr std::array grids = {GridSize{15, 15}, GridSize{151, 101}};
  constexpr double lambda_opt = 1.95;
  constexpr std::array lambda_array = {1.0, 0.5, 1.5, 1.95};
  constexpr double tol = 1.0e-4;
  constexpr std::size_t max_iter = 2000;

  const auto describe = [&](double n, Grid psi, double lambda,
                            const SorResult &result) {
    const double drop = result.residual.back() / result.residual.front();
    return std::format("n={} Nx={} Ny={} lambda={} tol={} max_iter={} "
                       "iterations={} eps_ratio={} converged={}",
                       n, psi.extent(0), psi.extent(1), lambda, tol, max_iter,
                       result.residual.size(), drop, drop < tol);
  };
  const auto summarize = [](const std::filesystem::path &path,
                            std::string_view comment) {
    std::println("=== RUN SUMMARY ===");
    std::println("path = {}", path.string());
    std::println("{}", comment);
  };

  for (const double n : n_array) {
    for (const auto [Nx, Ny] : grids) {
      std::vector<double> psi(Nx * Ny);
      Grid psi_grid{psi.data(), Nx, Ny};
      apply_bc(psi_grid, n);
      const SorResult result = solve_sor(psi_grid, lambda_opt, tol, max_iter);

      const auto path =
          base_output_path / std::format("ex1_n{}_{}x{}_psi.csv", n, Nx, Ny);
      const auto comment = describe(n, psi_grid, lambda_opt, result);
      write_solution(path, comment, psi_grid);
      summarize(path, comment);
    }
  }

  for (const double lambda : lambda_array) {
    const std::size_t Nx = 151, Ny = 101;
    const double n = 1.0;
    std::vector<double> psi(Nx * Ny);
    Grid psi_grid{psi.data(), Nx, Ny};
    apply_bc(psi_grid, n);
    const SorResult result = solve_sor(psi_grid, lambda, tol, max_iter);

    const auto path =
        base_output_path / std::format("ex1_residual_lambda{}.csv", lambda);
    const auto comment = describe(n, psi_grid, lambda, result);
    write_residual(path, comment, result);
    summarize(path, comment);
  }
}
