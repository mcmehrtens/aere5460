/// @file
///
/// AERE 5460, Homework 6, Exercise 2. Flow around a stair-step elliptical
/// cylinder: solves \f$\nabla^2\Psi=0\f$ on a 271 x 241 grid with the cylinder
/// represented by i-blanking, using SOR and Conjugate Gradient.

#include "num/csv.hpp"
#include <array>
#include <cmath>
#include <cstddef>
#include <ctime>
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

constexpr double deg2rad(double deg) { return deg * std::numbers::pi / 180.0; }

constexpr std::size_t Nx = 271;
constexpr std::size_t Ny = 241;
constexpr double dx = (1.0 - (-1.0)) / (static_cast<double>(Nx) - 1.0);
constexpr double dy = (0.5 - (-0.5)) / (static_cast<double>(Ny) - 1.0);

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

struct CgResult {
  bool converged = false;
  std::size_t iterations = 0;
  double eps_ratio = 1.0;
};

CgResult solve_cg(Grid psi, ConstGrid IB, double tol, std::size_t max_iter) {
  // initialize vectors
  std::vector<double> R(Nx * Ny), z(Nx * Ny), d(Nx * Ny), Ad(Nx * Ny);
  Grid R_grid{R.data()}, z_grid{z.data()}, d_grid{d.data()}, Ad_grid{Ad.data()};

  // 5-point Laplacian stencil coefficients
  constexpr double A1 = 1.0 / (dy * dy), A2 = 1.0 / (dx * dx),
                   A4 = 1.0 / (dx * dx), A5 = 1.0 / (dy * dy),
                   A3 = -(A1 + A2 + A4 + A5);

  // initialize R (no source term)
  for (std::size_t k = 1; k < Ny - 1; ++k)
    for (std::size_t j = 1; j < Nx - 1; ++j)
      R_grid[j, k] =
          (-A1 * psi[j, k - 1] - A2 * psi[j - 1, k] - A3 * psi[j, k] -
           A4 * psi[j + 1, k] - A5 * psi[j, k + 1]) *
          IB[j, k];

  CgResult result;

  // main CG loop
  double s = 1.0e15, err0 = 0.0;
  for (std::size_t iter = 0; iter < max_iter; ++iter) {
    // precondition with SGS
    // forward sweep
    for (std::size_t k = 1; k < Ny - 1; ++k)
      for (std::size_t j = 1; j < Nx - 1; ++j)
        z_grid[j, k] =
            (R_grid[j, k] - (A2 * z_grid[j - 1, k] + A1 * z_grid[j, k - 1])) /
            A3 * IB[j, k];
    // backward sweep
    for (std::size_t k = Ny - 2; k > 0; --k)
      for (std::size_t j = Nx - 2; j > 0; --j)
        z_grid[j, k] -=
            (A4 * z_grid[j + 1, k] + A5 * z_grid[j, k + 1]) / A3 * IB[j, k];

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
        Ad_grid[j, k] =
            (A1 * d_grid[j, k - 1] + A2 * d_grid[j - 1, k] + A3 * d_grid[j, k] +
             A4 * d_grid[j + 1, k] + A5 * d_grid[j, k + 1]) *
            IB[j, k];

    // calculate alpha denominator
    double den = 0.0;
    for (std::size_t k = 0; k < Ny; ++k)
      for (std::size_t j = 0; j < Nx; ++j)
        den += d_grid[j, k] * Ad_grid[j, k];

    const double alpha = s / den;

    // update solution and residual
    for (std::size_t k = 0; k < Ny; ++k)
      for (std::size_t j = 0; j < Nx; ++j) {
        psi[j, k] += alpha * d_grid[j, k];
        R_grid[j, k] -= alpha * Ad_grid[j, k];
      }

    double err = 0.0;
    for (std::size_t k = 0; k < Ny; ++k)
      for (std::size_t j = 0; j < Nx; ++j)
        err += d_grid[j, k] * d_grid[j, k];

    if (iter == 0)
      err0 = alpha * alpha * err;

    result.iterations = iter + 1;
    result.eps_ratio = std::sqrt(alpha * alpha * err / err0);
    if (result.eps_ratio < tol) {
      result.converged = true;
      break;
    }
  }
  return result;
}

void apply_bc(Grid psi, ConstGrid IB, double alpha, double psi_cyl) {
  const double U_infty = std::cos(alpha);
  const double V_infty = std::sin(alpha);

  // vertical boundaries
  for (std::size_t k = 0; k < Ny; ++k) {
    const double x_left = -1.0;
    const double x_right = 1.0;
    const double y = -0.5 + static_cast<double>(k) * dy;

    psi[0, k] = -U_infty * y + V_infty * x_left;
    psi[Nx - 1, k] = -U_infty * y + V_infty * x_right;
  }

  // horizontal boundaries
  for (std::size_t j = 0; j < Nx; ++j) {
    const double y_bottom = -0.5;
    const double y_top = 0.5;
    const double x = -1.0 + static_cast<double>(j) * dx;

    psi[j, 0] = -U_infty * y_bottom + V_infty * x;
    psi[j, Ny - 1] = -U_infty * y_top + V_infty * x;
  }

  // cylinder
  for (std::size_t k = 0; k < Ny; ++k)
    for (std::size_t j = 0; j < Nx; ++j)
      if (IB[j, k] == 0.0)
        psi[j, k] = psi_cyl;
}

void write_solution(const std::filesystem::path &path, std::string_view comment,
                    ConstGrid psi, ConstGrid IB) {
  std::vector<double> x(psi.size()), y(psi.size());
  Grid x_grid{x.data()};
  Grid y_grid{y.data()};
  for (std::size_t k = 0; k < Ny; ++k)
    for (std::size_t j = 0; j < Nx; ++j) {
      x_grid[j, k] = -1.0 + static_cast<double>(j) * dx;
      y_grid[j, k] = -0.5 + static_cast<double>(k) * dy;
    }

  constexpr std::array<std::string_view, 4> names = {"x", "y", "psi", "iblank"};
  const std::array<std::span<const double>, 4> columns = {
      x, y, std::span<const double>{psi.data_handle(), psi.size()},
      std::span<const double>{IB.data_handle(), IB.size()}};
  num::write_output(path, comment, names, columns);
}

} // namespace

int main() {
  constexpr double lambda = 1.9;
  constexpr std::size_t max_iter = 2000;
  constexpr double tol = 1.0e-4;

  std::vector<double> IB(Nx * Ny, 1.0);
  Grid IB_grid{IB.data()};
  for (std::size_t k = 0; k < Ny; ++k)
    for (std::size_t j = 0; j < Nx; ++j) {
      const double x = -1.0 + static_cast<double>(j) * dx;
      const double y = -0.5 + static_cast<double>(k) * dy;
      if (x * x + 9.0 * y * y <= 0.25 * 0.25 + 1e-12)
        IB_grid[j, k] = 0.0;
    }

  constexpr std::array<std::string_view, 5> part = {"1a", "1a", "1b", "1c",
                                                    "2"};
  constexpr std::array alpha = {0.0, 0.0, 0.0, 0.0, deg2rad(30.0)};
  constexpr std::array psi_cyl = {0.0, 0.0, 0.2, 0.4, 0.0};
  constexpr std::array run_cg = {false, true, false, false, false};

  const auto base_output_path = std::filesystem::path{HW_SOURCE_DIR} / "data";

  const auto describe = [&](std::size_t n, std::string_view solver, double tol,
                            const auto &result, double cpu_time) {
    return std::format("part={} solver={} Nx={} Ny={} alpha={} psi_cyl={} "
                       "tol={} max_iter={} iterations={} eps_ratio={} "
                       "converged={} cpu_time={}",
                       part[n], solver, Nx, Ny, alpha[n], psi_cyl[n], tol,
                       max_iter, result.iterations, result.eps_ratio,
                       result.converged, cpu_time);
  };
  const auto seconds_since = [](std::clock_t start) {
    return static_cast<double>(std::clock() - start) / CLOCKS_PER_SEC;
  };

  for (std::size_t n = 0; n < alpha.size(); ++n) {
    // initialize psi
    std::vector<double> psi(Nx * Ny);
    Grid psi_grid{psi.data()};
    apply_bc(psi_grid, IB_grid, alpha[n], psi_cyl[n]);

    std::string_view solver;
    std::string comment;
    const std::clock_t start = std::clock();
    if (run_cg[n]) {
      const CgResult result = solve_cg(psi_grid, IB_grid, tol, max_iter);
      const double cpu_time = seconds_since(start);
      solver = "cg";
      comment = describe(n, solver, tol, result, cpu_time);
    } else {
      const SorResult result =
          solve_sor(psi_grid, IB_grid, lambda, tol, max_iter);
      const double cpu_time = seconds_since(start);
      solver = "sor";
      comment = describe(n, solver, tol, result, cpu_time) +
                std::format(" lambda={}", lambda);
    }

    const auto path =
        base_output_path / std::format("ex2_{}_{}_psi.csv", part[n], solver);
    write_solution(path, comment, psi_grid, IB_grid);

    std::println("=== RUN SUMMARY ===");
    std::println("path = {}", path.string());
    std::println("{}", comment);
  }
}
