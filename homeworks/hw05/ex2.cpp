/// @file
///
/// AERE 5460, Homework 5, Exercise 2. Solves the Poisson equation
/// \f$\nabla^2\psi=50e^{-100((x-0.5)^2+(y-0.25)^2)}
/// -50e^{-100((x-0.5)^2+(y-0.75)^2)}\f$ on the unit square with finite
/// differences on a non-uniform Cartesian grid (spacing expanding in x and y),
/// using Gauss-Seidel with SOR, \f$\lambda=1.9\f$. J = K = 151.

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

struct Weight {
  double minus, center, plus;
};

using GridExtents = std::dextents<std::size_t, 2>;
using Grid = std::mdspan<double, GridExtents, std::layout_left>;
using ConstGrid = std::mdspan<const double, GridExtents, std::layout_left>;

bool solve_sor(Grid psi, ConstGrid f, std::span<const Weight> w,
               std::span<const Weight> v, double lambda) {
  const std::size_t J = psi.extent(0), K = psi.extent(1);
  constexpr std::size_t max_iter = 2000;
  constexpr double tol = 1.0e-4;
  double eps0 = 0.0;
  bool converged = false;
  for (std::size_t iter = 0; iter < max_iter; ++iter) {
    // Gauss–Seidel sweep
    double eps = 0.0;
    for (std::size_t k = 1; k < K - 1; ++k)
      for (std::size_t j = 1; j < J - 1; ++j) {
        const double psi_new =
            (f[j, k] - v[k].minus * psi[j, k - 1] - w[j].minus * psi[j - 1, k] -
             w[j].plus * psi[j + 1, k] - v[k].plus * psi[j, k + 1]) /
            (w[j].center + v[k].center);
        const double Deltapsi = psi_new - psi[j, k];
        eps += Deltapsi * Deltapsi;
        psi[j, k] = psi[j, k] + lambda * Deltapsi;
      }

    eps = std::sqrt(eps / (static_cast<double>((J - 2) * (K - 2))));

    if (iter == 0)
      eps0 = eps;
    if (eps / eps0 < tol) {
      converged = true;
      break;
    }
  }
  return converged;
}
} // namespace

int main() {
  const auto base_output_path = std::filesystem::path{HW_SOURCE_DIR} / "data";

  constexpr std::size_t J = 151, K = 151;
  constexpr double lambda = 1.9;

  std::vector<double> psi(J * K);
  Grid psi_grid{psi.data(), J, K};

  // gonna shift the grid to be 0-indexed
  // build the grid
  const auto stretched = [](std::size_t N) {
    std::vector<double> s(N);
    for (std::size_t i = 0; i < N; ++i)
      s[i] =
          (std::exp(2.0 * static_cast<double>(i) / static_cast<double>(N - 1)) -
           1.0) /
          (std::exp(2.0) - 1.0);
    return s;
  };
  const std::vector<double> x = stretched(J);
  const std::vector<double> y = stretched(K);

  // apply BCs
  for (std::size_t j = 0; j < J; ++j) {
    psi_grid[j, 0] = 0.0;
    psi_grid[j, K - 1] = -1.0;
  }
  for (std::size_t k = 0; k < K; ++k) {
    psi_grid[0, k] = -y[k];
    psi_grid[J - 1, k] = -y[k];
  }

  // calculate weights
  const auto weights = [](std::span<const double> coords) {
    const std::size_t N = coords.size();
    std::vector<Weight> s(N);
    for (std::size_t i = 1; i < N - 1; ++i) {
      const double hminus = coords[i] - coords[i - 1];
      const double hplus = coords[i + 1] - coords[i];
      s[i].minus = 2.0 / (hminus * (hminus + hplus));
      s[i].center = -2.0 / (hminus * hplus);
      s[i].plus = 2.0 / (hplus * (hminus + hplus));
    }
    return s;
  };
  const std::vector<Weight> w = weights(x);
  const std::vector<Weight> v = weights(y);

  // calculate RHS
  const std::vector<double> f = [&] {
    std::vector<double> out(J * K);
    Grid out_grid{out.data(), J, K};
    for (std::size_t k = 0; k < K; ++k)
      for (std::size_t j = 0; j < J; ++j) {
        const double xj = x[j];
        const double yk = y[k];
        out_grid[j, k] = 50.0 * std::exp(-100.0 * ((xj - 0.5) * (xj - 0.5) +
                                                   (yk - 0.25) * (yk - 0.25))) -
                         50.0 * std::exp(-100.0 * ((xj - 0.5) * (xj - 0.5) +
                                                   (yk - 0.75) * (yk - 0.75)));
      }
    return out;
  }();
  const ConstGrid f_grid{f.data(), J, K};

  const bool converged = solve_sor(psi_grid, f_grid, w, v, lambda);

  // one row per node, x varying fastest
  std::vector<double> x_nodes(J * K), y_nodes(J * K);
  Grid x_nodes_grid{x_nodes.data(), J, K};
  Grid y_nodes_grid{y_nodes.data(), J, K};
  for (std::size_t k = 0; k < K; ++k)
    for (std::size_t j = 0; j < J; ++j) {
      x_nodes_grid[j, k] = x[j];
      y_nodes_grid[j, k] = y[k];
    }

  const auto path = base_output_path / "ex2_psi.csv";
  const auto comment = std::format("Nx={} Ny={} lambda={} converged={}", J, K,
                                   lambda, converged);
  constexpr std::array<std::string_view, 3> names = {"x", "y", "psi"};
  const std::array<std::span<const double>, 3> columns = {x_nodes, y_nodes,
                                                          psi};
  num::write_output(path, comment, names, columns);

  std::println("=== RUN SUMMARY ===");
  std::println("path = {}", path.string());
  std::println("{}", comment);
}
