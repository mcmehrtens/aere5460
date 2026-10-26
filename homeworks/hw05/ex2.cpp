/// @file
///
/// AERE 5460, Homework 5, Exercise 2. Solves the Poisson equation
/// \f$\nabla^2\psi=50e^{-100((x-0.5)^2+(y-0.25)^2)}
/// -50e^{-100((x-0.5)^2+(y-0.75)^2)}\f$ on the unit square with finite
/// differences on a non-uniform Cartesian grid (spacing expanding in x and y),
/// using Gauss-Seidel with SOR, \f$\lambda=1.9\f$. J = K = 151.

#include <cstddef>
#include <filesystem>
#include <vector>

namespace {

/// Builds the stretched coordinates
/// \f$s_k=(e^{2k/(n-1)}-1)/(e^2-1)\f$, k = 0, ..., n - 1.
std::vector<double> stretched_grid(std::size_t /*n*/) {
  // TODO: algorithm
  return {};
}

/// Runs Gauss-Seidel with SOR on the Poisson equation on the non-uniform grid
/// given by `x` and `y`.
///
/// @param psi holds the boundary values on entry and the solution on exit
/// @param lambda relaxation parameter
void solve_sor(std::vector<double> & /*psi*/, const std::vector<double> & /*x*/,
               const std::vector<double> & /*y*/, double /*lambda*/) {
  // TODO: algorithm
}

} // namespace

int main() {
  const auto base_output_path = std::filesystem::path{HW_SOURCE_DIR} / "data";

  // TODO: build the grid, set boundary values, solve, write x, y and psi
}
