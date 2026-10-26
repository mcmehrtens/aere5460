/// @file
///
/// AERE 5460, Homework 5, Exercise 1. Solves the Laplace equation
/// \f$\frac{\partial^2\psi}{\partial x^2}+\frac{\partial^2\psi}{\partial
/// y^2}=0\f$ on the unit square with finite differences, using Gauss-Seidel
/// with successive over-relaxation (SOR). \f$\psi=\sin^2(3n\pi y)\f$ on x = 0,
/// \f$\psi=\sin^2(3n\pi x)\f$ on y = 0 and \f$\psi=0\f$ on x = 1 and y = 1.

#include <cstddef>
#include <filesystem>
#include <vector>

namespace {

/// Outcome of one SOR solve.
struct SorResult {
  /// L2 norm of the update, \f$\epsilon\f$, after each iteration. Entry 0 is
  /// \f$\epsilon_0\f$.
  std::vector<double> residual;
};

/// Runs Gauss-Seidel with SOR on the Laplace equation until the residual has
/// dropped by `tol` or `max_iter` iterations have been taken.
///
/// @param psi holds the boundary values on entry and the solution on exit
/// @param lambda relaxation parameter
/// @param tol stop when \f$\epsilon/\epsilon_0\f$ falls below this
/// @param max_iter iteration cap
SorResult solve_sor(std::vector<double> & /*psi*/, std::size_t /*Nx*/,
                    std::size_t /*Ny*/, double /*lambda*/, double /*tol*/,
                    std::size_t /*max_iter*/) {
  // TODO: algorithm
  return {};
}

} // namespace

int main() {
  const auto base_output_path = std::filesystem::path{HW_SOURCE_DIR} / "data";

  // TODO: loop over n in {1, 3} and the 15x15 and 151x101 grids, write one
  // CSV per case (solution), and a residual-history CSV for 151x101, n = 1,
  // lambda in {1, 0.5, 1.5, 1.95}
}
