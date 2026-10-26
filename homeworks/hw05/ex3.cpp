/// @file
///
/// AERE 5460, Homework 5, Exercise 3. Solves the Poisson equation
/// \f$\nabla^2\psi=2000[\sin(4\pi x)\sin(4\pi y)]^5\f$ on the unit square
/// with finite differences on a 151 x 151 grid, using the conjugate gradient
/// method with symmetric Gauss-Seidel preconditioning.
/// \f$\psi(x,0)=\sin^2(\pi x)+5\sin^2(4\pi x)+10\sin^2(8\pi x)\f$,
/// \f$\psi(0,y)=\sin^2(4\pi y)\f$, \f$\psi=0\f$ on x = 1 and y = 1.

#include <cstddef>
#include <filesystem>
#include <vector>

namespace {

/// Applies the symmetric Gauss-Seidel preconditioner, solving
/// \f$M z = r\f$ with one forward and one backward sweep.
void precondition(const std::vector<double> & /*r*/,
                  std::vector<double> & /*z*/) {
  // TODO: algorithm
}

/// Runs preconditioned conjugate gradient until the residual,
/// \f$\|\Delta\psi\|_{L2}/\|\Delta\psi\|_{L2}(0)\f$, falls below `tol`.
///
/// @param psi holds the boundary values on entry and the solution on exit
/// @param tol stop when the relative residual falls below this
/// @return the relative residual after each iteration, entry 0 being 1
std::vector<double> solve_cg(std::vector<double> & /*psi*/, double /*tol*/) {
  // TODO: algorithm
  return {};
}

} // namespace

int main() {
  const auto base_output_path = std::filesystem::path{HW_SOURCE_DIR} / "data";

  // TODO: set up the grid, source and boundary values, solve, write psi and
  // the residual history
}
