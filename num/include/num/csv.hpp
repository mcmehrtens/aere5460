#pragma once

#include "num/ode.hpp"
#include <cassert>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <ostream> // IWYU pragma: keep
#include <print>
#include <span>
#include <stdexcept>
#include <string_view>

namespace num {

/// Write the solution vectors to a CSV. The format of the CSV is as follows:
///
// clang-format off
/// ```
/// # <comment>
/// columns[0],columns[1],...,columns[columns.size() - 1]
/// sol.t[0],sol.u[0][0],sol.u[0][1],...,sol.u[0][Dim - 1]
/// sol.t[1],sol.u[1][0],sol.u[1][1],...,sol.u[1][Dim - 1]
/// ...
/// ```
// clang-format on
///
/// `comment` should contain relevant metadata to reproduce the solution vector.
/// There should be `Dim + 1` column names (one for time and one per state
/// variable). Data is written with the default `{}` formatting (shortest
/// round-trip representation).
///
/// @tparam Dim                 The number of state variables.
/// @param  path                The path to write the output file to; parent
///                             directories will be created if they do not
///                             already exist, and the file is overwritten if it
///                             exists.
/// @param  comment             The comment to write at the top (line 1).
/// @param  columns             The list of column headers (line 2).
/// @param  sol                 The solution data structure to write (lines
///                             3-end).
/// @throws std::runtime_error  If the file at `path` cannot be opened.
template <std::size_t Dim>
void write_output(const std::filesystem::path &path, std::string_view comment,
                  std::span<const std::string_view> columns,
                  const Solution<Dim> &sol) {
  assert(columns.size() == Dim + 1);

  namespace fs = std::filesystem;

  // checks if `path` is somewhere other than `./`
  if (const fs::path parent = path.parent_path(); !parent.empty())
    fs::create_directories(parent);

  std::ofstream out{path};
  if (!out)
    throw std::runtime_error("cannot open " + path.string());

  std::println(out, "# {}", comment);

  for (std::size_t i = 0; i < columns.size(); ++i)
    std::print(out, "{}{}", i == 0 ? "" : ",", columns[i]);
  std::println(out);

  for (std::size_t n = 0; n < sol.t.size(); ++n) {
    std::print(out, "{}", sol.t[n]);
    for (const double v : sol.u[n])
      std::print(out, ",{}", v);
    std::println(out);
  }
}

} // namespace num
