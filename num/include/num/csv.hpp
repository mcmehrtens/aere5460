#pragma once

#include "num/ode.hpp"
#include <algorithm> // IWYU pragma: keep
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

/// Write the sequences in `columns` to a CSV. The format of the CSV is as
/// follows:
///
// clang-format off
/// ```
/// # <comment>
/// column_names[0],column_names[1],...,column_names[column_names.size() - 1]
/// columns[0][0],columns[1][0],...,columns[columns.size() - 1][0]
/// columns[0][1],columns[1][1],...,columns[columns.size() - 1][1]
/// ...
/// ```
// clang-format on
///
/// `comment` should contain relevant metadata to reproduce the solution vector.
/// There should be the same number of column names as columns. All columns must
/// have the same number of elements. Data is written with the default `{}`
/// formatting (shortest round-trip representation).
///
/// @param  path                The path to write the output file to; parent
///                             directories will be created if they do not
///                             already exist, and the file is overwritten if it
///                             exists.
/// @param  comment             The comment to write at the top (line 1).
/// @param  column_names        The list of column headers (line 2).
/// @param  columns             The data sequences to write (lines 3-end).
/// @throws std::runtime_error  If the file at `path` cannot be opened.
inline void write_output(const std::filesystem::path &path,
                         std::string_view comment,
                         std::span<const std::string_view> column_names,
                         std::span<const std::span<const double>> columns) {
  assert(column_names.size() == columns.size());
  assert(columns.size() > 0);

  const std::size_t N = columns[0].size();
  assert(std::ranges::all_of(columns, [N](std::span<const double> column) {
    return column.size() == N;
  }));

  namespace fs = std::filesystem;

  // checks if `path` is somewhere other than `./`
  if (const fs::path parent = path.parent_path(); !parent.empty())
    fs::create_directories(parent);

  std::ofstream out{path};
  if (!out)
    throw std::runtime_error("cannot open " + path.string());

  std::println(out, "# {}", comment);

  for (std::size_t i = 0; i < column_names.size(); ++i)
    std::print(out, "{}{}", i == 0 ? "" : ",", column_names[i]);
  std::println(out);

  for (std::size_t i = 0; i < N; ++i) {
    for (std::size_t j = 0; j < columns.size(); ++j)
      std::print(out, "{}{}", j == 0 ? "" : ",", columns[j][i]);
    std::println(out);
  }
}

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
/// @param  column_names        The list of column headers (line 2).
/// @param  sol                 The solution data structure to write (lines
///                             3-end).
/// @throws std::runtime_error  If the file at `path` cannot be opened.
template <std::size_t Dim>
void write_output(const std::filesystem::path &path, std::string_view comment,
                  std::span<const std::string_view> column_names,
                  const Solution<Dim> &sol) {
  assert(column_names.size() == Dim + 1);

  namespace fs = std::filesystem;

  // checks if `path` is somewhere other than `./`
  if (const fs::path parent = path.parent_path(); !parent.empty())
    fs::create_directories(parent);

  std::ofstream out{path};
  if (!out)
    throw std::runtime_error("cannot open " + path.string());

  std::println(out, "# {}", comment);

  for (std::size_t i = 0; i < column_names.size(); ++i)
    std::print(out, "{}{}", i == 0 ? "" : ",", column_names[i]);
  std::println(out);

  for (std::size_t n = 0; n < sol.t.size(); ++n) {
    std::print(out, "{}", sol.t[n]);
    for (const double v : sol.u[n])
      std::print(out, ",{}", v);
    std::println(out);
  }
}

} // namespace num
