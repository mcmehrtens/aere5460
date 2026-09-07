#pragma once

#include "num/ode.hpp"
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <ostream> // IWYU pragma: keep
#include <print>
#include <span>
#include <stdexcept>
#include <string_view>

namespace num {

template <std::size_t Dim>
void write_output(const std::filesystem::path &path, std::string_view comment,
                  std::span<const std::string_view> columns,
                  const Solution<Dim> &sol) {
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
