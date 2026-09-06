cmake_minimum_required(VERSION 4.3)

project(AERE5460Homework
        VERSION 0.0.1
        SPDX_LICENSE "MIT"
        DESCRIPTION "Code for AERE 5460 homework problems."
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(default_flags INTERFACE)
target_compile_options(default_flags INTERFACE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)

enable_testing()

add_subdirectory(num)
add_subdirectory(homeworks/hw01)
