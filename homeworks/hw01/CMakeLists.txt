add_executable(hw01_2a ex2a.cpp)

target_compile_definitions(hw01_2a PRIVATE HW01_SOURCE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\")
target_link_libraries(hw01_2a PRIVATE num)
