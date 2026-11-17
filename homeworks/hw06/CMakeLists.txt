foreach(ex IN ITEMS ex1 ex2 ex3)
  add_executable(hw06_${ex} ${ex}.cpp)
  target_compile_definitions(
    hw06_${ex}
    PRIVATE
    HW_SOURCE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"
  )
  target_link_libraries(hw06_${ex} PRIVATE num)
endforeach()

add_custom_target(hw06_run
                  COMMAND hw06_ex1
                  COMMAND hw06_ex2
                  COMMAND hw06_ex3
                  COMMAND uv run plot.py
                  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                  USES_TERMINAL
)
add_dependencies(hw06_run hw06_ex1 hw06_ex2 hw06_ex3)
