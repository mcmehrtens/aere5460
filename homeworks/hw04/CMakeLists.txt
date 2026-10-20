foreach(ex IN ITEMS ex1 ex2 ex3)
  add_executable(hw04_${ex} ${ex}.cpp)
  target_compile_definitions(
    hw04_${ex}
    PRIVATE
    HW_SOURCE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"
  )
  target_link_libraries(hw04_${ex} PRIVATE num)
endforeach()

add_custom_target(hw04_run
                  COMMAND hw04_ex1
                  COMMAND hw04_ex2
                  COMMAND hw04_ex3
                  COMMAND uv run plot.py
                  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                  USES_TERMINAL
)
add_dependencies(hw04_run hw04_ex1 hw04_ex2 hw04_ex3)
