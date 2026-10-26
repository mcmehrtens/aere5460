foreach(ex IN ITEMS ex1 ex2 ex3)
  add_executable(hw05_${ex} ${ex}.cpp)
  target_compile_definitions(
    hw05_${ex}
    PRIVATE
    HW_SOURCE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"
  )
  target_link_libraries(hw05_${ex} PRIVATE num)
endforeach()

add_custom_target(hw05_run
                  COMMAND hw05_ex1
                  COMMAND hw05_ex2
                  COMMAND hw05_ex3
                  COMMAND uv run plot.py
                  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                  USES_TERMINAL
)
add_dependencies(hw05_run hw05_ex1 hw05_ex2 hw05_ex3)
