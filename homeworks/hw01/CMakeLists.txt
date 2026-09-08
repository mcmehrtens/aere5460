foreach(ex IN ITEMS ex2a ex2b ex3)
  add_executable(hw01_${ex} ${ex}.cpp)
  target_compile_definitions(
    hw01_${ex}
    PRIVATE
    HW01_SOURCE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"
  )
  target_link_libraries(hw01_${ex} PRIVATE num)
endforeach()

add_custom_target(hw01_run
                  COMMAND hw01_ex2a
                  COMMAND hw01_ex2b
                  COMMAND hw01_ex3
                  COMMAND uv run plot.py
                  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                  USES_TERMINAL
)
add_dependencies(hw01_run hw01_ex2a hw01_ex2b hw01_ex3)
