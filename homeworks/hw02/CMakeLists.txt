foreach(ex IN ITEMS ex1b)
  add_executable(hw02_${ex} ${ex}.cpp)
  target_compile_definitions(
    hw02_${ex}
    PRIVATE
    HW_SOURCE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"
  )
  target_link_libraries(hw02_${ex} PRIVATE num)
endforeach()

add_custom_target(hw02_run
                  COMMAND hw02_ex1b
                  # COMMAND uv run plot.py
                  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                  USES_TERMINAL
)
add_dependencies(hw02_run hw02_ex1b)
