foreach(ex IN ITEMS ex1 ex2)
  add_executable(hw03_${ex} ${ex}.cpp)
  target_compile_definitions(
    hw03_${ex}
    PRIVATE
    HW_SOURCE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"
  )
  target_link_libraries(hw03_${ex} PRIVATE num)
endforeach()

add_custom_target(hw03_run
                  COMMAND hw03_ex1
                  COMMAND hw03_ex2
                  COMMAND uv run plot.py
                  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                  USES_TERMINAL
)
add_dependencies(hw03_run hw03_ex1 hw03_ex2)
