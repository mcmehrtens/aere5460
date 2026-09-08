foreach(ex IN ITEMS ex2a ex2b)
  add_executable(hw01_${ex} ${ex}.cpp)
  target_compile_definitions(
    hw01_${ex}
    PRIVATE
    HW01_SOURCE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"
  )
  target_link_libraries(hw01_${ex} PRIVATE num)
endforeach()
