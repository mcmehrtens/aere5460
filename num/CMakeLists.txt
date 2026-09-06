add_library(num INTERFACE)

target_include_directories(num INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(num INTERFACE default_flags)

