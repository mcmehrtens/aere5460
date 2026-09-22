add_executable(test_tridiag tridiag_residual.cpp)
target_link_libraries(test_tridiag PRIVATE num)
add_test(NAME tridiag_residual COMMAND test_tridiag)
