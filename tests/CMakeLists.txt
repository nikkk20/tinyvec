add_executable(test_vector_ops test_vector_ops.cpp)
target_link_libraries(test_vector_ops PRIVATE tinyvec_core)

add_test(NAME test_vector_ops COMMAND test_vector_ops)
