add_executable(test_vector_ops test_vector_ops.cpp)
target_link_libraries(test_vector_ops PRIVATE tinyvec_core)

add_test(NAME test_vector_ops COMMAND test_vector_ops)

add_executable(test_flat_index test_flat_index.cpp)
target_link_libraries(test_flat_index PRIVATE tinyvec_core)

add_test(NAME test_flat_index COMMAND test_flat_index)
