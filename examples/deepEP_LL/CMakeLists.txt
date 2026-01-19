add_executable(low_latency
  low_latency.cpp
)

target_link_libraries(
  low_latency
  PRIVATE
    roc::rocshmem
    $<TARGET_NAME_IF_EXISTS:MPI::MPI_CXX>
)
