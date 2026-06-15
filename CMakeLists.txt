cmake_minimum_required(VERSION 3.12)
project(gnn_runtime CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)
find_package(ZLIB REQUIRED)

if(NOT MPI_CXX_COMPILER)
    find_program(MPI_CXX_COMPILER NAMES mpic++ mpicxx mpic++)
endif()

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

set(SOURCES
    src/assertions.cpp
    src/graph.cpp
    src/vertex_map.cpp
    src/frontier.cpp
    src/inbox.cpp
    src/outbox.cpp
    src/compute.cpp
    src/gnn_layer.cpp
    src/gnn_model.cpp
    src/partition.cpp
    src/loader.cpp
    src/mpi_exchange.cpp
    src/superstep.cpp
    src/local_backward.cpp
    src/gradient_exchange.cpp
    src/runtime.cpp
    src/checkpoint.cpp
    src/dataset.cpp
    src/distributed_optimizer.cpp
    src/trainer.cpp
    src/main.cpp
)

add_executable(gnn_runtime ${SOURCES})

target_include_directories(gnn_runtime PRIVATE
    include
    ${MPI_CXX_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

target_compile_options(gnn_runtime PRIVATE
    -Wall -Wextra -O2
    ${TORCH_CXX_FLAGS}
)

target_link_libraries(gnn_runtime PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    ${TORCH_LIBRARIES}
    ZLIB::ZLIB
)
