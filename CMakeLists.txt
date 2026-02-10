cmake_minimum_required(VERSION 3.18) 

project(EvaluacionSumativa LANGUAGES CXX CUDA) 

# Configuración CUDA
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit REQUIRED) 
find_package(fmt CONFIG REQUIRED)
find_package(SFML COMPONENTS Network Graphics Window Audio System CONFIG REQUIRED)

add_executable(EvaluacionSumativa
    main.cpp
    kernel.cu
)

# Propiedades CUDA
set_target_properties(EvaluacionSumativa PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(EvaluacionSumativa PRIVATE 
    fmt::fmt
    SFML::Graphics
    CUDA::cudart
)