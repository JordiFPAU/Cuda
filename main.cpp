#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <fmt/core.h>

extern "C" void mat_vec_gpu(const float *d_A, const float *d_B, float *d_c, int N);
int main()
{
    const int N = 1024;             // Tamaño de la matriz y el vector
    size_t matrix_elements = N * N; // Dimensiones de la matriz

    size_t bytes_matrix = matrix_elements * sizeof(float); // calcular cuanta memoria se necesita para la matriz
    size_t bytes_vector = N * sizeof(float);               // calcular cuanta memoria se necesita para el vector

    std::vector<float> h_A(matrix_elements, 1.0); // Crear un vector para la matriz en el host
    std::vector<float> h_B(N, 2.0);               // Crear un vector para el vector en el host
    std::vector<float> h_c(N, 0.0);               // Crear un vector para el resultado en el host
    // Punteros para el GPU
    float *d_A, *d_B, *d_c;
    // Reservar memoria en el GPU
    cudaMalloc((void **)&d_A, bytes_matrix);
    cudaMalloc((void **)&d_B, bytes_vector);
    cudaMalloc((void **)&d_c, bytes_vector);
    // Copiar los datos de CPU A GPU
    cudaMemcpy(d_A, h_A.data(), bytes_matrix, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytes_vector, cudaMemcpyHostToDevice);
    // Llamar a la función del kernel
    mat_vec_gpu(d_A, d_B, d_c, N);
    // Copiar el resultado de GPU a CPU
    cudaMemcpy(h_c.data(), d_c, bytes_vector, cudaMemcpyDeviceToHost);
    // Imprimir los primeros 10 resultados
    for (int i = 0; i < 10; ++i)
    {
        fmt::print("El vector Resultamte es [{}] = {}\n", i, h_c[i]);
    }
    // Liberar memoria en el GPU
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_c);

    return 0;
}