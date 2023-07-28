// matrix_mult.cu
extern "C" {

__global__ void matrix_multiply(const float* A, const float* B, float* C, int rows_A, int cols_A, int cols_B) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows_A && col < cols_B) {
        float sum = 0.0;
        for (int i = 0; i < cols_A; i++) {
            sum += A[row * cols_A + i] * B[i * cols_B + col];
        }
        C[row * cols_B + col] = sum;
    }
}

}
