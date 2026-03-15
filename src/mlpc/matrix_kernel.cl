__kernel void vector_add(__global const float *A,
                         __global const float *B,
                         __global float *C) {
    int gid = get_global_id(0);

    C[gid] = A[gid] + B[gid];
}

__kernel void vector_multiply(__global const float *A,
                              __global const float *B,
                              __global float *C) {
    int gid = get_global_id(0);

    C[gid] = A[gid] * B[gid];
}

__kernel void vector_divide(__global const float *A,
                            __global const float *B,
                            __global float *C) {
    int gid = get_global_id(0);

    C[gid] = A[gid] / B[gid];
}

__kernel void matrix_multiply(__global float* C, 
                              __global const float* A, 
                              __global const float* B, 
                              int widthA, int heightA, int widthB) {
    int col = get_global_id(0);
    int row = get_global_id(1);

    float sum = 0.0f;
    for (int k = 0; k < widthA; ++k) {
        sum += A[row * widthA + k] * B[k * widthB + col];
    }

    C[row * widthB + col] = sum;
}


/*
#define TS 16
__kernel void matrix_multiply(__global float* C, 
                              __global const float* A, 
                              __global const float* B, 
                              int K, int M, int N) {
    const int row = get_local_id(0); // Local row ID (max: TS)
    const int col = get_local_id(1); // Local col ID (max: TS)
    const int globalRow = TS*get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = TS*get_group_id(1) + col; // Col ID of C (0..N)


    // Local memory to fit a tile of TS*TS elements of A and B
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];
 
    // Initialise the accumulation register
    float acc = 0.0f;
    
    // Loop over all tiles
    const int numTiles = K/TS;
    for (int t=0; t<numTiles; t++) {
 
        // Load one tile of A and B into local memory
        const int tiledRow = TS*t + row;
        const int tiledCol = TS*t + col;
        Asub[col][row] = A[tiledCol*M + globalRow];
        Bsub[col][row] = B[globalCol*K + tiledRow];
 
        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Perform the computation for a single tile
        for (int k=0; k<TS; k++) {
            acc += Asub[k][row] * Bsub[col][k];
        }
 
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
 
    // Store the final result in C
    C[globalCol*M + globalRow] = acc;

}*/

__kernel void matrix_multiply_transpose(__global float* C, 
                                        __global const float* A, 
                                        __global const float* B, 
                                        int widthA, int heightA, int widthB,
                                        int batch_size) {
    int col = get_global_id(0);
    int row = get_global_id(1);

    float sum = 0.0f;
    for (int k = 0; k < widthA; ++k) {
        sum += A[col * widthA + k] * B[k * widthB + row];
    }

    C[row * heightA + col] = sum / (float)batch_size;
}

__kernel void matrix_multiply_add(__global float* C, 
                                  __global const float* A, 
                                  __global const float* B, 
                                  __global const float* D, 
                                  int widthA, int heightA, int widthB) {
    int col = get_global_id(0);
    int row = get_global_id(1);

    float sum = 0.0f;
    for (int k = 0; k < widthA; ++k) {
        sum += A[row * widthA + k] * B[k * widthB + col];
    }

    C[row * widthB + col] = sum + D[row * widthB + col];
}

__kernel void matrix_ff_relu(__global float* C, 
                             __global const float* A, 
                             __global const float* B, 
                             __global const float* D, 
                             int widthA, int heightA, int widthB) {
    int col = get_global_id(0);
    int row = get_global_id(1);

    float sum = 0.0f;
    for (int k = 0; k < widthA; ++k) {
        sum += A[row * widthA + k] * B[k * widthB + col];
    }

    C[row * widthB + col] = sum + D[row * widthB + col];

    if (C[row * widthB + col] < 0) {
        C[row * widthB + col]  = 0;
    }
}

__kernel void matrix_ff_tanh(__global float* C, 
                             __global const float* A, 
                             __global const float* B, 
                             __global const float* D, 
                             int widthA, int heightA, int widthB) {
    int col = get_global_id(0);
    int row = get_global_id(1);

    float sum = 0.0f;
    for (int k = 0; k < widthA; ++k) {
        sum += A[row * widthA + k] * B[k * widthB + col];
    }

    C[row * widthB + col] = tanh(sum + D[row * widthB + col]);
}

__kernel void matrix_transpose(__global const float* A, 
                               __global float* result,
                               int A_cols, int result_cols) {
    int col = get_global_id(0);
    int row = get_global_id(1);

    result[col * result_cols + row] = A[row * A_cols + col];
}

__kernel void matrix_transpose_apply_anti_linear(__global const float* A, 
                                                 __global float* result,
                                                 int A_cols, int result_cols) {
    int col = get_global_id(0);
    int row = get_global_id(1);

    result[col * result_cols + row] = 1;
}

__kernel void matrix_transpose_apply_anti_relu(__global const float* A, 
                                               __global float* result,
                                               int A_cols, int result_cols) {
    int col = get_global_id(0);
    int row = get_global_id(1);

    if (A[row * A_cols + col] > 0)
        result[col * result_cols + row] = 1;
    else
        result[col * result_cols + row] = 0;
}

__kernel void matrix_transpose_apply_anti_tanh(__global const float* A, 
                                               __global float* result,
                                               int A_cols, int result_cols) {
    int col = get_global_id(0);
    int row = get_global_id(1);

    result[col * result_cols + row] = 1 - A[row * A_cols + col] * A[row * A_cols + col];
}

__kernel void matrix_sum_rows_transpose_p1(__global float* C, 
                                           __global const float* A,
                                           int C_Col, int A_Col, int A_Row) {
    int col = get_global_id(0);

    int idx = col * C_Col;
    C[idx] = 0;
    for (int row = 0; row < A_Row; row++)
        C[idx] += A[row * A_Col + col];
}

__kernel void matrix_sum_rows_transpose_p2(__global float* C, 
                                           int C_Col, int C_Row, int batch_size) {
    int col = get_global_id(0);

    if (col == 0)
    {
        for (int row = 0; row < C_Row; row++)
            C[row * C_Col] = C[row * C_Col] / (float)batch_size;
    }
    else
    {
        for (int row = 0; row < C_Row; row++)
            C[row * C_Col + col] = C[row * C_Col] / (float)batch_size;
    }
}

__kernel void adam_optimize(__global float* weight, 
                            __global float* mw, 
                            __global float* vw,
                            __global const float* gradient_weights, 
                            int weight_col, float beta1t, float beta2t) {

    int col = get_global_id(0);
    int row = get_global_id(1);

    const float adam_alpha = 0.001;
    const float adam_beta1 = 0.9;
    const float adam_beta2 = 0.999;
    const float adam_epsilon = 1e-7;

    int idx = row * weight_col + col;

    mw[idx] = adam_beta1 * mw[idx] + (1 - adam_beta1) * gradient_weights[idx];
    vw[idx] = adam_beta2 * vw[idx] + (1 - adam_beta2) * gradient_weights[idx] * gradient_weights[idx];

    float mw1 = mw[idx] / (1 - beta1t);
    float vw1 = vw[idx] / (1 - beta2t);

    weight[idx] -= adam_alpha * (mw1 / (sqrt(vw1) + adam_epsilon));
}
