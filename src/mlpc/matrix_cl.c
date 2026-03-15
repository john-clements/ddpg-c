#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matrix.h"
#include "activation.h"

#ifdef OPEN_CL_EN

#define MAX_KERNEL_SIZE (8192*4)

//open_cl_ctx g_cl_ctx = {0};

int open_cl_init(open_cl_ctx* ctx)
{
    if (ctx->cl_ctx_init)
        return 1;

    FILE* fp = fopen("/home/john/source/neuralnetsim3/ddpg/src/mlpc/matrix_kernel.cl", "r");
    if (!fp)
    {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }

    char* kernel_str = (char*)malloc(MAX_KERNEL_SIZE);
    size_t kernel_size = fread(kernel_str, 1, MAX_KERNEL_SIZE, fp);
    fclose(fp);

    if (kernel_size == MAX_KERNEL_SIZE)
    {
        printf("ERROR: KERNEL TOO LARGE\n");
        exit(1);
    }

    memset((void*)ctx, 0, sizeof(open_cl_ctx));

    ctx->ret = clGetPlatformIDs(1, &ctx->platform_id, &ctx->num_platforms);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clGetPlatformIDs) -> %d\n", ctx->ret);
        return 0;
    }

    ctx->ret = clGetDeviceIDs(ctx->platform_id,
                              CL_DEVICE_TYPE_GPU,
                              1,
                              &ctx->device_id,
                              &ctx->num_devices);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clGetDeviceIDs) -> %d\n", ctx->ret);
        return 0;
    }

    ctx->context = clCreateContext(NULL, 1, &ctx->device_id, NULL, NULL, &ctx->ret);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clCreateContext) -> %d\n", ctx->ret);
        return 0;
    }

    ctx->command_queue = clCreateCommandQueueWithProperties(ctx->context,
                                                            ctx->device_id,
                                                            NULL,
                                                            &ctx->ret);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clCreateCommandQueue) -> %d\n", ctx->ret);
        return 0;
    }

    ctx->program = clCreateProgramWithSource(ctx->context,
                                             1,
                                             (const char **)&kernel_str,
                                             (const size_t *)&kernel_size,
                                             &ctx->ret);

    free(kernel_str);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clCreateProgramWithSource) -> %d\n", ctx->ret);
        return 0;
    }

    ctx->ret = clBuildProgram(ctx->program, 1, &ctx->device_id, NULL, NULL, NULL);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clBuildProgram) -> %d\n", ctx->ret);
        return 0;
    }

    ctx->kernel_vector_add = clCreateKernel(ctx->program, "vector_add", &ctx->ret);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clCreateKernel) -> %d\n", ctx->ret);
        return 0;
    }

    ctx->kernel_vector_multiply = clCreateKernel(ctx->program, "vector_multiply", &ctx->ret);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clCreateKernel) -> %d\n", ctx->ret);
        return 0;
    }

    ctx->kernel_matrix_multiply = clCreateKernel(ctx->program, "matrix_multiply", &ctx->ret);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clCreateKernel) -> %d\n", ctx->ret);
        return 0;
    }

    ctx->kernel_matrix_multiply_transpose = clCreateKernel(ctx->program, "matrix_multiply_transpose", &ctx->ret);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clCreateKernel) -> %d\n", ctx->ret);
        return 0;
    }

    ctx->kernel_matrix_multiply_add = clCreateKernel(ctx->program, "matrix_multiply_add", &ctx->ret);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clCreateKernel) -> %d\n", ctx->ret);
        return 0;
    }

    ctx->kernel_matrix_ff_relu = clCreateKernel(ctx->program, "matrix_ff_relu", &ctx->ret);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clCreateKernel) -> %d\n", ctx->ret);
        return 0;
    }

    ctx->kernel_matrix_ff_tanh = clCreateKernel(ctx->program, "matrix_ff_tanh", &ctx->ret);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clCreateKernel) -> %d\n", ctx->ret);
        return 0;
    }

    ctx->kernel_matrix_transpose = clCreateKernel(ctx->program, "matrix_transpose", &ctx->ret);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clCreateKernel) -> %d\n", ctx->ret);
        return 0;
    }

    ctx->kernel_matrix_transpose_apply_anti_linear = clCreateKernel(ctx->program, "matrix_transpose_apply_anti_linear", &ctx->ret);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clCreateKernel) -> %d\n", ctx->ret);
        return 0;
    }

    ctx->kernel_matrix_transpose_apply_anti_tanh = clCreateKernel(ctx->program, "matrix_transpose_apply_anti_tanh", &ctx->ret);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clCreateKernel) -> %d\n", ctx->ret);
        return 0;
    }

    ctx->kernel_matrix_transpose_apply_anti_relu = clCreateKernel(ctx->program, "matrix_transpose_apply_anti_relu", &ctx->ret);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clCreateKernel) -> %d\n", ctx->ret);
        return 0;
    }

    ctx->kernel_matrix_sum_rows_transpose_p1 = clCreateKernel(ctx->program, "matrix_sum_rows_transpose_p1", &ctx->ret);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clCreateKernel) -> %d\n", ctx->ret);
        return 0;
    }

    ctx->kernel_matrix_sum_rows_transpose_p2 = clCreateKernel(ctx->program, "matrix_sum_rows_transpose_p2", &ctx->ret);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clCreateKernel) -> %d\n", ctx->ret);
        return 0;
    }

    ctx->kernel_adam_optimize = clCreateKernel(ctx->program, "adam_optimize", &ctx->ret);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clCreateKernel) -> %d\n", ctx->ret);
        return 0;
    }

    ctx->cl_ctx_init = 1;

    return 1;
}

int matrix_cl_create(open_cl_ctx* ctx, Matrix* matrix)
{
    matrix->total_size = matrix->rows * matrix->columns;

    matrix->mem_obj = clCreateBuffer(ctx->context,
                                     CL_MEM_READ_WRITE,
                                     matrix->total_size * sizeof(float),
                                     NULL,
                                     &ctx->ret);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clCreateBuffer) -> %d\n", ctx->ret);
        return 0;
    }

    return 1;
}

int matrix_cl_dystroy(open_cl_ctx* ctx, Matrix* matrix)
{
    ctx->ret = clReleaseMemObject(matrix->mem_obj);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clReleaseMemObject) -> %d\n", ctx->ret);
        return 0;
    }

    return 1;
}

int matrix_cl_copy_to_device(open_cl_ctx* ctx, Matrix* A)
{
    // Copy host data to device buffers
    ctx->ret = clEnqueueWriteBuffer(ctx->command_queue, A->mem_obj, CL_TRUE, 0, A->total_size * sizeof(float), A->data, 0, NULL, NULL);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clEnqueueWriteBuffer) -> %d\n", ctx->ret);
        return 0;
    }

    return 1;
}

int matrix_cl_copy_to_host(open_cl_ctx* ctx, Matrix* A)
{
    // Read the result back to host memory
    ctx->ret = clEnqueueReadBuffer(ctx->command_queue,
                                   A->mem_obj,
                                   CL_TRUE,
                                   0,
                                   A->total_size * sizeof(float),
                                   A->data,
                                   0,
                                   NULL,
                                   NULL);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clEnqueueReadBuffer) -> %d\n", ctx->ret);
        return 0;
    }

    return 1;
}
/*
int matrix_cl_add(Matrix* dst, Matrix* src)
{
    // Copy host data to device buffers
    ctx->ret = clEnqueueWriteBuffer(ctx->command_queue, src->mem_obj, CL_TRUE, 0, src->total_size * sizeof(float), src->data, 0, NULL, NULL);
    ctx->ret = clEnqueueWriteBuffer(ctx->command_queue, dst->mem_obj, CL_TRUE, 0, dst->total_size * sizeof(float), dst->data, 0, NULL, NULL);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clEnqueueWriteBuffer) -> %d\n", ctx->ret);
        return 0;
    }

    // Set the arguments of the kernel
    ctx->ret = clSetKernelArg(ctx->kernel_vector_add, 0, sizeof(cl_mem), (void *)&src->mem_obj);
    ctx->ret = clSetKernelArg(ctx->kernel_vector_add, 1, sizeof(cl_mem), (void *)&dst->mem_obj);
    ctx->ret = clSetKernelArg(ctx->kernel_vector_add, 2, sizeof(cl_mem), (void *)&dst->mem_obj);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clSetKernelArg) -> %d\n", ctx->ret);
        return 0;
    }

    size_t global_item_size = dst->total_size;
    size_t local_item_size = 64;
    ctx->ret = clEnqueueNDRangeKernel(ctx->command_queue,
                                          ctx->kernel_vector_add,
                                          1,
                                          NULL,
                                          &global_item_size,
                                          &local_item_size,
                                          0,
                                          NULL,
                                          NULL);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clEnqueueNDRangeKernel) -> %d\n", ctx->ret);
        return 0;
    }

    // Read the result back to host memory
    ctx->ret = clEnqueueReadBuffer(ctx->command_queue,
                                       dst->mem_obj,
                                       CL_TRUE,
                                       0,
                                       dst->total_size * sizeof(float),
                                       dst->data,
                                       0,
                                       NULL,
                                       NULL);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clEnqueueReadBuffer) -> %d\n", ctx->ret);
        return 0;
    }

    return 1;
}

int matrix_cl_multiply_add(Matrix* A, Matrix* B, Matrix* add, Matrix* result)
{
    // Copy host data to device buffers
    ctx->ret = clEnqueueWriteBuffer(ctx->command_queue, A->mem_obj, CL_TRUE, 0, A->total_size * sizeof(float), A->data, 0, NULL, NULL);
    ctx->ret = clEnqueueWriteBuffer(ctx->command_queue, B->mem_obj, CL_TRUE, 0, B->total_size * sizeof(float), B->data, 0, NULL, NULL);
    ctx->ret = clEnqueueWriteBuffer(ctx->command_queue, add->mem_obj, CL_TRUE, 0, add->total_size * sizeof(float), add->data, 0, NULL, NULL);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clEnqueueWriteBuffer) -> %d\n", ctx->ret);
        return 0;
    }

    // Set the arguments of the kernel
    ctx->ret = clSetKernelArg(ctx->kernel_matrix_multiply_add, 0, sizeof(cl_mem), (void *)&result->mem_obj);
    ctx->ret = clSetKernelArg(ctx->kernel_matrix_multiply_add, 1, sizeof(cl_mem), (void *)&A->mem_obj);
    ctx->ret = clSetKernelArg(ctx->kernel_matrix_multiply_add, 2, sizeof(cl_mem), (void *)&B->mem_obj);
    ctx->ret = clSetKernelArg(ctx->kernel_matrix_multiply_add, 3, sizeof(cl_mem), (void *)&add->mem_obj);
    ctx->ret = clSetKernelArg(ctx->kernel_matrix_multiply_add, 4, sizeof(int), (void *)&A->columns);
    ctx->ret = clSetKernelArg(ctx->kernel_matrix_multiply_add, 5, sizeof(int), (void *)&A->rows);
    ctx->ret = clSetKernelArg(ctx->kernel_matrix_multiply_add, 6, sizeof(int), (void *)&B->columns);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clSetKernelArg) -> %d\n", ctx->ret);
        return 0;
    }

    size_t localWorkSize[2], globalWorkSize[2];

   localWorkSize[0] = 16;
   localWorkSize[1] = 16;
   globalWorkSize[0] = B->columns;
   globalWorkSize[1] = A->rows;

    //size_t global_item_size = result->total_size;
    //size_t local_item_size = 64;
    ctx->ret = clEnqueueNDRangeKernel(ctx->command_queue,
                                          ctx->kernel_matrix_multiply_add,
                                          2,
                                          NULL,
                                          globalWorkSize,
                                          localWorkSize,
                                          0,
                                          NULL,
                                          NULL);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clEnqueueNDRangeKernel) -> %d\n", ctx->ret);
        return 0;
    }

    // Read the result back to host memory
    ctx->ret = clEnqueueReadBuffer(ctx->command_queue,
                                       result->mem_obj,
                                       CL_TRUE,
                                       0,
                                       result->total_size * sizeof(float),
                                       result->data,
                                       0,
                                       NULL,
                                       NULL);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clEnqueueReadBuffer) -> %d\n", ctx->ret);
        return 0;
    }

    return 1;
}*/

int matrix_cl_multiply(open_cl_ctx* ctx, Matrix* A, Matrix* B, Matrix* result)
{
    // Copy host data to device buffers
    //ctx->ret = clEnqueueWriteBuffer(ctx->command_queue, A->mem_obj, CL_TRUE, 0, A->total_size * sizeof(float), A->data, 0, NULL, NULL);
    //ctx->ret = clEnqueueWriteBuffer(ctx->command_queue, B->mem_obj, CL_TRUE, 0, B->total_size * sizeof(float), B->data, 0, NULL, NULL);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clEnqueueWriteBuffer) -> %d\n", ctx->ret);
        return 0;
    }

    // Set the arguments of the kernel
    ctx->ret = clSetKernelArg(ctx->kernel_matrix_multiply, 0, sizeof(cl_mem), (void *)&result->mem_obj);
    ctx->ret = clSetKernelArg(ctx->kernel_matrix_multiply, 1, sizeof(cl_mem), (void *)&A->mem_obj);
    ctx->ret = clSetKernelArg(ctx->kernel_matrix_multiply, 2, sizeof(cl_mem), (void *)&B->mem_obj);
    ctx->ret = clSetKernelArg(ctx->kernel_matrix_multiply, 3, sizeof(int), (void *)&A->columns);
    ctx->ret = clSetKernelArg(ctx->kernel_matrix_multiply, 4, sizeof(int), (void *)&A->rows);
    ctx->ret = clSetKernelArg(ctx->kernel_matrix_multiply, 5, sizeof(int), (void *)&B->columns);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clSetKernelArg) -> %d\n", ctx->ret);
        return 0;
    }

    size_t localWorkSize[2], globalWorkSize[2];

   localWorkSize[0] = 16;
   localWorkSize[1] = 16;
   globalWorkSize[0] = B->columns;
   globalWorkSize[1] = A->rows;

    //size_t global_item_size = result->total_size;
    //size_t local_item_size = 64;
    ctx->ret = clEnqueueNDRangeKernel(ctx->command_queue,
                                      ctx->kernel_matrix_multiply,
                                      2,
                                      NULL,
                                      globalWorkSize,
                                      localWorkSize,
                                      0,
                                      NULL,
                                      NULL);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clEnqueueNDRangeKernel) -> %d\n", ctx->ret);
        return 0;
    }

    // Read the result back to host memory
    ctx->ret = clEnqueueReadBuffer(ctx->command_queue,
                                   result->mem_obj,
                                   CL_TRUE,
                                   0,
                                   result->total_size * sizeof(float),
                                   result->data,
                                   0,
                                   NULL,
                                   NULL);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clEnqueueReadBuffer) -> %d\n", ctx->ret);
        return 0;
    }

    return 1;
}

int matrix_cl_multiply_transpose(open_cl_ctx* ctx, Matrix* A, Matrix* B, Matrix* result, int batch_size)
{
    // Copy host data to device buffers
    /*ctx->ret = clEnqueueWriteBuffer(ctx->command_queue, A->mem_obj, CL_TRUE, 0, A->total_size * sizeof(float), A->data, 0, NULL, NULL);
    ctx->ret = clEnqueueWriteBuffer(ctx->command_queue, B->mem_obj, CL_TRUE, 0, B->total_size * sizeof(float), B->data, 0, NULL, NULL);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clEnqueueWriteBuffer) -> %d\n", ctx->ret);
        return 0;
    }*/

    // Set the arguments of the kernel
    ctx->ret = clSetKernelArg(ctx->kernel_matrix_multiply_transpose, 0, sizeof(cl_mem), (void *)&result->mem_obj);
    ctx->ret = clSetKernelArg(ctx->kernel_matrix_multiply_transpose, 1, sizeof(cl_mem), (void *)&A->mem_obj);
    ctx->ret = clSetKernelArg(ctx->kernel_matrix_multiply_transpose, 2, sizeof(cl_mem), (void *)&B->mem_obj);
    ctx->ret = clSetKernelArg(ctx->kernel_matrix_multiply_transpose, 3, sizeof(int), (void *)&A->columns);
    ctx->ret = clSetKernelArg(ctx->kernel_matrix_multiply_transpose, 4, sizeof(int), (void *)&A->rows);
    ctx->ret = clSetKernelArg(ctx->kernel_matrix_multiply_transpose, 5, sizeof(int), (void *)&B->columns);
    ctx->ret = clSetKernelArg(ctx->kernel_matrix_multiply_transpose, 6, sizeof(int), (void *)&batch_size);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clSetKernelArg) -> %d\n", ctx->ret);
        return 0;
    }

    size_t localWorkSize[2], globalWorkSize[2];

    localWorkSize[0] = 16;
    localWorkSize[1] = 16;
    globalWorkSize[0] = result->columns;
    globalWorkSize[1] = result->rows;

    ctx->ret = clEnqueueNDRangeKernel(ctx->command_queue,
                                          ctx->kernel_matrix_multiply_transpose,
                                          2,
                                          NULL,
                                          globalWorkSize,
                                          localWorkSize,
                                          0,
                                          NULL,
                                          NULL);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clEnqueueNDRangeKernel) -> %d\n", ctx->ret);
        return 0;
    }

    // Read the result back to host memory
    ctx->ret = clEnqueueReadBuffer(ctx->command_queue,
                                       result->mem_obj,
                                       CL_TRUE,
                                       0,
                                       result->total_size * sizeof(float),
                                       result->data,
                                       0,
                                       NULL,
                                       NULL);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clEnqueueReadBuffer) -> %d\n", ctx->ret);
        return 0;
    }

    return 1;
}

int matrix_cl_transpose(open_cl_ctx* ctx, Matrix* A, Matrix* result)
{
    // Copy host data to device buffers
    ctx->ret = clEnqueueWriteBuffer(ctx->command_queue, A->mem_obj, CL_TRUE, 0, A->total_size * sizeof(float), A->data, 0, NULL, NULL);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clEnqueueWriteBuffer) -> %d\n", ctx->ret);
        return 0;
    }

    // Set the arguments of the kernel
    ctx->ret = clSetKernelArg(ctx->kernel_matrix_transpose, 0, sizeof(cl_mem), (void *)&A->mem_obj);
    ctx->ret = clSetKernelArg(ctx->kernel_matrix_transpose, 1, sizeof(cl_mem), (void *)&result->mem_obj);
    ctx->ret = clSetKernelArg(ctx->kernel_matrix_transpose, 2, sizeof(int), (void *)&A->columns);
    ctx->ret = clSetKernelArg(ctx->kernel_matrix_transpose, 3, sizeof(int), (void *)&result->columns);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clSetKernelArg) -> %d\n", ctx->ret);
        return 0;
    }

    size_t localWorkSize[2], globalWorkSize[2];

    localWorkSize[0] = 16;
    localWorkSize[1] = 16;
    globalWorkSize[0] = A->columns;
    globalWorkSize[1] = A->rows;

    //size_t global_item_size = result->total_size;
    //size_t local_item_size = 64;
    ctx->ret = clEnqueueNDRangeKernel(ctx->command_queue,
                                      ctx->kernel_matrix_transpose,
                                      2,
                                      NULL,
                                      globalWorkSize,
                                      localWorkSize,
                                      0,
                                      NULL,
                                      NULL);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clEnqueueNDRangeKernel) -> %d\n", ctx->ret);
        return 0;
    }

    // Read the result back to host memory
    ctx->ret = clEnqueueReadBuffer(ctx->command_queue,
                                   result->mem_obj,
                                   CL_TRUE,
                                   0,
                                   result->total_size * sizeof(float),
                                   result->data,
                                   0,
                                   NULL,
                                   NULL);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clEnqueueReadBuffer) -> %d\n", ctx->ret);
        return 0;
    }

    return 1;
}

int matrix_cl_odot(open_cl_ctx* ctx, Matrix* dst, Matrix* src)
{
    // Copy host data to device buffers
    //ctx->ret = clEnqueueWriteBuffer(ctx->command_queue, src->mem_obj, CL_TRUE, 0, src->total_size * sizeof(float), src->data, 0, NULL, NULL);
    //ctx->ret = clEnqueueWriteBuffer(ctx->command_queue, dst->mem_obj, CL_TRUE, 0, dst->total_size * sizeof(float), dst->data, 0, NULL, NULL);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clEnqueueWriteBuffer) -> %d\n", ctx->ret);
        return 0;
    }

    // Set the arguments of the kernel
    ctx->ret = clSetKernelArg(ctx->kernel_vector_multiply, 0, sizeof(cl_mem), (void *)&src->mem_obj);
    ctx->ret = clSetKernelArg(ctx->kernel_vector_multiply, 1, sizeof(cl_mem), (void *)&dst->mem_obj);
    ctx->ret = clSetKernelArg(ctx->kernel_vector_multiply, 2, sizeof(cl_mem), (void *)&dst->mem_obj);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clSetKernelArg) -> %d\n", ctx->ret);
        return 0;
    }

    size_t global_item_size = dst->total_size;
    size_t local_item_size = 64;
    ctx->ret = clEnqueueNDRangeKernel(ctx->command_queue,
                                      ctx->kernel_vector_multiply,
                                      1,
                                      NULL,
                                      &global_item_size,
                                      &local_item_size,
                                      0,
                                      NULL,
                                      NULL);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clEnqueueNDRangeKernel) -> %d\n", ctx->ret);
        return 0;
    }

    // Read the result back to host memory
    ctx->ret = clEnqueueReadBuffer(ctx->command_queue,
                                   dst->mem_obj,
                                   CL_TRUE,
                                   0,
                                   dst->total_size * sizeof(float),
                                   dst->data,
                                   0,
                                   NULL,
                                   NULL);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clEnqueueReadBuffer) -> %d\n", ctx->ret);
        return 0;
    }

    return 1;
}

int matrix_cl_ff(open_cl_ctx* ctx, Matrix* A, Matrix* B, Matrix* add, Matrix* result, int activation)
{
    cl_kernel operating_kernel;

    if (activation == ACTIVATION_LINEAR)
        operating_kernel = ctx->kernel_matrix_multiply_add;
    else if (activation == ACTIVATION_TANH)
        operating_kernel = ctx->kernel_matrix_ff_tanh;
    else if (activation == ACTIVATION_RELU)
        operating_kernel = ctx->kernel_matrix_ff_relu;
    else
    {
        printf("OPEN CL ERROR UNSUPPORTED ACTIVATION: %d\n", activation);
        return 0;
    }

    // Set the arguments of the kernel
    ctx->ret = clSetKernelArg(operating_kernel, 0, sizeof(cl_mem), (void *)&result->mem_obj);
    ctx->ret = clSetKernelArg(operating_kernel, 1, sizeof(cl_mem), (void *)&A->mem_obj);
    ctx->ret = clSetKernelArg(operating_kernel, 2, sizeof(cl_mem), (void *)&B->mem_obj);
    ctx->ret = clSetKernelArg(operating_kernel, 3, sizeof(cl_mem), (void *)&add->mem_obj);
    ctx->ret = clSetKernelArg(operating_kernel, 4, sizeof(int), (void *)&A->columns);
    ctx->ret = clSetKernelArg(operating_kernel, 5, sizeof(int), (void *)&A->rows);
    ctx->ret = clSetKernelArg(operating_kernel, 6, sizeof(int), (void *)&B->columns);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clSetKernelArg) -> %d\n", ctx->ret);
        return 0;
    }

    size_t localWorkSize[2], globalWorkSize[2];

    localWorkSize[0] = 16;
    localWorkSize[1] = 16;
    globalWorkSize[0] = B->columns;
    globalWorkSize[1] = A->rows;

    //size_t global_item_size = result->total_size;
    //size_t local_item_size = 64;
    ctx->ret = clEnqueueNDRangeKernel(ctx->command_queue,
                                      operating_kernel,
                                      2,
                                      NULL,
                                      globalWorkSize,
                                      localWorkSize,
                                      0,
                                      NULL,
                                      NULL);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clEnqueueNDRangeKernel) -> %d\n", ctx->ret);
        return 0;
    }
/*
    ctx->ret = clFinish(ctx->command_queue);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clFinish) -> %d\n", ctx->ret);
        return 0;
    }
*/
    return 1;
}

int matrix_cl_transpose_apply(open_cl_ctx* ctx, Matrix* A, Matrix* result, int anti_activation)
{
    // Copy host data to device buffers
    ctx->ret = clEnqueueWriteBuffer(ctx->command_queue, A->mem_obj, CL_TRUE, 0, A->total_size * sizeof(float), A->data, 0, NULL, NULL);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clEnqueueWriteBuffer) -> %d\n", ctx->ret);
        return 0;
    }

    cl_kernel operating_kernel;

    if (anti_activation == ACTIVATION_LINEAR)
        operating_kernel = ctx->kernel_matrix_transpose_apply_anti_linear;
    else if (anti_activation == ACTIVATION_TANH)
        operating_kernel = ctx->kernel_matrix_transpose_apply_anti_tanh;
    else if (anti_activation == ACTIVATION_RELU)
        operating_kernel = ctx->kernel_matrix_transpose_apply_anti_relu;
    else
    {
        printf("OPEN CL ERROR UNSUPPORTED ANTI ACTIVATION: %d\n", anti_activation);
        return 0;
    }

    // Set the arguments of the kernel
    ctx->ret = clSetKernelArg(operating_kernel, 0, sizeof(cl_mem), (void *)&A->mem_obj);
    ctx->ret = clSetKernelArg(operating_kernel, 1, sizeof(cl_mem), (void *)&result->mem_obj);
    ctx->ret = clSetKernelArg(operating_kernel, 2, sizeof(int), (void *)&A->columns);
    ctx->ret = clSetKernelArg(operating_kernel, 3, sizeof(int), (void *)&result->columns);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clSetKernelArg) -> %d\n", ctx->ret);
        return 0;
    }

    size_t localWorkSize[2], globalWorkSize[2];

    localWorkSize[0] = 16;
    localWorkSize[1] = 16;
    globalWorkSize[0] = A->columns;
    globalWorkSize[1] = A->rows;

    //size_t global_item_size = result->total_size;
    //size_t local_item_size = 64;
    ctx->ret = clEnqueueNDRangeKernel(ctx->command_queue,
                                      operating_kernel,
                                      2,
                                      NULL,
                                      globalWorkSize,
                                      localWorkSize,
                                      0,
                                      NULL,
                                      NULL);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clEnqueueNDRangeKernel) -> %d\n", ctx->ret);
        return 0;
    }

    // Read the result back to host memory
    ctx->ret = clEnqueueReadBuffer(ctx->command_queue,
                                   result->mem_obj,
                                   CL_TRUE,
                                   0,
                                   result->total_size * sizeof(float),
                                   result->data,
                                   0,
                                   NULL,
                                   NULL);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clEnqueueReadBuffer) -> %d\n", ctx->ret);
        return 0;
    }

    return 1;
}

int matrix_cl_sum_rows_transpose(open_cl_ctx* ctx, Matrix* matrix, Matrix* result, int batch_size)
{
    // Copy host data to device buffers
    //ctx->ret = clEnqueueWriteBuffer(ctx->command_queue, matrix->mem_obj, CL_TRUE, 0, matrix->total_size * sizeof(float), matrix->data, 0, NULL, NULL);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clEnqueueWriteBuffer) -> %d\n", ctx->ret);
        return 0;
    }

    // Set the arguments of the kernel
    ctx->ret = clSetKernelArg(ctx->kernel_matrix_sum_rows_transpose_p1, 0, sizeof(cl_mem), (void *)&result->mem_obj);
    ctx->ret = clSetKernelArg(ctx->kernel_matrix_sum_rows_transpose_p1, 1, sizeof(cl_mem), (void *)&matrix->mem_obj);
    ctx->ret = clSetKernelArg(ctx->kernel_matrix_sum_rows_transpose_p1, 2, sizeof(int), (void *)&result->columns);
    ctx->ret = clSetKernelArg(ctx->kernel_matrix_sum_rows_transpose_p1, 3, sizeof(int), (void *)&matrix->columns);
    ctx->ret = clSetKernelArg(ctx->kernel_matrix_sum_rows_transpose_p1, 4, sizeof(int), (void *)&matrix->rows);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clSetKernelArg) -> %d\n", ctx->ret);
        return 0;
    }

    size_t global_item_size = matrix->columns;
    size_t local_item_size = 64;
    ctx->ret = clEnqueueNDRangeKernel(ctx->command_queue,
                                      ctx->kernel_matrix_sum_rows_transpose_p1,
                                      1,
                                      NULL,
                                      &global_item_size,
                                      &local_item_size,
                                      0,
                                      NULL,
                                      NULL);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clEnqueueNDRangeKernel) -> %d\n", ctx->ret);
        return 0;
    }
/*
    // Read the result back to host memory
    ctx->ret = clEnqueueReadBuffer(ctx->command_queue,
                                   result->mem_obj,
                                   CL_TRUE,
                                   0,
                                   result->total_size * sizeof(float),
                                   result->data,
                                   0,
                                   NULL,
                                   NULL);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clEnqueueReadBuffer) -> %d\n", ctx->ret);
        return 0;
    }
*/
    // Set the arguments of the kernel
    ctx->ret = clSetKernelArg(ctx->kernel_matrix_sum_rows_transpose_p2, 0, sizeof(cl_mem), (void *)&result->mem_obj);
    ctx->ret = clSetKernelArg(ctx->kernel_matrix_sum_rows_transpose_p2, 1, sizeof(int), (void *)&result->columns);
    ctx->ret = clSetKernelArg(ctx->kernel_matrix_sum_rows_transpose_p2, 2, sizeof(int), (void *)&result->rows);
    ctx->ret = clSetKernelArg(ctx->kernel_matrix_sum_rows_transpose_p2, 3, sizeof(int), (void *)&batch_size);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clSetKernelArg) -> %d\n", ctx->ret);
        return 0;
    }

    global_item_size = result->columns;
    ctx->ret = clEnqueueNDRangeKernel(ctx->command_queue,
                                      ctx->kernel_matrix_sum_rows_transpose_p2,
                                      1,
                                      NULL,
                                      &global_item_size,
                                      &local_item_size,
                                      0,
                                      NULL,
                                      NULL);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clEnqueueNDRangeKernel) -> %d\n", ctx->ret);
        return 0;
    }

    // Read the result back to host memory
    ctx->ret = clEnqueueReadBuffer(ctx->command_queue,
                                   result->mem_obj,
                                   CL_TRUE,
                                   0,
                                   result->total_size * sizeof(float),
                                   result->data,
                                   0,
                                   NULL,
                                   NULL);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clEnqueueReadBuffer) -> %d\n", ctx->ret);
        return 0;
    }

    return 1;
}

int cl_adam_optimize(open_cl_ctx* ctx, Matrix* weights, Matrix* mw, Matrix* vw, Matrix* gradient, float beta1t, float beta2t)
{
    // Copy host data to device buffers
    //ctx->ret = clEnqueueWriteBuffer(ctx->command_queue, A->mem_obj, CL_TRUE, 0, A->total_size * sizeof(float), A->data, 0, NULL, NULL);
    //ctx->ret = clEnqueueWriteBuffer(ctx->command_queue, B->mem_obj, CL_TRUE, 0, B->total_size * sizeof(float), B->data, 0, NULL, NULL);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clEnqueueWriteBuffer) -> %d\n", ctx->ret);
        return 0;
    }

    // Set the arguments of the kernel
    ctx->ret = clSetKernelArg(ctx->kernel_adam_optimize, 0, sizeof(cl_mem), (void *)&weights->mem_obj);
    ctx->ret = clSetKernelArg(ctx->kernel_adam_optimize, 1, sizeof(cl_mem), (void *)&mw->mem_obj);
    ctx->ret = clSetKernelArg(ctx->kernel_adam_optimize, 2, sizeof(cl_mem), (void *)&vw->mem_obj);
    ctx->ret = clSetKernelArg(ctx->kernel_adam_optimize, 3, sizeof(cl_mem), (void *)&gradient->mem_obj);
    ctx->ret = clSetKernelArg(ctx->kernel_adam_optimize, 4, sizeof(int), (void *)&weights->columns);
    ctx->ret = clSetKernelArg(ctx->kernel_adam_optimize, 5, sizeof(float), (void *)&beta1t);
    ctx->ret = clSetKernelArg(ctx->kernel_adam_optimize, 6, sizeof(float), (void *)&beta2t);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clSetKernelArg) -> %d\n", ctx->ret);
        return 0;
    }

    size_t localWorkSize[2], globalWorkSize[2];

    localWorkSize[0] = 16;
    localWorkSize[1] = 16;
    globalWorkSize[0] = weights->columns;
    globalWorkSize[1] = weights->rows;

    ctx->ret = clEnqueueNDRangeKernel(ctx->command_queue,
                                      ctx->kernel_adam_optimize,
                                      2,
                                      NULL,
                                      globalWorkSize,
                                      localWorkSize,
                                      0,
                                      NULL,
                                      NULL);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clEnqueueNDRangeKernel) -> %d\n", ctx->ret);
        return 0;
    }

    // Read the result back to host memory
    ctx->ret = clEnqueueReadBuffer(ctx->command_queue,
                                   weights->mem_obj,
                                   CL_TRUE,
                                   0,
                                   weights->total_size * sizeof(float),
                                   weights->data,
                                   0,
                                   NULL,
                                   NULL);

    if (ctx->ret != CL_SUCCESS)
    {
        printf("OPEN CL ERROR (clEnqueueReadBuffer) -> %d\n", ctx->ret);
        return 0;
    }

    return 1;
}

int open_cl_test()
{
    // 1. Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint num_platforms, num_devices;
    cl_int ret;

    ret = clGetPlatformIDs(1, &platform_id, &num_platforms);

    printf("RET: %d -> %d\n", ret, num_platforms);

    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &num_devices);

    printf("RET: %d -> %d\n", ret, num_devices);

    // 2. Create an OpenCL context
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

    printf("RET: %d\n", ret);

    // 3. Create a command queue
    cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device_id, NULL, &ret);

    printf("RET: %d\n", ret);

    cl_ulong localMemSize;

    // Get local memory size
    ret = clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &localMemSize, NULL);
    if (ret != CL_SUCCESS) {
        printf("Error getting local memory size: %d\n", ret);
        return 1;
    }

    printf("Local Memory Size: %lu bytes\n", localMemSize);

    // 4. Create host data
    int N = 10;
    float *A = (float*)malloc(sizeof(float) * N);
    float *B = (float*)malloc(sizeof(float) * N);
    float *C = (float*)malloc(sizeof(float) * N);

    for (int i = 0; i < N; i++) {
        A[i] = (float)i;
        B[i] = (float)(N - i);
    }

    // 5. Create device memory buffers
    cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, N * sizeof(float), NULL, &ret);
    cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, N * sizeof(float), NULL, &ret);
    cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, N * sizeof(float), NULL, &ret);

    // 6. Copy host data to device buffers
    ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0, N * sizeof(float), A, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0, N * sizeof(float), B, 0, NULL, NULL);

    // 7. Load the kernel source code
    FILE *fp;
    char *source_str;
    size_t source_size;

    fp = fopen("/home/john/source/neuralnetsim3/ddpg/src/mlpc/matrix_kernel.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_KERNEL_SIZE);
    source_size = fread(source_str, 1, MAX_KERNEL_SIZE, fp);
    fclose(fp);

    // 8. Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);

    // 9. Build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    // 10. Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "vector_add", &ret);

    // 11. Set the arguments of the kernel
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_mem_obj);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b_mem_obj);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&c_mem_obj);
    ret = clSetKernelArg(kernel, 3, sizeof(int), (void *)&N);

    // 12. Execute the OpenCL kernel
    size_t global_item_size = N;
    size_t local_item_size = 1; // Example: can be optimized
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);

    printf("RET: %d\n", ret);

    // 13. Read the result back to host memory
    ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0, N * sizeof(float), C, 0, NULL, NULL);

    // 14. Print the result
    printf("Result of Vector Addition:\n");
    for (int i = 0; i < N; i++) {
        printf("%f + %f = %f\n", A[i], B[i], C[i]);
    }

    // 15. Clean up
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(a_mem_obj);
    ret = clReleaseMemObject(b_mem_obj);
    ret = clReleaseMemObject(c_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);

    free(A);
    free(B);
    free(C);
    free(source_str);

    return 0;
}

#endif
