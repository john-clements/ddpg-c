#include <malloc.h>
#include "matrix.h"
#include "random.h"

Matrix matrix_create(int rows, int columns)
{
    Matrix matrix;
    matrix.rows = rows;
    matrix.columns = columns;
    matrix.data = malloc(rows * columns * sizeof(float));

//#ifdef OPEN_CL_EN
//    matrix_cl_create(&matrix);
//#endif
    return matrix;
}

Matrix matrix_clone(Matrix matrix)
{
    Matrix clone;
    clone.rows = matrix.rows;
    clone.columns = matrix.columns;
    clone.data = malloc(matrix.rows * matrix.columns * sizeof(float));

    for (int i = 0; i < matrix.rows * matrix.columns; i++)
        clone.data[i] = matrix.data[i];
    
//#ifdef OPEN_CL_EN
//    matrix_cl_create(&clone);
//#endif

    return clone;
}

Matrix matrix_load(const char *filename)
{
    Matrix matrix;
    matrix.rows = 0;
    matrix.columns = 0;
    matrix.data = NULL;

    FILE *file = fopen(filename, "rb");
    if (file == NULL)
        return matrix;
    
    matrix = matrix_read(file);
    fclose(file);

//#ifdef OPEN_CL_EN
//    matrix_cl_create(&matrix);
//#endif

    return matrix;
}

Matrix matrix_read(FILE *file)
{
    Matrix matrix;
    matrix.rows = 0;
    matrix.columns = 0;
    matrix.data = NULL;

    int columns, rows, n;
    float *data;

    if (fread(&rows, sizeof(int), 1, file) != 1)
        return matrix;
    
    if (fread(&columns, sizeof(int), 1, file) != 1)
        return matrix;
    
    if ((n = rows * columns) <= 0)
        return matrix;
    
    if ((data = malloc(n * sizeof(float))) == NULL)
        return matrix;
    
    if (fread(data, sizeof(float), n, file) != n)
    {
        free(data);
        return matrix;
    }

    matrix.rows = rows;
    matrix.columns = columns;
    matrix.data = data;

//#ifdef OPEN_CL_EN
//    matrix_cl_create(&matrix);
//#endif

    return matrix;
}

int matrix_save(Matrix matrix, const char *filename)
{
    FILE *file = fopen(filename, "wb");
    if (file == NULL)
        return -1;
    
    int result = matrix_write(matrix, file);
    fclose(file);
    
    return result;
}

int matrix_write(Matrix matrix, FILE *file)
{
    if (fwrite(&matrix.rows, sizeof(int), 1, file) != 1)
        return -1;
    
    if (fwrite(&matrix.columns, sizeof(int), 1, file) != 1)
        return -1;
    
    int n = matrix.rows * matrix.columns;
    if (fwrite(matrix.data, sizeof(float), n, file) != n)
        return -1;

    return 0;
}

void matrix_destroy(Matrix matrix)
{
    if (matrix.data != NULL)
        free(matrix.data);

//#ifdef OPEN_CL_EN
//    matrix_cl_dystroy(&matrix);
//#endif
}

void matrix_clear(Matrix matrix)
{
    for (int i = 0; i < matrix.rows * matrix.columns; i++)
        matrix.data[i] = 0;
}

void matrix_copy(Matrix dst, Matrix src)
{
    for (int i = 0; i < dst.rows * dst.columns; i++)
        dst.data[i] = src.data[i];
}

void matrix_soft_copy(Matrix dst, Matrix src, float tau)
{
    for (int i = 0; i < dst.rows * dst.columns; i++)
        dst.data[i] = tau*src.data[i] + (1.0f - tau)*dst.data[i];
}

void matrix_fill(Matrix matrix, float value)
{
    for (int i = 0; i < matrix.rows * matrix.columns; i++)
        matrix.data[i] = value;
}

void matrix_randomize(Matrix matrix, float min, float max)
{
    for (int i = 0; i < matrix.rows * matrix.columns; i++)
        matrix.data[i] = deepc_random_float(min, max);
}

void matrix_sum(Matrix matrix1, Matrix matrix2, Matrix result)
{
    for (int i = 0; i < result.rows * result.columns; i++)
        result.data[i] = matrix1.data[i] + matrix2.data[i];
}

void matrix_add(Matrix dst, Matrix src)
{
//#ifdef OPEN_CL_EN
//    matrix_cl_add(&dst, &src);
//#else
    for (int i = 0; i < dst.rows * dst.columns; i++)
        dst.data[i] += src.data[i];
//#endif
}

void matrix_difference(Matrix matrix1, Matrix matrix2, Matrix result)
{
    for (int i = 0; i < result.rows * result.columns; i++)
        result.data[i] = matrix1.data[i] - matrix2.data[i];
}

void matrix_subtract(Matrix dst, Matrix src)
{
    for (int i = 0; i < dst.rows * dst.columns; i++)
        dst.data[i] -= src.data[i];
}

void matrix_multiply(Matrix matrix, float value)
{
    for (int i = 0; i < matrix.rows * matrix.columns; i++)
        matrix.data[i] *= value;
}

void matrix_divide(Matrix matrix, float value)
{
    for (int i = 0; i < matrix.rows * matrix.columns; i++)
        matrix.data[i] /= value;
}

void matrix_odot(Matrix dst, Matrix src)
{
//#ifdef OPEN_CL_EN
//    matrix_cl_odot(&dst, &src);
//#else
    for (int i = 0; i < dst.rows * dst.columns; i++)
        dst.data[i] *= src.data[i];
//#endif
}

void matrix_dot(Matrix matrix1, Matrix matrix2, Matrix result)
{
//#ifdef OPEN_CL_EN
//    matrix_cl_multiply(&matrix1, &matrix2, &result);
//#else
    float *p = result.data;
    for (int row = 0; row < result.rows; row++)
    {
        for (int col = 0; col < result.columns; col++)
        {
            float *p1 = matrix1.data + matrix1.columns * row;
            float *p2 = matrix2.data + col;
            float sum = 0;
            for (int k = 0; k < matrix1.columns; k++)
            {
                sum += *p1 * *p2;
                p1 += 1;
                p2 += matrix2.columns;
            }
            *(p++) = sum;
        }
    }
//#endif
}

void matrix_transpose(Matrix matrix, Matrix result)
{
     for (int row = 0; row < matrix.rows; row++)
        for (int col = 0; col < matrix.columns; col++)
            result.data[col * result.columns + row] = matrix.data[row * matrix.columns + col];
}

void matrix_dot_transpose(Matrix matrix1, Matrix matrix2, Matrix result)
{
    for (int col = 0; col < result.columns; col++)
    {
        for (int row = 0; row < result.rows; row++)
        {
            float *p1 = matrix1.data + matrix1.columns * col;
            float *p2 = matrix2.data + row;
            float sum = 0;
            for (int k = 0; k < matrix1.columns; k++)
            {
                sum += *p1 * *p2;
                p1 += 1;
                p2 += matrix2.columns;
            }
            result.data[row * result.columns + col] = sum;
        }
    }
}

void matrix_sum_rows_transpose(Matrix matrix, Matrix result)
{
    for (int col = 0; col < matrix.columns; col++)
    {
        int idx = col * result.columns;
        result.data[idx] = 0;
        for (int row = 0; row < matrix.rows; row++)
            result.data[idx] += matrix.data[row * matrix.columns + col];
    }

    for (int col = 1; col < result.columns; col++)
    {
        for (int row = 0; row < result.rows; row++)
            result.data[row * result.columns + col] = result.data[row * result.columns];
    }
}

void matrix_apply(Matrix matrix, ActivationFunction activationFunction)
{
    for (int i = 0; i < matrix.rows * matrix.columns; i++)
        matrix.data[i] = activationFunction(matrix.data[i]);
}
