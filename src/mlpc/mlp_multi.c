#include <malloc.h>
#include <math.h>
#include "random.h"
#include "mlp.h"
#include "loss.h"

MLP_MULTI *mlp_multi_create(
    int inputSize,
    int outputSize, // Per head
    int depth,
    int *hiddenLayerSizes,
    int hiddenLayerActivation,
    int outputLayerActivation,
    int batchSize,
    int headCnt,
    int headDepth,
    int* headHiddenSize)
{
    MLP_MULTI* mlp = malloc(sizeof(MLP_MULTI));

    mlp->input = mlp_create(inputSize,
                            hiddenLayerSizes[depth-1],
                            depth-1,
                            hiddenLayerSizes,
                            hiddenLayerActivation,
                            hiddenLayerActivation,
                            batchSize);

    mlp->head_cnt           = headCnt;
    mlp->head_input_size    = hiddenLayerSizes[depth-1]/headCnt;
    mlp->head_output_size   = outputSize;

    mlp->head = (MLP**)malloc(sizeof(MLP*) * mlp->head_cnt);

    for (int i = 0; i < mlp->head_cnt; i++)
        mlp->head[i] = mlp_create(mlp->head_input_size,
                                  outputSize,
                                  headDepth,
                                  headHiddenSize,
                                  hiddenLayerActivation,
                                  outputLayerActivation,
                                  batchSize);

    mlp->output = matrix_create(batchSize, outputSize*mlp->head_cnt);

    matrix_clear(mlp->output);

    return mlp;
}

MLP_MULTI* mlp_multi_clone(MLP_MULTI* mlp)
{
    MLP_MULTI* clone = malloc(sizeof(MLP_MULTI));

    clone->head_cnt         = mlp->head_cnt;
    clone->head_input_size  = mlp->head_input_size;
    clone->head_output_size = mlp->head_output_size;

    clone->input = mlp_clone(mlp->input);

    clone->head = (MLP**)malloc(sizeof(MLP*) * clone->head_cnt);

    for (int i = 0; i < mlp->head_cnt; i++)
        clone->head[i] = mlp_clone(mlp->head[i]);

    clone->output = matrix_clone(mlp->output);

    return clone;
}

void mlp_multi_copy(MLP_MULTI* dst, MLP_MULTI* src)
{
    dst->head_cnt         = src->head_cnt;
    dst->head_input_size  = src->head_input_size;
    dst->head_output_size = src->head_output_size;

    matrix_copy(dst->output, src->output);

    for (int i = 0; i < src->head_cnt; i++)
        mlp_copy(dst->head[i], src->head[i]);

    mlp_copy(dst->input, src->input);
}

void mlp_multi_destroy(MLP_MULTI *mlp)
{
    mlp_destroy(mlp->input);

    for (int i = 0; i < mlp->head_cnt; i++)
        mlp_destroy(mlp->head[i]);

    free(mlp->head);

    matrix_destroy(mlp->output);

    free(mlp);
}

Matrix mlp_multi_feedforward(MLP_MULTI *mlp, Matrix x)
{
    Matrix x_int = mlp_feedforward(mlp->input, x);

    Matrix x_int_sub = matrix_create(mlp->input->batchSize, mlp->head_input_size);

    for (int i = 0; i < mlp->head_cnt; i++)
    {

        for (int batch = 0; batch < mlp->head[i]->batchSize; batch++)   
            for (int out_index = 0; out_index < mlp->head_input_size; out_index++)   
                MATRIX(x_int_sub, batch, out_index) = MATRIX(x_int, batch, i*mlp->head_input_size + out_index);

        Matrix out = mlp_feedforward(mlp->head[i], x_int_sub);

        for (int batch = 0; batch < mlp->head[i]->batchSize; batch++)   
            for (int out_index = 0; out_index < mlp->head_output_size; out_index++)   
                MATRIX(mlp->output, batch, i*mlp->head_output_size + out_index) = MATRIX(out, batch, out_index);
    }

    matrix_destroy(x_int_sub);

    return mlp->output;
}

double mlp_multi_backpropagate(MLP_MULTI *mlp, Matrix y, int lossFunctionCode)
{
    double loss  = 0;
    Matrix y_sub = matrix_create(mlp->input->batchSize, mlp->head[0]->output.columns); // head_output_size
    Matrix y_int = matrix_create(mlp->input->batchSize, mlp->input->output.columns);
//printf("%d %d %d\n", mlp->head[0]->output.columns, mlp->head_output_size, mlp->input->output.columns);
    for (int i = 0; i < mlp->head_cnt; i++)
    {
        for (int batch = 0; batch < mlp->head[i]->batchSize; batch++)   
            for (int out_index = 0; out_index < mlp->head_output_size; out_index++)
                MATRIX(y_sub, batch, out_index) = MATRIX(y, batch, i*mlp->head_output_size + out_index);

        loss += mlp_backpropagate(mlp->head[i], y_sub, lossFunctionCode);

        Matrix errors = mlp_get_input_errors(mlp->head[i]);
//printf("%d %d\n", errors.columns, y_int.columns);
        // Copy errors to intermediate Y
        for (int batch = 0; batch < mlp->head[i]->batchSize; batch++)   
            for (int out_index = 0; out_index < errors.columns; out_index++)
                MATRIX(y_int, batch, i*errors.columns + out_index) = MATRIX(errors, batch, out_index);
    }

    loss += mlp_backpropagate(mlp->input, y_int, lossFunctionCode);

    matrix_destroy(y_sub);
    matrix_destroy(y_int);

    return loss;
}
