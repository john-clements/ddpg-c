#include <malloc.h>
#include <math.h>
#include <string.h>
#include "random.h"
#include "mlp.h"
#include "loss.h"

/* Initialize the MLPC library. */
void mlp_init()
{
#ifdef OPEN_CL_EN
    //open_cl_init();
#endif
    deepc_random_init();
}

/* A helper function to create a layer. */
Layer mlp_create_layer(int inputSize, int outputSize, int batchSize, int activation)
{
    Layer layer;
    layer.weights = matrix_create(outputSize, inputSize);
    layer.biases = matrix_create(outputSize, batchSize);
    layer.output = matrix_create(outputSize, batchSize);
    layer.errors = matrix_create(batchSize, outputSize);
    layer.deltas = matrix_create(batchSize, outputSize);
    layer.gradWeights = matrix_create(outputSize, inputSize);
    layer.gradBiases = matrix_create(outputSize, batchSize);
    layer.activation = getActivationFunction(activation);
    layer.activationDeriv = getActivationFunctionDeriv(activation);
    layer.activation_id = activation;
    
    return layer;
}

#ifdef OPEN_CL_EN
void mlp_create_opencl(MLP *mlp)
{
    open_cl_init(&mlp->ocl);

    matrix_cl_create(&mlp->ocl, &mlp->input);

    matrix_cl_create(&mlp->ocl, &mlp->inputErrors);
    matrix_cl_create(&mlp->ocl, &mlp->output);

    for (int i = 0; i <= mlp->depth; i++)
    {
        matrix_cl_create(&mlp->ocl, &mlp->layers[i].weights);
        matrix_cl_create(&mlp->ocl, &mlp->layers[i].biases);
        matrix_cl_create(&mlp->ocl, &mlp->layers[i].output);
        matrix_cl_create(&mlp->ocl, &mlp->layers[i].errors);
        matrix_cl_create(&mlp->ocl, &mlp->layers[i].deltas);
        matrix_cl_create(&mlp->ocl, &mlp->layers[i].gradWeights);
        matrix_cl_create(&mlp->ocl, &mlp->layers[i].gradBiases);
    }
}
#endif

/* Creates a new neural network on heap. */
MLP *mlp_create(int inputSize, int outputSize, int depth, int *hiddenLayerSizes, int hiddenLayerActivation, int outputLayerActivation, int batchSize)
{
    MLP *mlp = malloc(sizeof(MLP));

    memset((void*)mlp, 0, sizeof(MLP));

    mlp->depth = depth;
    mlp->batchSize = batchSize;
    mlp->layers = malloc((depth + 1) * sizeof(Layer));
    
    int layerInputSize = inputSize;
    for (int i = 0; i < depth; i++)
    {
        mlp->layers[i] = mlp_create_layer(layerInputSize, hiddenLayerSizes[i], batchSize, hiddenLayerActivation);
        layerInputSize = hiddenLayerSizes[i];
    }
    mlp->layers[depth] = mlp_create_layer(layerInputSize, outputSize, batchSize, outputLayerActivation);

    mlp->input = matrix_create(inputSize, batchSize);
    mlp->inputErrors = matrix_create(batchSize, inputSize);
    mlp->output = matrix_create(batchSize, outputSize);

    mlp_initialize(mlp);

#ifdef OPEN_CL_EN
    mlp_create_opencl(mlp);
#endif

    return mlp;
}

/* Creates a new neural network on heap that is a clone of the given neural network. */
MLP *mlp_clone(MLP *mlp)
{
    MLP *clone = malloc(sizeof(MLP));

    memset((void*)clone, 0, sizeof(MLP));

    clone->depth = mlp->depth;
    clone->batchSize = mlp->batchSize;
    clone->layers = malloc((mlp->depth + 1) * sizeof(Layer));

    for (int i = 0; i <= mlp->depth; i++)
    {
        clone->layers[i].weights = matrix_clone(mlp->layers[i].weights);
        clone->layers[i].biases = matrix_clone(mlp->layers[i].biases);
        clone->layers[i].output = matrix_clone(mlp->layers[i].output);
        clone->layers[i].errors = matrix_clone(mlp->layers[i].errors);
        clone->layers[i].deltas = matrix_clone(mlp->layers[i].deltas);
        clone->layers[i].gradWeights = matrix_clone(mlp->layers[i].gradWeights);
        clone->layers[i].gradBiases = matrix_clone(mlp->layers[i].gradBiases);
        clone->layers[i].activation = mlp->layers[i].activation;
        clone->layers[i].activationDeriv = mlp->layers[i].activationDeriv;
        clone->layers[i].activation_id = mlp->layers[i].activation_id;
    }

    clone->input = matrix_clone(mlp->input);
    clone->inputErrors = matrix_clone(mlp->inputErrors);
    clone->output = matrix_clone(mlp->output);

#ifdef OPEN_CL_EN
    mlp_create_opencl(clone);
#endif

    return clone;
}

/* Destroys a neural network created with mlp_create or mlp_clone. */
void mlp_destroy(MLP *mlp)
{
    for (int i = 0; i <= mlp->depth; i++)
    {
#ifdef OPEN_CL_EN
        matrix_cl_dystroy(&mlp->ocl, &mlp->layers[i].weights);
        matrix_cl_dystroy(&mlp->ocl, &mlp->layers[i].biases);
        matrix_cl_dystroy(&mlp->ocl, &mlp->layers[i].output);
        matrix_cl_dystroy(&mlp->ocl, &mlp->layers[i].errors);
        matrix_cl_dystroy(&mlp->ocl, &mlp->layers[i].deltas);
        matrix_cl_dystroy(&mlp->ocl, &mlp->layers[i].gradWeights);
        matrix_cl_dystroy(&mlp->ocl, &mlp->layers[i].gradBiases);
#endif
        matrix_destroy(mlp->layers[i].weights);
        matrix_destroy(mlp->layers[i].biases);
        matrix_destroy(mlp->layers[i].output);
        matrix_destroy(mlp->layers[i].errors);
        matrix_destroy(mlp->layers[i].deltas);
        matrix_destroy(mlp->layers[i].gradWeights);
        matrix_destroy(mlp->layers[i].gradBiases);
    }
#ifdef OPEN_CL_EN
    matrix_cl_dystroy(&mlp->ocl, &mlp->input);
    matrix_cl_dystroy(&mlp->ocl, &mlp->inputErrors);
    matrix_cl_dystroy(&mlp->ocl, &mlp->output);
#endif
    matrix_destroy(mlp->input);
    matrix_destroy(mlp->inputErrors);
    matrix_destroy(mlp->output);

    free(mlp->layers);
    free(mlp);
}

/* Clears all existing values and sets random weights using the Glorot method. */
void mlp_initialize(MLP *mlp)
{
    for (int i = 0; i <= mlp->depth; i++)
    {        
        float limit = sqrt(6.0 / (float)(mlp->layers[i].weights.rows + mlp->layers[i].weights.columns));
        float *data = mlp->layers[i].weights.data;
        for (int k = 0; k < mlp->layers[i].weights.rows * mlp->layers[i].weights.columns; k++)
            data[k] = deepc_random_float(-limit, limit);
        
        matrix_clear(mlp->layers[i].biases);
        matrix_clear(mlp->layers[i].output);
        matrix_clear(mlp->layers[i].errors);
        matrix_clear(mlp->layers[i].deltas);
        matrix_clear(mlp->layers[i].gradWeights);
        matrix_clear(mlp->layers[i].gradBiases);
    }

    matrix_clear(mlp->input);
    matrix_clear(mlp->inputErrors);
    matrix_clear(mlp->output);
}

/*
   Copies all content from the src to the dst neural network but doesn't change
   its architecture. The caller must ensure identical architectures.
*/
void mlp_copy(MLP *dst, MLP *src)
{
    for (int i = 0; i <= src->depth; i++)
    {
        matrix_copy(dst->layers[i].weights, src->layers[i].weights);
        matrix_copy(dst->layers[i].biases, src->layers[i].biases);
        matrix_copy(dst->layers[i].output, src->layers[i].output);
        matrix_copy(dst->layers[i].errors, src->layers[i].errors);
        matrix_copy(dst->layers[i].deltas, src->layers[i].deltas);
        matrix_copy(dst->layers[i].gradWeights, src->layers[i].gradWeights);
        matrix_copy(dst->layers[i].gradBiases, src->layers[i].gradBiases);
    }
    
    matrix_copy(dst->input, src->input);
    matrix_copy(dst->inputErrors, src->inputErrors);
    matrix_copy(dst->output, src->output);
}

void mlp_soft_copy(MLP *dst, MLP *src, float tau)
{
    for (int i = 0; i <= src->depth; i++)
    {
        matrix_soft_copy(dst->layers[i].weights, src->layers[i].weights, tau);
        matrix_soft_copy(dst->layers[i].biases, src->layers[i].biases, tau);
        matrix_soft_copy(dst->layers[i].output, src->layers[i].output, tau);
        matrix_soft_copy(dst->layers[i].errors, src->layers[i].errors, tau);
        matrix_soft_copy(dst->layers[i].deltas, src->layers[i].deltas, tau);
        matrix_soft_copy(dst->layers[i].gradWeights, src->layers[i].gradWeights, tau);
        matrix_soft_copy(dst->layers[i].gradBiases, src->layers[i].gradBiases, tau);
    }
    
    matrix_soft_copy(dst->input, src->input, tau);
    matrix_soft_copy(dst->inputErrors, src->inputErrors, tau);
    matrix_soft_copy(dst->output, src->output, tau);
}

/*
   Performs a feedforward operation with the given input values x and returns the
   output values. All the intermediate layer outputs as well as the final output
   are stored internally to be used during back-propagation.
*/
/*
Matrix mlp_feedforward(MLP *mlp, Matrix x)
{
    matrix_transpose(x, mlp->input);
    Matrix *input = &mlp->input;

    for (int i = 0; i <= mlp->depth; i++)
    {
#ifdef OPEN_CL_EN
        matrix_cl_multiply_add(&mlp->layers[i].weights,
                               input,
                               &mlp->layers[i].biases,
                               &mlp->layers[i].output);
#else
        matrix_dot(mlp->layers[i].weights, *input, mlp->layers[i].output);
        matrix_add(mlp->layers[i].output, mlp->layers[i].biases);
#endif

        matrix_apply(mlp->layers[i].output, mlp->layers[i].activation);
        input = &mlp->layers[i].output;
    }
    matrix_transpose(*input, mlp->output);
    return mlp->output;
}
*/

Matrix mlp_feedforward(MLP *mlp, Matrix x)
{
    matrix_transpose(x, mlp->input);
    Matrix *input = &mlp->input;

#ifdef OPEN_CL_EN
    matrix_cl_copy_to_device(&mlp->ocl, input);
#endif

    for (int i = 0; i <= mlp->depth; i++)
    {
#ifdef OPEN_CL_EN
        matrix_cl_copy_to_device(&mlp->ocl, &mlp->layers[i].weights);
        matrix_cl_copy_to_device(&mlp->ocl, &mlp->layers[i].biases);

        matrix_cl_ff(&mlp->ocl,
                     &mlp->layers[i].weights,
                     input,
                     &mlp->layers[i].biases,
                     &mlp->layers[i].output,
                     mlp->layers[i].activation_id);

        matrix_cl_copy_to_host(&mlp->ocl, &mlp->layers[i].output); // TODO: Figure out why this is necessary
#else
        matrix_dot(mlp->layers[i].weights, *input, mlp->layers[i].output);
        matrix_add(mlp->layers[i].output, mlp->layers[i].biases);
        matrix_apply(mlp->layers[i].output, mlp->layers[i].activation);
#endif
        input = &mlp->layers[i].output;
    }

#ifdef OPEN_CL_EN
    //matrix_cl_copy_to_host(&mlp->ocl, input);
#endif

    matrix_transpose(*input, mlp->output);
    return mlp->output;
}

/*
   Backpropagates the error according to the given true values y and the given
   loss function. The resulting gradients are stored internally. The total error
   over all samples is returned.
*/
float mlp_backpropagate(MLP *mlp, Matrix y, int lossFunctionCode)
{   
    /* Use the loss function to compute the error values. */
    LossFunction lossFunction = getLossFunction(lossFunctionCode);
    float loss = lossFunction(mlp->output, y, mlp->layers[mlp->depth].errors);

    /* Compute the deltas of the last layer. */
    matrix_copy(mlp->layers[mlp->depth].deltas, mlp->output);
    matrix_apply(mlp->layers[mlp->depth].deltas, mlp->layers[mlp->depth].activationDeriv);
    
    /* Compute the error of the previous layer. */
    if (mlp->depth > 0)
    {
#ifdef OPEN_CL_EN
        matrix_cl_copy_to_device(&mlp->ocl, &mlp->layers[mlp->depth].deltas);
        matrix_cl_copy_to_device(&mlp->ocl, &mlp->layers[mlp->depth].errors);

        matrix_cl_odot(&mlp->ocl, &mlp->layers[mlp->depth].deltas, &mlp->layers[mlp->depth].errors);
#else
        matrix_odot(mlp->layers[mlp->depth].deltas, mlp->layers[mlp->depth].errors);
#endif
    }

    /* Propagate the deltas towards the first layer. */
    for (int i = mlp->depth - 1; i >= 0; i--)
    {
#ifdef OPEN_CL_EN
        matrix_cl_copy_to_device(&mlp->ocl, &mlp->layers[i].deltas);

        matrix_cl_multiply(&mlp->ocl, &mlp->layers[i+1].deltas, &mlp->layers[i+1].weights, &mlp->layers[i].errors);
        //matrix_cl_transpose(&mlp->ocl, &mlp->layers[i].output, &mlp->layers[i].deltas);
        //matrix_apply(mlp->layers[i].deltas, mlp->layers[i].activationDeriv);
        matrix_cl_transpose_apply(&mlp->ocl, &mlp->layers[i].output, &mlp->layers[i].deltas, mlp->layers[i].activation_id);

        matrix_cl_odot(&mlp->ocl, &mlp->layers[i].deltas, &mlp->layers[i].errors);
#else
        matrix_dot(mlp->layers[i+1].deltas, mlp->layers[i+1].weights, mlp->layers[i].errors);
        matrix_transpose(mlp->layers[i].output, mlp->layers[i].deltas);
        matrix_apply(mlp->layers[i].deltas, mlp->layers[i].activationDeriv);
        matrix_odot(mlp->layers[i].deltas, mlp->layers[i].errors);
#endif
    }

    /* Compute the input errors. */
#ifdef OPEN_CL_EN
    matrix_cl_copy_to_device(&mlp->ocl, &mlp->layers[0].weights);
    matrix_cl_multiply(&mlp->ocl, &mlp->layers[0].deltas, &mlp->layers[0].weights, &mlp->inputErrors);
#else
    matrix_dot(mlp->layers[0].deltas, mlp->layers[0].weights, mlp->inputErrors);
#endif

    /* Compute the gradients. */
    Matrix *input = &mlp->input;
#ifdef OPEN_CL_EN
    matrix_cl_copy_to_device(&mlp->ocl, input);
#endif

    for (int i = 0; i <= mlp->depth; i++)
    {
#ifdef OPEN_CL_EN
        matrix_cl_multiply_transpose(&mlp->ocl, input, &mlp->layers[i].deltas, &mlp->layers[i].gradWeights, mlp->batchSize);
        matrix_cl_sum_rows_transpose(&mlp->ocl, &mlp->layers[i].deltas, &mlp->layers[i].gradBiases, mlp->batchSize);
#else
        matrix_dot_transpose(*input, mlp->layers[i].deltas, mlp->layers[i].gradWeights);
        matrix_divide(mlp->layers[i].gradWeights, (float)mlp->batchSize);
        matrix_sum_rows_transpose(mlp->layers[i].deltas, mlp->layers[i].gradBiases);
        matrix_divide(mlp->layers[i].gradBiases, (float)mlp->batchSize);
#endif

        //matrix_sum_rows_transpose(mlp->layers[i].deltas, mlp->layers[i].gradBiases);
        //matrix_divide(mlp->layers[i].gradBiases, (float)mlp->batchSize);

        input = &mlp->layers[i].output;
    }

    return loss;
}

/*
    Returns the error at the inout layer. This can used to interconnect
    the back-propagation process across different neural networks.
*/
Matrix mlp_get_input_errors(MLP *mlp)
{
    return mlp->inputErrors;
}

/* 
   Performs stohastic gradient descent. The function can be called after the gradients
   have been computed through back-propagation.
*/
void mlp_sgd(MLP *mlp, float lr)
{
    for (int i = 0; i <= mlp->depth; i++)
    {
        matrix_multiply(mlp->layers[i].gradWeights, lr);
        matrix_subtract(mlp->layers[i].weights, mlp->layers[i].gradWeights);

        matrix_multiply(mlp->layers[i].gradBiases, lr);
        matrix_subtract(mlp->layers[i].biases, mlp->layers[i].gradBiases);
    }
}

void mlp_clip_gradients(Matrix gradients, float clipnorm)
{
    float norm = 0;
    for (int i = 0; i < gradients.rows * gradients.columns; i++)
        norm += pow(gradients.data[i], 2);
    norm = sqrt(norm);
    
    if (norm > clipnorm)
        matrix_multiply(gradients, clipnorm / norm);
}

void mlp_sgd_clip(MLP *mlp, float lr, float clipnorm)
{
    for (int i = 0; i <= mlp->depth; i++)
    {
        mlp_clip_gradients(mlp->layers[i].gradWeights, clipnorm);
        matrix_multiply(mlp->layers[i].gradWeights, lr);
        matrix_subtract(mlp->layers[i].weights, mlp->layers[i].gradWeights);

        matrix_multiply(mlp->layers[i].gradBiases, lr);
        matrix_subtract(mlp->layers[i].biases, mlp->layers[i].gradBiases);
    }
}

int mlp_load_weights(MLP *mlp, const char *filename)
{
    FILE *file = fopen(filename, "rb");
    if (file == NULL)
        return -1;
    
    int result = mlp_read_weights(mlp, file);
    fclose(file);
    
    return result;
}

int mlp_read_weights(MLP *mlp, FILE *file)
{
    int rows, columns;
    Matrix matrix;

    for (int i = 0; i <= mlp->depth; i++)
    {
        rows = mlp->layers[i].weights.rows;
        columns = mlp->layers[i].weights.columns;
        
        matrix = matrix_read(file);
        if (matrix.rows != rows || matrix.columns != columns || matrix.data == NULL)
            return -1;
        
        matrix_copy(mlp->layers[i].weights, matrix);
        matrix_destroy(matrix);

        rows = mlp->layers[i].biases.rows;
        columns = mlp->layers[i].biases.columns;

        matrix = matrix_read(file);
        if (matrix.rows != rows || matrix.columns != columns || matrix.data == NULL)
            return -1;
        
        matrix_copy(mlp->layers[i].biases, matrix);
        matrix_destroy(matrix);
    }

    return 0;
}

int mlp_save_weights(MLP *mlp, const char *filename)
{
    FILE *file = fopen(filename, "wb");
    if (file == NULL)
        return -1;
    
    int result = mlp_write_weights(mlp, file);
    fclose(file);
    
    return result;
}

int mlp_write_weights(MLP *mlp, FILE *file)
{
    for (int i = 0; i <= mlp->depth; i++)
    {
        if (matrix_write(mlp->layers[i].weights, file) != 0)
            return -1;

        if (matrix_write(mlp->layers[i].biases, file) != 0)
            return -1;
    }

    return 0;
}
