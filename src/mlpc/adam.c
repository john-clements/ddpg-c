#include <malloc.h>
#include <math.h>
#include "adam.h"

Adam *adam_create(MLP *mlp)
{
    Adam *adam = malloc(sizeof(Adam));
    adam->alpha = 0.001;
    adam->beta1 = 0.9;
    adam->beta2 = 0.999;
    adam->epsilon = 1e-7;
    adam->depth = mlp->depth + 1;

    adam->mw = malloc(adam->depth * sizeof(Matrix));
    adam->mb = malloc(adam->depth * sizeof(Matrix));
    adam->vw = malloc(adam->depth * sizeof(Matrix));
    adam->vb = malloc(adam->depth * sizeof(Matrix));

    for (int i = 0; i < adam->depth; i++)
    {
        adam->mw[i] = matrix_clone(mlp->layers[i].weights);
        adam->mb[i] = matrix_clone(mlp->layers[i].biases);
        adam->vw[i] = matrix_clone(mlp->layers[i].weights);
        adam->vb[i] = matrix_clone(mlp->layers[i].biases);

#ifdef OPEN_CL_EN
        adam->ocl = &mlp->ocl;

        matrix_cl_create(adam->ocl, &adam->mw[i]);
        matrix_cl_create(adam->ocl, &adam->mb[i]);
        matrix_cl_create(adam->ocl, &adam->vw[i]);
        matrix_cl_create(adam->ocl, &adam->vb[i]);
#endif
    }

    adam_reset(adam);

    return adam;
}

/* ---- loading and storing Adam is probably not needed. ----

Adam *adam_read(FILE *file)
{
    float parameters[6];
    int depth;

    if (fread(parameters, sizeof(float), 6, file) != 6)
        return NULL;
    
    if (fread(&depth, sizeof(int), 1, file) != 1)
        return NULL;
    
    Adam *adam = malloc(sizeof(Adam));

    adam->alpha = parameters[0];
    adam->beta1 = parameters[1];
    adam->beta2 = parameters[2];
    adam->beta1t = parameters[3];
    adam->beta2t = parameters[4];
    adam->epsilon = parameters[5];
    adam->depth = depth;

    adam->mw = malloc(depth * sizeof(Matrix));
    adam->mb = malloc(depth * sizeof(Matrix));
    adam->vw = malloc(depth * sizeof(Matrix));
    adam->vb = malloc(depth * sizeof(Matrix));

    for (int i = 0; i < depth; i++)
    {
        adam->mw[i].data = NULL;
        adam->mb[i].data = NULL;
        adam->vw[i].data = NULL;
        adam->vb[i].data = NULL;
    }

    int error = 0;
    int i = 0;

    while (i < depth && !error)
    {
        adam->mw[i] = matrix_read(file);
        if (adam->mw[i].data == NULL)
            error = 1;
        
        adam->mb[i] = matrix_read(file);
        if (adam->mb[i].data == NULL)
            error = 1;

        adam->vw[i] = matrix_read(file);
        if (adam->vw[i].data == NULL)
            error = 1;
        
        adam->vb[i] = matrix_read(file);
        if (adam->vb[i].data == NULL)
            error = 1;

        i++;
    }

    if (error)
    {
        adam_destroy(adam);
        return NULL;
    }

    return adam;
}

int adam_write(Adam *adam, FILE *file)
{
    if (fwrite(&adam->alpha, sizeof(float), 1, file) != 1)
        return -1;
    
    if (fwrite(&adam->beta1, sizeof(float), 1, file) != 1)
        return -1;
    
    if (fwrite(&adam->beta2, sizeof(float), 1, file) != 1)
        return -1;
    
    if (fwrite(&adam->beta1t, sizeof(float), 1, file) != 1)
        return -1;
    
    if (fwrite(&adam->beta2t, sizeof(float), 1, file) != 1)
        return -1;
    
    if (fwrite(&adam->epsilon, sizeof(float), 1, file) != 1)
        return -1;
    
    if (fwrite(&adam->depth, sizeof(int), 1, file) != 1)
        return -1;
    
    for (int i = 0; i < adam->depth; i++)
    {
        if (matrix_write(adam->mw[i], file) != 0)
            return -1;
        if (matrix_write(adam->mb[i], file) != 0)
            return -1;
        if (matrix_write(adam->vw[i], file) != 0)
            return -1;
        if (matrix_write(adam->vb[i], file) != 0)
            return -1;
    }

    return 0;
}
*/

void adam_destroy(Adam *adam)
{
    for (int i = 0; i < adam->depth; i++)
    {
        matrix_destroy(adam->mw[i]);
        matrix_destroy(adam->mb[i]);
        matrix_destroy(adam->vw[i]);
        matrix_destroy(adam->vb[i]);

#ifdef OPEN_CL_EN
        matrix_cl_dystroy(adam->ocl, &adam->mw[i]);
        matrix_cl_dystroy(adam->ocl, &adam->mb[i]);
        matrix_cl_dystroy(adam->ocl, &adam->vw[i]);
        matrix_cl_dystroy(adam->ocl, &adam->vb[i]);
#endif
    }

    free(adam->mw);
    free(adam->mb);
    free(adam->vw);
    free(adam->vb);

    free(adam);
}

void adam_set(Adam *adam, float alpha, float beta1, float beta2, float epsilon)
{
    adam->alpha = alpha;
    adam->beta1t = adam->beta1 = beta1;
    adam->beta2t = adam->beta2 = beta2;
    adam->epsilon = epsilon;
}

void adam_reset(Adam *adam)
{
    adam->t = 0;
    adam->beta1t = adam->beta1;
    adam->beta2t = adam->beta2;

    for (int i = 0; i < adam->depth; i++)
    {
        matrix_clear(adam->mw[i]);
        matrix_clear(adam->mb[i]);
        matrix_clear(adam->vw[i]);
        matrix_clear(adam->vb[i]);

#ifdef OPEN_CL_EN
        matrix_cl_copy_to_device(adam->ocl, &adam->mw[i]);
        matrix_cl_copy_to_device(adam->ocl, &adam->mb[i]);
        matrix_cl_copy_to_device(adam->ocl, &adam->vw[i]);
        matrix_cl_copy_to_device(adam->ocl, &adam->vb[i]);
#endif
    }   
}

void adam_optimize(MLP *mlp, Adam *adam)
{
    adam->t++;

    for (int i = 0; i < adam->depth; i++)
    {
        Matrix *w = &mlp->layers[i].weights;
        Matrix *b = &mlp->layers[i].biases;
        Matrix *gw = &mlp->layers[i].gradWeights;
        Matrix *gb = &mlp->layers[i].gradBiases;
        Matrix *mw = &adam->mw[i];
        Matrix *mb = &adam->mb[i];
        Matrix *vw = &adam->vw[i];
        Matrix *vb = &adam->vb[i];

#ifdef OPEN_CL_EN
        cl_adam_optimize(adam->ocl, w, mw, vw, gw, adam->beta1t, adam->beta2t);
        cl_adam_optimize(adam->ocl, b, mb, vb, gb, adam->beta1t, adam->beta2t);
#else
        float mw1, mb1, vw1, vb1;

        for (int row = 0; row < w->rows; row++)
        {
            for (int col = 0; col < w->columns; col++)
            {
                int idx = row * w->columns + col;

                mw->data[idx] = adam->beta1 * mw->data[idx] + (1 - adam->beta1) * gw->data[idx];
                vw->data[idx] = adam->beta2 * vw->data[idx] + (1 - adam->beta2) * gw->data[idx] * gw->data[idx];
                mw1 = mw->data[idx] / (1 - adam->beta1t);
                vw1 = vw->data[idx] / (1 - adam->beta2t);

                w->data[idx] -= adam->alpha * (mw1 / (sqrt(vw1) + adam->epsilon));
            }
        }

        for (int row = 0; row < b->rows; row++)
        {
            for (int col = 0; col < b->columns; col++)
            {
                int idx = row * b->columns + col;

                mb->data[idx] = adam->beta1 * mb->data[idx] + (1 - adam->beta1) * gb->data[idx];
                vb->data[idx] = adam->beta2 * vb->data[idx] + (1 - adam->beta2) * gb->data[idx] * gb->data[idx];
                mb1 = mb->data[idx] / (1 - adam->beta1t);
                vb1 = vb->data[idx] / (1 - adam->beta2t);

                b->data[idx] -= adam->alpha * (mb1 / (sqrt(vb1) + adam->epsilon));
            }
        }
#endif
    }

    adam->beta1t *= adam->beta1;
    adam->beta2t *= adam->beta2;
}
