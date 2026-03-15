#include <math.h>
#include "activation.h"

float activation_linear(float x)
{
    return x;
}

float activation_linearDeriv(float y)
{
    return 1;
}

float activation_sigmoid(float x)
{
    if (x >= 0)
        return 1.0 / (1 + exp(-x));
    else
        return 1.0 - (1.0 / 1 + exp(x));
}

float activation_sigmoidDeriv(float y)
{
    return y * (1 - y);
}

float activation_tanh(float x)
{
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

float activation_tanhDeriv(float y)
{
    return 1 - y*y;
}

float activation_relu(float x)
{
    if (x >= 0)
        return x;
    else
        return 0;
}

float activation_reluDeriv(float y)
{
    if (y > 0)
        return 1;
    else
        return 0;
}

ActivationFunction getActivationFunction(int code)
{
    switch (code)
    {
        case ACTIVATION_SIGMOID:
            return activation_sigmoid;
        case ACTIVATION_TANH:
            return activation_tanh;
        case ACTIVATION_RELU:
            return activation_relu;
        default:
            return activation_linear;
    }
}

ActivationFunction getActivationFunctionDeriv(int code)
{
    switch (code)
    {
        case ACTIVATION_SIGMOID:
            return activation_sigmoidDeriv;
        case ACTIVATION_TANH:
            return activation_tanhDeriv;
        case ACTIVATION_RELU:
            return activation_reluDeriv;
        default:
            return activation_linearDeriv;
    }
}