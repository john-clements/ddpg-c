#ifndef __ACTIVATION_H__
#define __ACTIVATION_H__

/**
 * When not using an activation function, provide the ACTIVATION_NONE code.
 * This is equivalent to using the linear activation function f(x) = x.
 */
#define ACTIVATION_NONE    0

/** The linear activation function integer code. */
#define ACTIVATION_LINEAR  0

/** The sigmoid activation function integer code. */
#define ACTIVATION_SIGMOID 1

/** The tanh activation function integer code. */
#define ACTIVATION_TANH    2

/** The ReLu activation function integer code. */
#define ACTIVATION_RELU    3

/**
 * The definition of a pointer to an activation function.
 */
typedef double (*ActivationFunction)(double);

/**
 * \returns the pointer to the activation function that corresponds to the
 * given integer `code`.
 */
ActivationFunction getActivationFunction(int code);

/**
 * \returns the pointer to the derivative of the activation function that
 * corresponds to the given integer `code`.
 */
ActivationFunction getActivationFunctionDeriv(int code);

#endif
