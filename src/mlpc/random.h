#ifndef __RANDOM_H__
#define __RANDOM_H__

/**
 * Initializes the seed. This is done when initializing the library.
 */
void deepc_random_init();

/**
 * \returns A random number of type int between ´min´ and ´max´, both extremes
 * inclusive.
 */
int deepc_random_int(int min, int max);

/**
 * \returns A random number of type double between ´min´ and ´max´.
 */
double deepc_random_double(double min, double max);

#endif
