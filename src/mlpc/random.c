#include <stdlib.h>
#include <time.h>
#include "random.h"

void deepc_random_init()
{
    srand((unsigned int)time(NULL));
}

int deepc_random_int(int min, int max)
{
    return rand() % (max - min + 1) + min;
}

float deepc_random_float(float min, float max)
{
    return ((float)rand() / RAND_MAX) * (max - min) + min;
}