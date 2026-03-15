#include <stdio.h>
#include "mlp.h"
#include "adam.h"
#include "loss.h"

int main()
{
#ifdef OPEN_CL_EN
    open_cl_test();

#else

    printf("OPEN CL NOT ENABLED\n");

#endif

    return 0;
}
