#include <stdio.h>
#include "mttkrp.h"
#include "../args.h"

#ifdef NATIVE
  #define BILLION 1000000000L
  #include <time.h>
#endif

int main() {
   #ifdef NATIVE
   struct timespec start, end;
   float diff;

   clock_gettime(CLOCK_MONOTONIC, &start); /* mark start time */
   #endif

   mttkrp_csf(_tensors, _mats, MTTKRP_MODE, _thds, _mttkrp_ws, _opts);

   #ifdef NATIVE
   clock_gettime(CLOCK_MONOTONIC, &end); /* mark the end time */
   diff += ( BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec );
   printf("Time taken: %f ns (%ld cycles)\n", diff,  (long int)(diff/0.5)); // number if cycles at 2GHz clock
   #endif

   printf("%f\n", _mats[3]->vals[0]);
}
