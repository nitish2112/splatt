#include <stdio.h>
#include "mttkrp.h"
#include "args.h"

int main() {
   mttkrp_csf(_tensors, _mats, _m, _thds, _mttkrp_ws, _opts);

   printf("%f\n", _mats[3]->vals[0]);

}
