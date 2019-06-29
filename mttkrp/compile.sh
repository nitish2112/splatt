#!/bin/bash

gcc -g -DNATIVE -DMTTKRP_MODE=1 *.c -lm -o mttkrp
