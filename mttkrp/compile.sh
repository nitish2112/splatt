#!/bin/bash

gcc -g -DMTTKRP_MODE=$1 -mcmodel=medium *.c -lm -o mttkrp.x86
