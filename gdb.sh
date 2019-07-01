#!/bin/bash

gdb -args ./build/Linux-x86_64/bin/splatt cpd $1.tns -r $2 -t 1
