#!/bin/bash

premake5 gmake

cd build 

make config=debug

echo "finished"
cd ..