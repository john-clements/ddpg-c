#!/bin/bash

## Parameters
PROJECT=target_seeker

## Build Variables
CLEAN_ONLY=0
LOW_LEVEL=0
RUN=0
MAKE_PARAM+=
EXPORT_VAR+=

# Build use case instructions
if [[ $1 = "?" ]]; then
	echo "Build Arguments:"
	echo "clean             -> Removes output binaries."
	echo "WINDOWS           -> Builds support for Windows"
	echo "RUN               -> Builds and executes"
	exit 1
fi

# Build use case switches
echo "*****************************************************************"
if [[ $# -eq 0 ]]; then
	echo "BUILD TYPE:			ALL"
else
	echo "BUILD TYPE:			"$@

fi

if [[ $@ == **clean** ]]; then
	CLEAN_ONLY=1
fi

if [[ $@ == **RUN** ]]; then
	RUN=1
fi

if [[ $@ == **WINDOWS** ]]; then
	WINDOWS=1
	MAKE_PARAM+="WINDOWS=1"
	EXPORT_VAR+="WINDOWS"
fi

echo "*****************************************************************"

# Clean
echo "Cleaning..."
rm -rf bin
make clean
rm -rf obj
echo "*****************************************************************"

if [[ $CLEAN_ONLY == 1 ]]; then
	exit 1
fi

# Build
echo "Building..."
mkdir bin
mkdir obj
make $MAKE_PARAM
echo "*****************************************************************"
if [[ $RUN == 1 ]]; then
	./bin/$PROJECT
	echo "*****************************************************************"
fi
