#!/bin/sh
# 
# EXAMPLE MPICH SCRIPT FOR SGE
# To use, change "MPICH_JOB", "NUMBER_OF_CPUS" 
# and "MPICH_PROGRAM_NAME" to real values. 
#
# Your job name 
#$ -N jobname
#
#$ -o output/batchJobOutputs/jobname
#
# Use current working directory
#$ -cwd
#
# Join stdout and stderr
#$ -j y
#
# pe request for MPICH. Set your number of processors here.
#$ -pe mpi 1
#
# Run job through bash shell
#$ -S /bin/bash
#
# The following is for reporting only. It is not really needed
# to run the job. It will show up in your output file.
#

echo "Got $NSLOTS processors for jobname"

#
# Use full pathname to make sure we are using the right mpirun
#

~/miniconda3/envs/gerry/bin/python src/scripts/run.py

#
# Commands to do something with the data after the
# program has finished.
