#!/bin/bash

# Example launcher for AFsample2:

# https://github.com/wallnerlab/AFsample2

# This is a site-specific template, NOT a ready-to-run script.

# Before using it, you must:

# 1. Clone and follow the installation instructions in the original AFsample2 repository.

# 2. Replace all example paths below with paths valid on your system.

# 3. Ensure the Apptainer image, databases, model parameters, and input FASTA exist.

# 4. Check the AFsample2 documentation for the available flags and requirements.

# The paths and settings below are examples from a specific HPC setup.

# They will NOT work unchanged on another machine.

# This script assumes that the container and AFsample2 environment have

# already been configured correctly. It does not install AFsample2 or

# download the databases or model parameters.

# ---------------------------------------------------------------------

# Configuration: EDIT THESE PATHS AND SETTINGS FOR YOUR SYSTEM

# ---------------------------------------------------------------------

# Host path to the AlphaFold databases

DATABASES_HOST="/path/to/alphafold/databases"

# Host directory containing input FASTA files and output predictions

INPUT_HOST="/path/to/AFsample2/examples"

# Host path to the AFsample2 source directory

AFSAMPLE2_HOST="/path/to/AFsample2/AF_multitemplate"

# Host directory containing the model parameters

PARAMS_HOST="/path/to/alphafold/parameters"

# Path to the Apptainer container image

CONTAINER="/path/to/afsample2.sif"

# Input FASTA file, relative to INPUT_HOST

FASTA="/input/your_protein/your_protein.fasta"

# AFsample2 flag file, inside the container

FLAGFILE="/app/alphafold/AF_multitemplate/monomer_full_dbs.flag"

# ---------------------------------------------------------------------

# Run AFsample2

# ---------------------------------------------------------------------

apptainer run --nv 
-B "${DATABASES_HOST}:/databases" 
-B "${INPUT_HOST}:/input" 
-B "${AFSAMPLE2_HOST}:/app/alphafold/AF_multitemplate" 
-B "${PARAMS_HOST}:/params" 
"${CONTAINER}" 
--method afsample2 
--fasta_paths "${FASTA}" 
--flagfile "${FLAGFILE}" 
--nstruct 10 
--model_preset monomer 
--output_dir /input/ 
--use_precomputed_features=False 
--dropout=True 
--msa_rand_fraction=0.5
