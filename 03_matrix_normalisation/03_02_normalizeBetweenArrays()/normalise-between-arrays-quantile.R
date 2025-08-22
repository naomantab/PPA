#!/bin/Rscript

# Setup

if (!require("BiocManager", quietly = TRUE)) install.packages("BiocManager")
BiocManager::install(version = "3.21")

BiocManager::install('limma')
library(limma)

install.packages('dplyr')
library(dplyr)



# set current working directory
setwd("C:/Users/tnaom/OneDrive/Desktop/PPA/03_matrix_normalisation")

# load matrix with column names as header (note, this matrix has column 1 as phosphosite_IDs) # nolint
matrix <- read.csv('C:/Users/tnaom/OneDrive/Desktop/PPA/02_raw_matrix/RawMatrix_NoOutliers.csv', header = T) # nolint
matrix <- as.data.frame(matrix)  # Ensure it's a data framen

head(matrix)

#------#

# Save phosphosite IDs
phosphosite_IDs <- matrix$DatasetName

# Remove ID column, leaving only numeric data
matrix_numeric <- matrix[, -1]

# Transpose: now samples are rows, phosphosites are columns
matrix_numeric_t <- t(matrix_numeric)

# Quantile normalization
quantile_matrix <- normalizeBetweenArrays(as.matrix(matrix_numeric_t), method = 'quantile')

# Transpose back: phosphosites are rows again
quantile_matrix_t <- t(quantile_matrix)

# Add back phosphosite IDs as first column
quantile_normalised <- data.frame(DatasetName = phosphosite_IDs, quantile_matrix_t, check.names = FALSE)   

## Save the quantile normalized matrix
write.csv(quantile_normalised, 'C:/Users/tnaom/OneDrive/Desktop/PPA/03_matrix_normalisation/NBA-Matrix_Quantile.csv', row.names = F) # nolint
