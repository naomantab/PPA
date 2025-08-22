import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

start_time = time.time()

# Create a 2x2 grid of subplots
fig, ax = plt.subplots(2, 4, figsize=(22, 10))


# -------------------- #
# Left-hand side plots (raw)
# -------------------- #

# data_raw = pd.read_csv('C:/Users/tnaom/OneDrive/Desktop/PPA/02_raw_matrix/RawMatrix.csv', index_col=0)
data_raw = pd.read_csv('/data/home/bt23917/PPA/RawMatrix.csv', index_col=0)
data_raw = data_raw.apply(pd.to_numeric, errors='coerce')

# Top-left: scatter plot
x_raw = np.arange(data_raw.size)
ax[0][0].scatter(x_raw, data_raw.values.flatten(), c=x_raw, cmap="plasma", s=10)
ax[0][0].set_title('Raw log2 phospophorylation values')
ax[0][0].set_ylabel('Quantification value')
ax[0][0].set_xticks([])
ax[0][0].set_xlabel('Dataset')

# Bottom-left: mean line plot
means_raw = data_raw.mean(axis=1)
ax[1][0].plot(means_raw.index, means_raw.values, color='blue')
ax[1][0].set_ylabel('Mean Quantification Value')
ax[1][0].set_xticks([])
ax[1][0].set_xlabel('Dataset')

data_raw_numeric = data_raw.select_dtypes(include=[np.number])
print("Raw matrix shape:", data_raw_numeric.shape)
print("Scatter shape:", data_raw_numeric.values.flatten().shape)
print("Mean shape:", means_raw.shape)

# -------------------- #
# Middle left plots (normalised)
# -------------------- #

# data_norm = pd.read_csv('C:/Users/tnaom/OneDrive/Desktop/PPA/03_matrix_normalisation/NormalisedMatrix.csv', index_col=0)
data_norm = pd.read_csv('/data/home/bt23917/PPA/NormalisedMatrix.csv', index_col=0)
data_norm = data_norm.apply(pd.to_numeric, errors='coerce')

# Top-right: scatter plot
x_norm = np.arange(data_norm.size)
ax[0][1].scatter(x_norm, data_norm.values.flatten(), c=x_norm, cmap="plasma", s=10)
ax[0][1].set_title('MinMax Normalised log2 \nphospophorylation values')
ax[0][1].set_ylabel('Quantification value')
ax[0][1].set_xticks([])
ax[0][1].set_xlabel('Dataset')

# Bottom-right: mean line plot
means_norm = data_norm.mean(axis=1)
ax[1][1].plot(means_norm.index, means_norm.values, color='blue')
ax[1][1].set_title('Mean Quantification (Normalised)')
ax[1][1].set_xticks([])
ax[1][1].set_xlabel('Dataset')
ax[1][1].set_ylim(0, 1)  # Set y-axis limits between 0 and 1

data_norm_numeric = data_norm.select_dtypes(include=[np.number])
print("Normalised matrix shape:", data_norm_numeric.shape)
print("Scatter shape:", data_norm_numeric.values.flatten().shape)
print("Mean shape:", means_norm.shape)



# # ----------------------------------------------- #
# # Middle Right side plots (Quantile normalised)
# # ----------------------------------------------- #

# data_quan = pd.read_csv('C:/Users/tnaom/OneDrive/Desktop/PPA/03_matrix_normalisation/NormalisedMatrix(Quantile).csv', index_col=0)
data_quan = pd.read_csv('/data/home/bt23917/PPA/NormalisedMatrix(Quantile).csv', index_col=0)
data_quan = data_quan.apply(pd.to_numeric, errors='coerce')

# Top-left: scatter plot
x_quan = np.arange(data_quan.size)
ax[0][2].scatter(x_quan, data_quan.values.flatten(), c=x_quan, cmap="plasma", s=10)
ax[0][2].set_title('Normalised between arrays (Quantile) \nand min-max scaled log2 \nphospophorylation values')
ax[0][2].set_ylabel('Quantification value')
ax[0][2].set_xticks([])
ax[0][2].set_xlabel('Dataset')

# Bottom-left: mean line plot
means_quan = data_quan.mean(axis=1) # looks at rows
ax[1][2].plot(means_quan.index, means_quan.values, color='blue')
ax[1][2].set_ylabel('Mean Quantification Value')
ax[1][2].set_xticks([])
ax[1][2].set_xlabel('Dataset')
ax[1][2].set_ylim(0, 1)  # Set y-axis limits between 0 and 1

data_quan_numeric = data_quan.select_dtypes(include=[np.number])
print("Raw matrix shape:", data_quan_numeric.shape)
print("Scatter shape:", data_quan_numeric.values.flatten().shape)
print("Mean shape:", means_quan.shape)

# ------------------------------------------------------ #
# Right mid side plots (Quantile normalised with z-score)
# ------------------------------------------------------ #

# data_cyclo = pd.read_csv('C:/Users/tnaom/OneDrive/Desktop/PPA/03_matrix_normalisation/NormalisedMatrix(Quantile)(Z-score).csv', index_col=0)
data_cyclo = pd.read_csv('/data/home/bt23917/PPA/NormalisedMatrix(Quantile)(Z-score).csv', index_col=0)
data_cyclo = data_cyclo.apply(pd.to_numeric, errors='coerce')

# Top-left: scatter plot
x_cyclo = np.arange(data_cyclo.size)
ax[0][3].scatter(x_cyclo, data_cyclo.values.flatten(), c=x_cyclo, cmap="plasma", s=10)
ax[0][3].set_title('Normalised between arrays (Quantile) \n, min-max scaled with z-score, \nlog 2phospophorylation values')
ax[0][3].set_ylabel('Quantification value')
ax[0][3].set_xticks([])
ax[0][3].set_xlabel('Dataset')

# Bottom-left: mean line plot
means_cyclo = data_cyclo.mean(axis=1)
ax[1][3].plot(means_cyclo.index, means_cyclo.values, color='blue')
ax[1][3].set_ylabel('Mean Quantification Value')
ax[1][3].set_xticks([])
ax[1][3].set_xlabel('Dataset')
ax[1][3].set_ylim(0, 1)  # Set y-axis limits between 0 and 1

data_cyclo_numeric = data_cyclo.select_dtypes(include=[np.number])
print("Raw matrix shape:", data_cyclo_numeric.shape)
print("Scatter shape:", data_cyclo_numeric.values.flatten().shape)
print("Mean shape:", means_cyclo.shape)

# # ----------------- #

# # ------------------------------------------------------------------------- #
# # Right side plots (Quantile normalised with z-score and robust scaling)
# # ------------------------------------------------------------------------- #

# data_cyclo2 = pd.read_csv('C:/Users/tnaom/OneDrive/Desktop/PPA/03_matrix_normalisation/NormalisedMatrix(Quantile)(Z-score)(robust-scaling).csv', index_col=0)
# data_cyclo2 = data_cyclo2.apply(pd.to_numeric, errors='coerce')

# # Top-left: scatter plot
# x_cyclo2 = np.arange(data_cyclo2.size)
# ax[0][4].scatter(x_cyclo2, data_cyclo2.values.flatten(), c=x_cyclo2, cmap="plasma", s=10)
# ax[0][4].set_title('Normalised between arrays (Quantile) \nand min-max scaled with z-score & robust \nscaling log2phospophorylation values')
# ax[0][4].set_ylabel('Quantification value')
# ax[0][4].set_xticks([])
# ax[0][4].set_xlabel('Dataset')

# # Bottom-left: mean line plot
# means_cyclo2 = data_cyclo2.mean(axis=1)
# ax[1][4].plot(means_cyclo2.index, means_cyclo2.values, color='blue')
# ax[1][4].set_ylabel('Mean Quantification Value')
# ax[1][4].set_xticks([])
# ax[1][4].set_xlabel('Dataset')
# ax[1][4].set_ylim(0, 1)  # Set y-axis limits between 0 and 1

# data_cyclo2_numeric = data_cyclo2.select_dtypes(include=[np.number])
# print("Raw matrix shape:", data_cyclo2_numeric.shape)
# print("Scatter shape:", data_cyclo2_numeric.values.flatten().shape)
# print("Mean shape:", means_cyclo2.shape)

# ----------------- #

plt.tight_layout()
# plt.savefig('C:/Users/tnaom/OneDrive/Desktop/PPA/03_matrix_normalisation/03_05_plots/normalisation_plots.png', dpi=300)
plt.savefig('/data/home/bt23917/PPA/normalisation_plots.png', dpi=300)    

end_time = time.time()
print(f"\nRuntime: {end_time - start_time:.2f} seconds")

