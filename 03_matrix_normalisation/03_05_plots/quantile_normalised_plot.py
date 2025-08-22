import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import logging

# Set up logging
logging.basicConfig(filename='/data/home/bt23917/PPA/03_matrix_normalisation/quantile_normalised_debug.log',
                    level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

start_time = time.time()

logging.info("Script started.")

# -------------------- #
# Middle Right side plots (Quantile normalised)
# -------------------- #
logging.info("Reading quantile normalised data...")
data_quan = pd.read_csv('/data/home/bt23917/PPA/03_matrix_normalisation/NormalisedMatrix(Quantile).csv', index_col=0)  # Updated path
data_quan = data_quan.apply(pd.to_numeric, errors='coerce')

# Create a 1x2 grid of subplots (only two plots)
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Left plot: scatter plot
logging.info("Creating scatter plot for quantile normalised data...")
x_quan = np.arange(data_quan.size)
ax[0].scatter(x_quan, data_quan.values.flatten(), c=x_quan, cmap="plasma", s=10)
ax[0].set_title('Normalised between arrays (quantile) \nand min-max scaled log2 \nphospophorylation values')
ax[0].set_ylabel('Quantification value')
ax[0].set_xticks([])
ax[0].set_xlabel('Dataset')

# Right plot: mean line plot
logging.info("Creating mean line plot for quantile normalised data...")
means_quan = data_quan.mean(axis=1)
ax[1].plot(means_quan.index, means_quan.values, color='blue')
ax[1].set_title('Means of normalised between arrays \n (quantile) and min-max scaled log2 \nphospophorylation values')
ax[1].set_ylabel('Mean quantification value')
ax[1].set_xticks([])
ax[1].set_xlabel('Dataset')
ax[1].set_ylim(0, 1)  # Set y-axis limits between 0 and 1

# Save the quantile normalised plot
logging.info("Saving quantile normalised data plot...")
plt.tight_layout()
plt.savefig('/data/home/bt23917/PPA/03_matrix_normalisation/quantile_normalised_plots.png', dpi=300)  # Updated path
plt.clf()

end_time = time.time()
logging.info(f"Script finished. Runtime: {end_time - start_time:.2f} seconds")
