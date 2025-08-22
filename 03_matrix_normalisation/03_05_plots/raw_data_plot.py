import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import logging

# Set up logging
logging.basicConfig(filename='/data/home/bt23917/PPA/03_matrix_normalisation/raw_data_debug.log',
                    level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

start_time = time.time()

logging.info("Script started.")

# -------------------- #
# Left-hand side plots (raw)
# -------------------- #
logging.info("Reading raw data...")
data_raw = pd.read_csv('/data/home/bt23917/PPA/03_matrix_normalisation/RawMatrix.csv', index_col=0)  # Updated path
data_raw = data_raw.apply(pd.to_numeric, errors='coerce')

# Create a 1x2 grid of subplots (only two plots)
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Left plot: scatter plot
logging.info("Creating scatter plot for raw data...")
x_raw = np.arange(data_raw.size)
ax[0].scatter(x_raw, data_raw.values.flatten(), c=x_raw, cmap="plasma", s=10)
ax[0].set_title('Raw log2 phospophorylation values')
ax[0].set_ylabel('Quantification value')
ax[0].set_xticks([])
ax[0].set_xlabel('Dataset')

# Right plot: mean line plot
logging.info("Creating mean line plot for raw data...")
means_raw = data_raw.mean(axis=1)
ax[1].plot(means_raw.index, means_raw.values, color='blue')
ax[1].set_title('Means of raw log2 phospophorylation values')
ax[1].set_ylabel('Mean quantification value')
ax[1].set_xticks([])
ax[1].set_xlabel('Dataset')

# Save the raw plot
logging.info("Saving raw data plot...")
plt.tight_layout()
plt.savefig('/data/home/bt23917/PPA/03_matrix_normalisation/raw_normalisation_plots.png', dpi=300)  # Updated path
plt.clf()

end_time = time.time()
logging.info(f"Script finished. Runtime: {end_time - start_time:.2f} seconds.")