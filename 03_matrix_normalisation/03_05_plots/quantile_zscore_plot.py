import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import logging

# Set up logging
logging.basicConfig(filename='/data/home/bt23917/PPA/03_matrix_normalisation/quantile_zscore_debug.log',
                    level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

start_time = time.time()

logging.info("Script started.")

# -------------------- #
# Right mid side plots (Quantile normalised with z-score)
# -------------------- #
logging.info("Reading quantile z-score normalised data...")
data_cyclo = pd.read_csv('/data/home/bt23917/PPA/03_matrix_normalisation/NormalisedMatrix(Quantile)(Z-score).csv', index_col=0)  # Updated path
data_cyclo = data_cyclo.apply(pd.to_numeric, errors='coerce')

# Create a 1x2 grid of subplots (only two plots)
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Left plot: scatter plot
logging.info("Creating scatter plot for quantile z-score normalised data...")
x_cyclo = np.arange(data_cyclo.size)
ax[0].scatter(x_cyclo, data_cyclo.values.flatten(), c=x_cyclo, cmap="plasma", s=10)
ax[0].set_title('Normalised between arrays (quantile),\n min-max scaled with z-score, \nlog 2phospophorylation values')
ax[0].set_ylabel('Quantification value')
ax[0].set_xticks([])
ax[0].set_xlabel('Dataset')

# Right plot: mean line plot
logging.info("Creating mean line plot for quantile z-score normalised data...")
means_cyclo = data_cyclo.mean(axis=1)
ax[1].plot(means_cyclo.index, means_cyclo.values, color='blue')
ax[1].set_title('Means of normalised between arrays \n(quantile), min-max scaled with z-score, \nlog 2phospophorylation values')
ax[1].set_ylabel('Mean quantification value')
ax[1].set_xticks([])
ax[1].set_xlabel('Dataset')
ax[1].set_ylim(0, 1)  # Set y-axis limits between 0 and 1

# Save the quantile z-score normalised plot
logging.info("Saving quantile z-score normalised data plot...")
plt.tight_layout()
plt.savefig('/data/home/bt23917/PPA/03_matrix_normalisation/quantile_zscore_plots.png', dpi=300)  # Updated path
plt.clf()

end_time = time.time()
logging.info(f"Script finished. Runtime: {end_time - start_time:.2f} seconds")
