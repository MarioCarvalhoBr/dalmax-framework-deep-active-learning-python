import torch
import numpy as np
import time 

from core.tools.SSL.src.clusters import HierarchicalCluster
from core.tools.SSL.src import (
  hierarchical_kmeans_gpu as hkmg,
  hierarchical_sampling as hs
)

def make_ring(n, rmin, rmax):
    r = np.random.rand(n) * (rmax - rmin) + rmin
    alpha = np.random.rand(n) * 2 * np.pi
    return np.vstack([r * np.cos(alpha), r * np.sin(alpha)]).T

data = np.concatenate([
    make_ring(20000, 0.7, 1.0) + np.array([-2.2, 1.]),
    make_ring(200, 0.7, 1.0) + np.array([0., 1.]),
    make_ring(1000, 0.7, 1.0) + np.array([2.2, 1.]),
    make_ring(500, 0.7, 1.0) + np.array([-1.2, 0.2]),
    make_ring(8000, 0.7, 1.0) + np.array([1.2, 0.2]),
])

time_start = time.time()

clusters = hkmg.hierarchical_kmeans_with_resampling(
  data=torch.tensor(data, device="cuda", dtype=torch.float32),
  n_clusters=[1000, 300],
  n_levels=2,
  sample_sizes=[15, 2],
  verbose=False,
)

cl = HierarchicalCluster.from_dict(clusters)
sampled_indices = hs.hierarchical_sampling(cl, target_size=1000)

# Show matplotlib figure
import matplotlib.pyplot as plt

# Original data
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1], s=1, alpha=0.5)
plt.title("Original Data")
plt.axis('equal')   
plt.grid(True)
# Sampled data
plt.subplot(1, 2, 2)
plt.scatter(data[sampled_indices, 0], data[sampled_indices, 1], s=10, color='red', alpha=0.7)
plt.title("Sampled Data")
plt.axis('equal')
plt.grid(True)
plt.show()

# Save figure pdf
plt.savefig("sampled_data.pdf", format="pdf")

final_time = f"Time taken: {time.time() - time_start:.2f} seconds"

# Salvar em arquivo
with open("time_log.txt", "w") as f:
    f.write(final_time)