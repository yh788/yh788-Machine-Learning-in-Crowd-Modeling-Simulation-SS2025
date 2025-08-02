import matplotlib
matplotlib.use("TkAgg")

from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3  # noqa: F401
import datafold.dynfold as dfold
import datafold.pcfold as pfold
from datafold.utils.plot import plot_pairwise_eigenvector

# Simply generate dataset with make_swiss_roll, random state is set randomly as 608 to make the result reproducible
X, t = make_swiss_roll(n_samples=5000, noise=0, random_state=608, hole=False)
X_pcm = pfold.PCManifold(X)
X_pcm.optimize_parameters()

print(f"epsilon={X_pcm.kernel.epsilon}, cut-off={X_pcm.cut_off}")

# Initialize the DiffusionMaps model.
dmap = dfold.DiffusionMaps(
    kernel=pfold.GaussianKernel(
        epsilon=X_pcm.kernel.epsilon, distance=dict(cut_off=X_pcm.cut_off)
    ),
    n_eigenpairs=9,
)

# Fit the model to the data.
dmap = dmap.fit(X)
evecs, evals = dmap.eigenvectors_, dmap.eigenvalues_

# Use datafold's built-in utility to plot the computed eigenvectors against each other.
plot_pairwise_eigenvector(
    eigenvectors=dmap.eigenvectors_,
    n=1,
    fig_params=dict(figsize=[13, 15]),
    scatter_params=dict(cmap=plt.cm.Spectral, c=t),
)
plt.suptitle("Eigenvectors of Swiss Roll computed with datafold")
plt.show()
plt.pause(100)

# P.S. the code is basically from https://datafold-dev.gitlab.io/datafold/tutorial_03_dmap_scurve.html, and the link in
# exercise sheet is invalid.
