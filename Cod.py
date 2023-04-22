from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt

# Generate data
X, y = make_moons(n_samples=200, noise=0.3, random_state=200)

# Train a decision tree without restrictions
model = DecisionTreeClassifier(random_state=50)
model.fit(X, y)

# Train a decision tree with more fine-grained settings
model_restricted = DecisionTreeClassifier(min_samples_leaf=5, min_samples_split=5, max_depth=4, random_state=50)
model_restricted.fit(X, y)



def plot_decision_boundary(clf, X, y, axes, cmap):
    """
    Plots the decision boundary of a classifier along with the data points.

    Parameters:
    clf (object): classifier object with a `predict` method.
    X (array-like): feature data.
    y (array-like): target data.
    axes (list): limits of the plot axis.
    cmap (str): name of the colormap.

    Returns:
    None
    """
    # Create meshgrid for plotting
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]

    # Use the classifier to make predictions and plot decision boundaries
    y_pred = clf.predict(X_new).reshape(x1.shape)
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=cmap)
    plt.contour(x1, x2, y_pred, cmap="Greys", alpha=0.8)

    # Plot the data points
    colors = {"Wistia": ["#78785c", "#c47b27"], "Pastel1": ["red", "blue"]}
    markers = ("o", "^")
    for idx in (0, 1):
        plt.scatter(X[:, 0][y == idx], X[:, 1][y == idx],
                    color=colors[cmap][idx], marker=markers[idx], alpha=0.8, edgecolor='black', linewidth=1.5)


    # Set axis labels and limits
    plt.axis(axes)
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$", rotation=0)

# Plot the decision boundaries for the two models
fig, axes = plt.subplots(ncols=2, figsize=(12, 4), sharey=True)

# Plot the decision boundary for the first model with no restrictions
plt.sca(axes[0])
plot_decision_boundary(model, X, y, axes=[-1.5, 2.4, -1, 1.5], cmap="Wistia")
plt.title("Decision Boundary without Restrictions")
plt.scatter(X[25:26, 0], X[25:26, 1], s=200, facecolors='none', edgecolors='r', linewidths=2, label="Sample 1")
plt.scatter(X[161:162, 0], X[161:162, 1], s=200, facecolors='none', edgecolors='b', linewidths=2, label="Sample 2")
plt.legend(loc="upper right")

# Plot the decision boundary for the second model with restrictions
plt.sca(axes[1])
plot_decision_boundary(model_restricted, X, y, axes=[-1.5, 2.4, -1, 1.5], cmap="Wistia")
plt.title(f"Decision Boundary with Restrictions\n(min_samples_leaf={model_restricted.min_samples_leaf}, min_samples_split={model_restricted.min_samples_split}, max_depth={model_restricted.max_depth})")
plt.scatter(X[25:26, 0], X[25:26, 1], s=200, facecolors='none', edgecolors='r', linewidths=2, label="Sample 1")
plt.scatter(X[161:162, 0], X[161:162, 1], s=200, facecolors='none', edgecolors='b', linewidths=2, label="Sample 2")

# Add legend and set axis labels and limits
plt.legend(loc="upper right")
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$", rotation=0)
plt.axis([-1.5, 2.4, -1, 1.5])
plt.show()
