from typing import Callable

import numpy as np
from matplotlib import pyplot as plt

from knn_classifier.math import Math


class Plot:
    """Plots the data."""

    @staticmethod
    def confusion_matrix(
        confusion_matrix: np.ndarray,
        accuracy: float,
        k_used: int | None = None,
        d_used: Callable[[np.ndarray, np.ndarray], float] | None = None,
    ) -> None:
        """Plots the confusion matrix."""

        fig, ax = plt.subplots()

        title = f"Confusion Matrix (k={k_used}, d={d_used.__name__})"
        fig.suptitle(title)
        fig.canvas.manager.set_window_title(title)

        ax.set_title(f"Accuracy: {accuracy * 100:.2f}%")

        im = ax.imshow(confusion_matrix)

        ax.set_xticks(np.arange(confusion_matrix.shape[0]))
        ax.set_yticks(np.arange(confusion_matrix.shape[1]))

        ax.set_xticklabels(np.arange(confusion_matrix.shape[0]))
        ax.set_yticklabels(np.arange(confusion_matrix.shape[1]))

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                percentage: float = (
                    confusion_matrix[i, j] / np.sum(confusion_matrix[i]) * 100
                )
                ax.text(
                    j,
                    i,
                    f"{confusion_matrix[i, j]} ({percentage:.2f}%)",
                    ha="center",
                    va="center",
                    color="w",
                )

        fig.colorbar(im)
        plt.show()

    @staticmethod
    def accuracies(
        accuracies: dict[tuple[int, Callable[[np.ndarray, np.ndarray], float]], float]
    ) -> None:
        """Plots the accuracies."""

        k_values = set(map(lambda x: x[0], accuracies.keys()))
        d_values = set(map(lambda x: x[1], accuracies.keys()))

        fig, axs = plt.subplots(len(d_values), sharex=True, sharey=True)

        for i, d in enumerate(d_values):
            axs[i].set_title(f"d={d.__name__}")
            axs[i].set_xlabel("k")
            axs[i].set_ylabel("Accuracy")

            x = []
            y = []

            for k in k_values:
                x.append(k)
                y.append(accuracies[(k, d)])

            axs[i].bar(x, y)

        fig.suptitle("Accuracies")
        fig.canvas.manager.set_window_title("Accuracies")

        plt.show()

    @staticmethod
    def model(
        x_predict: np.ndarray,
        y_predict: np.ndarray,
        x_validate: np.ndarray,
        y_validate: np.ndarray,
        k: int,
        d: Callable[[np.ndarray, np.ndarray], float],
        x1_feature_index: int,
        x2_feature_index: int,
        x1_feature_name: str,
        x2_feature_name: str,
    ) -> None:
        """Plots the model."""

        x1_predict = x_predict[:, x1_feature_index]
        x2_predict = x_predict[:, x2_feature_index]
        x_predict_selected = x_predict[:, [x1_feature_index, x2_feature_index]]

        x1_validate = x_validate[:, x1_feature_index]
        x2_validate = x_validate[:, x2_feature_index]

        x1_min, x1_max = (
            min(x1_predict.min(), x1_validate.min()),
            max(x1_predict.max(), x1_validate.max()),
        )
        x2_min, x2_max = (
            min(x2_predict.min(), x2_validate.min()),
            max(x2_predict.max(), x2_validate.max()),
        )

        x1_grid, x2_grid = np.meshgrid(
            np.arange(x1_min - 1, x1_max + 1, 1),
            np.arange(x2_min - 1, x2_max + 1, 1),
        )

        fig, ax = plt.subplots()

        title = f"Model (k={k}, d={d.__name__})"
        fig.suptitle(title)
        fig.canvas.manager.set_window_title(title)

        ax.set_xlabel(x1_feature_name)
        ax.set_ylabel(x2_feature_name)

        x_grid_selected = np.column_stack((x1_grid.ravel(), x2_grid.ravel()))

        x_grid = np.zeros((len(x_grid_selected), x_predict.shape[1]))
        x_grid[:, [x1_feature_index, x2_feature_index]] = x_grid_selected
        for i in range(x_predict.shape[1]):
            if i != x1_feature_index and i != x2_feature_index:
                x_grid[:, i] = np.mean(x_predict[:, i])

        # y_hat_selected, _ = Math.knn_predict(x_grid, x_predict, y_predict, k, d)
        y_hat_selected, _ = Math.knn_predict(
            x_grid_selected, x_predict_selected, y_predict, k, d
        )

        plt.pcolormesh(
            x1_grid,
            x2_grid,
            y_hat_selected.reshape(x1_grid.shape),
            cmap="viridis",
            shading="auto",
        )
        plt.scatter(
            x1_validate[:20],
            x2_validate[:20],
            c=y_validate[:20],
            s=50,
            cmap="viridis",
            marker="^",
            edgecolor="black",
        )
        plt.scatter(
            x1_predict[:40],
            x2_predict[:40],
            c=y_predict[:40],
            s=50,
            cmap="viridis",
            edgecolor="black",
        )

        plt.show()
