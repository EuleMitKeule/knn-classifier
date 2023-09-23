from typing import Callable

import numpy as np


class Math:
    @staticmethod
    def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
        return np.sqrt(np.sum(np.square(x - y)))

    @staticmethod
    def manhattan_distance(x: np.ndarray, y: np.ndarray) -> float:
        return np.sum(np.abs(x - y))

    @staticmethod
    def chebyshev_distance(x: np.ndarray, y: np.ndarray) -> float:
        return np.max(np.abs(x - y))

    @staticmethod
    def knn_predict(
        x: np.ndarray,
        x_train: np.ndarray,
        y_train: np.ndarray,
        k: int,
        d: Callable[[np.ndarray, np.ndarray], float],
    ) -> tuple[np.ndarray, np.ndarray]:
        x_normalized: np.ndarray = Math.normalize(x)
        x_train_normalized: np.ndarray = Math.normalize(x_train)

        y_hat_classification: np.ndarray = np.zeros(len(x_normalized), dtype=int)
        y_hat_regression: np.ndarray = np.zeros(len(x_normalized), dtype=float)

        for i, x_i in enumerate(x_normalized):
            distances = np.array([d(x_i, x_j) for x_j in x_train_normalized])
            nearest_neighbors = np.argsort(distances)[:k]
            nearest_classes = y_train[nearest_neighbors]
            y_hat_classification[i] = np.argmax(np.bincount(nearest_classes))
            y_hat_regression[i] = np.mean(nearest_classes)

        return y_hat_classification, y_hat_regression

    @staticmethod
    def normalize(x: np.ndarray) -> np.ndarray:
        """Normalizes the given data."""
        return (x - np.mean(x, axis=0)) / np.std(x, axis=0)

    @staticmethod
    def confusion_matrix(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        """Returns the confusion matrix of the given data."""
        n_classes = len(np.unique(y))
        matrix = np.zeros((n_classes, n_classes), dtype=int)
        for i in range(len(y)):
            matrix[y[i]][y_hat[i]] += 1
        return matrix

    @staticmethod
    def accuracy(y: np.ndarray, y_hat: np.ndarray) -> float:
        """Returns the accuracy of the given data."""
        return np.mean(y == y_hat)
