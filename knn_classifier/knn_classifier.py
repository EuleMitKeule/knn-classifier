from itertools import product
from typing import Callable

import numpy as np

from knn_classifier.math import Math
from knn_classifier.plot import Plot

D_VALUES_DEFAULT: set[Callable[[np.ndarray, np.ndarray], float]] = {
    Math.euclidean_distance,
    Math.manhattan_distance,
    Math.chebyshev_distance,
}

K_VALUES_DEFAULT: set[int] = set(range(1, 10))


class KnnClassifier:
    k_values: list[int]
    d_values: list[Callable[[np.ndarray, np.ndarray], float]]
    is_trained: bool
    k_optimal: int
    d_optimal: Callable[[np.ndarray, np.ndarray], float]
    x_predict: np.ndarray
    y_predict: np.ndarray

    def __init__(
        self,
        k_values: set[int] = K_VALUES_DEFAULT,
        d_values: set[Callable[[np.ndarray, np.ndarray], float]] = D_VALUES_DEFAULT,
    ):
        """Initializes the model."""

        self.k_values = list(k_values)
        self.d_values = list(d_values)
        self.is_trained = False
        self.k_optimal = self.k_values[0]
        self.d_optimal = self.d_values[0]

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_validation: np.ndarray,
        y_validation: np.ndarray,
    ) -> None:
        """Fits the model to the training data and finds the optimal k and d values."""

        accuracy_optimal: float = 0.0

        for k, d in product(self.k_values, self.d_values):
            y_hat_validation, _ = Math.knn_predict(x_validation, x_train, y_train, k, d)
            accuracy = Math.accuracy(y_validation, y_hat_validation)

            if accuracy > accuracy_optimal:
                accuracy_optimal = accuracy
                self.k_optimal = k
                self.d_optimal = d

        self.x_predict = x_train
        self.y_predict = y_train
        self.is_trained = True

    def predict(
        self,
        x: np.ndarray,
        x_predict: np.ndarray | None = None,
        y_predict: np.ndarray | None = None,
        k: int | None = None,
        d: Callable[[np.ndarray, np.ndarray], float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predicts the class of the given data."""

        if not self.is_trained:
            raise Exception("Model must be trained before predicting.")

        if k is None:
            k = self.k_optimal

        if d is None:
            d = self.d_optimal

        if x_predict is None:
            x_predict = self.x_predict

        if y_predict is None:
            y_predict = self.y_predict

        return Math.knn_predict(x, x_predict, y_predict, k, d)

    def visualize_model(
        self,
        x_validate: np.ndarray,
        y_validate: np.ndarray,
        x1_feature_index: int,
        x2_feature_index: int,
        x1_feature_name: str,
        x2_feature_name: str,
    ) -> None:
        """Visualize the models classification behaviour."""

        Plot.model(
            self.x_predict,
            self.y_predict,
            x_validate,
            y_validate,
            self.k_optimal,
            self.d_optimal,
            x1_feature_index,
            x2_feature_index,
            x1_feature_name,
            x2_feature_name,
        )

    def visualize_optimal(self, x: np.ndarray, y: np.ndarray) -> None:
        """Visualize the results for the optimal k and d values."""

        if not self.is_trained:
            raise Exception("Model must be trained before visualizing.")

        y_hat, _ = self.predict(x)

        confusion_matrix = Math.confusion_matrix(y, y_hat)
        accuracy = Math.accuracy(y, y_hat)

        Plot.confusion_matrix(
            confusion_matrix, accuracy, self.k_optimal, self.d_optimal
        )

    def visualize_all(self, x: np.ndarray, y: np.ndarray) -> None:
        """Visualize the results for all k and d values."""

        if not self.is_trained:
            raise Exception("Model must be trained before visualizing.")

        accuracies: dict[
            tuple[int, Callable[[np.ndarray, np.ndarray], float]], float
        ] = {}

        for k, d in product(self.k_values, self.d_values):
            y_hat, _ = self.predict(x, k=k, d=d)
            accuracy = Math.accuracy(y, y_hat)
            accuracies[(k, d)] = accuracy

        Plot.accuracies(accuracies)
