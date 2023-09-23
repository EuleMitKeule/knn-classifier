import numpy as np
import pandas as pd

from knn_classifier import KnnClassifier


def main() -> None:
    DATA_PATH: str = "data.csv"
    TRAIN_VALIDATION_TEST_SPLIT = (0.7, 0.2, 0.1)

    dataframe: pd.DataFrame = pd.read_csv(DATA_PATH)
    data: np.ndarray = dataframe.to_numpy()

    np.random.shuffle(data)

    n: int = len(data)
    train, validation, test = np.split(
        data,
        [
            int(TRAIN_VALIDATION_TEST_SPLIT[0] * n),
            int((TRAIN_VALIDATION_TEST_SPLIT[0] + TRAIN_VALIDATION_TEST_SPLIT[1]) * n),
        ],
    )

    x_train: np.ndarray = train[:, 1:]
    y_train: np.ndarray = train[:, 0].astype(int)

    x_validate: np.ndarray = validation[:, 1:]
    y_validate: np.ndarray = validation[:, 0].astype(int)

    x_test: np.ndarray = test[:, 1:]
    y_test: np.ndarray = test[:, 0].astype(int)

    knn_classifier = KnnClassifier()

    knn_classifier.fit(x_train, y_train, x_validate, y_validate)

    _ = knn_classifier.predict(x_test)

    knn_classifier.visualize_optimal(x_test, y_test)
    knn_classifier.visualize_all(x_test, y_test)

    x1_label = dataframe.columns[1]
    x2_label = dataframe.columns[2]
    knn_classifier.visualize_model(x_validate, y_validate, 0, 1, x1_label, x2_label)


if __name__ == "__main__":
    main()
