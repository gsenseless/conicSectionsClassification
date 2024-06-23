import os
import pandas as pd
import numpy as np
import click
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.utils import compute_class_weight
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import confusion_matrix
from autogluon.tabular import TabularPredictor


SYNTHETIC_DATASET_SIZE: int = 50123


def generate_data() -> tuple:
    np.random.seed(222)
    
    coefficients = np.random.uniform(-1, 1, size=(SYNTHETIC_DATASET_SIZE, 6))
    # Let's add some zero values in order to obtain fewer ellipses and hyperbolas.
    mask = np.random.randint(0, 6, size=coefficients.shape).astype(np.bool_)
    mask = np.invert(mask)
    zeros_matrix = np.zeros(coefficients.shape)
    coefficients[mask] = zeros_matrix[mask]
    dataset = pd.DataFrame(coefficients, columns=list("ABCDEF"))

    # Let's filter out degenerate conic sections using matrix representation of conic sections.
    dataset["determinant"] = dataset.eval(
        "A*C*F + B*E*D*.25 - D*D*C*.25 - B*B*F*.25 - E*E*A*.25"
    )
    # dataset = dataset.loc[dataset.determinant != 0, :]
    dataset.drop(dataset[dataset.determinant == 0].index, inplace=True)
    print(dataset.shape)

    dataset["discriminant"] = dataset.B**2 - 4 * dataset.A * dataset.C
    dataset["conicSection"] = np.select(
        [dataset.discriminant < 0, dataset.discriminant > 0],
        ["Ellipse", "Hyperbola"],
        default="Parabola",
    )
    print(dataset.conicSection.value_counts())

    # Let's forget about the analytical solution. Later we are going to use ML technics.
    dataset.drop(columns=["discriminant", "determinant"], inplace=True)

    x = dataset.drop(columns="conicSection")
    y = dataset.conicSection

    return train_test_split(x, y, test_size=0.2, random_state=222)

def make_predictions_classic_approach(
    x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series
) -> np.ndarray:
    poly = PolynomialFeatures(interaction_only=False, include_bias=False)
    x_train = poly.fit_transform(x_train)

    # Let's target class imbalance problem.
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train), y=y_train
    )
    class_weights_dict = {
        np.unique(y_train)[i]: w for i, w in enumerate(class_weights)}
    print(class_weights_dict)

    clf = LogisticRegression(
        penalty="l2",
        random_state=222,
        solver="newton-cg",
        C=999999,
        class_weight=class_weights_dict,
        multi_class="multinomial",
    )
    clf.fit(x_train, y_train)

    x_test = poly.fit_transform(x_test)
    y_pred = clf.predict(x_test)
    
    print(
        f"Accuracy of logistic regression classifier on test set: {round(clf.score(x_test, y_test), 4)}"
    )

    return y_pred

def make_predictions_automl_approach(
    x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series
) -> pd.Series:
    train_data = x_train.join(y_train)
    clf = TabularPredictor(label="conicSection")
    clf.fit(
        train_data=train_data
        #, ag_args_fit={'num_gpus': 1} # start docker with GPU
    )

    test_data = x_test.join(y_test)
    clf.leaderboard(test_data)
    y_pred = clf.predict(test_data)
    print(
        f'Accuracy of autoML classifier on test set: {round(clf.evaluate_predictions(y_test, y_pred)["accuracy"], 4)}'
    )

    return y_pred

@click.command()
@click.option('--approach', type=click.Choice(['classic', 'automl']), default='classic', help='Choose the prediction approach.')    
def main(approach: str):
    print("Generating data...")
    x_train, x_test, y_train, y_test = generate_data()

    print("Saving training and test data to CSV files...")
    train = pd.concat([x_train, y_train], axis=1)
    train.to_csv("train.csv", index=False)
    x_test.to_csv("test.csv")

    
    if approach == 'classic':
        print("Making predictions using classic approach...")
        y_pred = make_predictions_classic_approach(x_train, x_test, y_train, y_test)
    elif approach == 'automl':
        print("Making predictions using AutoML approach...")
        y_pred = make_predictions_automl_approach(x_train, x_test, y_train, y_test)
        
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("Saving test results to file...")
    test_results = pd.DataFrame(np.column_stack([x_test.index.values, y_pred]))
    test_results.columns = ["ID", "conicSection"]
    test_results.to_csv("test_results.txt", index=False)

if __name__ == "__main__":
    main()
