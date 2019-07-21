import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.utils import compute_class_weight
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import confusion_matrix


def generate_data():
    np.random.seed(222)
    DATASET_SIZE = 50123

    coefficients = np.random.uniform(-1, 1, size=(DATASET_SIZE, 6))
    # Let's add some zero values in order to get fewer ellipses and hyperbolas.
    mask = np.random.randint(0, 6, size = coefficients.shape).astype(np.bool)
    mask = np.invert(mask)
    zeros_matrix = np.zeros(coefficients.shape)
    coefficients[mask] = zeros_matrix[mask]
    dataset = pd.DataFrame(coefficients, columns=list('ABCDEF'))
    
    # Let's filter out degenerate conic sections using matrix representation of conic sections.
    dataset['determinant'] = dataset.eval('A*C*F + B*E*D*.25 - D*D*C*.25 - B*B*F*.25 - E*E*A*.25')
    #dataset = dataset.loc[dataset.determinant != 0, :]
    dataset.drop(dataset[dataset.determinant == 0].index, inplace = True)
    print(dataset.shape)
    
    dataset['discriminant'] = dataset.B ** 2 - 4 * dataset.A * dataset.C
    dataset['conicSection'] = np.select(
        [
            dataset.discriminant < 0,
            dataset.discriminant > 0
        ],
        [
            'Ellipse',
            'Hyperbola'
        ], default = 'Parabola')
    print(dataset.conicSection.value_counts())
    
    # Let's try to forget about the analytical solution. Later we are going to use standard ML technics.
    dataset.drop(columns = ['discriminant', 'determinant'], inplace = True)
    
    X = dataset.drop(columns = 'conicSection')
    y = dataset.conicSection
    
    return(train_test_split(X, y, test_size = 0.2, random_state = 222))    

def make_predictions(x_train, x_test, y_train, y_test):
    poly = PolynomialFeatures(interaction_only = False,include_bias = False)
    x_train = poly.fit_transform(x_train)
    
    # Let's target class imbalance problem.
    class_weights = compute_class_weight('balanced', np.unique(y_train), y_train)
    class_weights_dict = {np.unique(y_train)[i]: w for i, w in enumerate(class_weights)}
    print(class_weights_dict)
    
    clf = LogisticRegression(penalty = 'l2', random_state = 222, solver = 'newton-cg', C = 999999,
                            class_weight = class_weights_dict,
                            multi_class = 'multinomial').fit(x_train, y_train)
 
    IDs = x_test.index
    x_test = poly.fit_transform(x_test)
    y_pred = clf.predict(x_test)
    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(clf.score(x_test, y_test)))
    confusion_matrix_ = confusion_matrix(y_test, y_pred)
    print('Confusion matrix:')
    print(confusion_matrix_)
    
    testres = pd.DataFrame(np.column_stack([IDs.values, y_pred]))
    testres.columns = ['ID', 'conicSection']
    return(testres)

def main():
    print(os.getcwd())
    print(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    x_train, x_test, y_train, y_test = generate_data()
    
    train = pd.concat([x_train, y_train], axis = 1)
    train.to_csv("train.csv", index = False)
    x_test.to_csv("test.csv")
    print(os.listdir(os.curdir))    

    y_pred = make_predictions(x_train, x_test, y_train, y_test)
    y_pred.to_csv("testres.txt", index = False)
    
if __name__ == '__main__':
    main()

