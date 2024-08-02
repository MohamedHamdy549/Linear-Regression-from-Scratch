import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


def y_bar(a, b, x):
    return a + b * x


def loss(a, b, x, y, m):
    return np.sum(np.square(y_bar(a, b, x) - y)) / (2 * m)


def grd(a, b, x, y, m):
    Error = y_bar(a, b, x)
    da = np.sum(Error - y) / m
    db = np.sum((Error - y) * x) / m
    return da, db


def update(a, b, da, db, LR):
    a = a - LR * da
    b = b - LR * db
    return a, b


def main():
    data = np.array(pd.read_csv('data/Salary.csv'))

    X = data[:, 0]
    y = data[:, -1]

    plt.scatter(X, y, marker='v', color='b')
    plt.xlabel('Years Experience')
    plt.ylabel('Salary')
    plt.title('YearsExperience vs Salary')
    plt.show()

    a = b = 0
    m = len(X)
    iter = 1000
    LR = .01

    loss_vector = []
    all_theta_0 = []
    all_theta_1 = []

    for i in range(iter):
        all_theta_0.append(a)
        all_theta_1.append(b)

        print(f'\n******** iter : {i + 1} ********\n')

        y_pred = y_bar(a, b, X)
        print(f'predict : {y_pred}')

        Error_vector = y_pred - y
        print(f'Error vector : {Error_vector}')

        cost = loss(a, b, X, y, m)
        loss_vector.append(cost)
        print('J(theta_0, theta_1) : ', cost)

        da, db = grd(a, b, X, y, m)
        grd_vector = np.array([da, db])
        grd_norm = np.linalg.norm(grd_vector)
        print('Gradiant Vector : ', grd_vector)
        print('Gradiant Norm : ', grd_norm)

        # Stop
        if grd_norm <= .01:
            break
        elif i > 0 and np.absolute(loss_vector[i - 1] - loss_vector[i]) < .01:
            break

        a, b = update(a, b, da, db, LR)
        print(f'New theta 0 : {a}\nNew theta 1 : {b}')

    # evaluate
    R = r2_score(y, y_pred)
    print('r sqr : ', R)

    plt.xlabel('theta_0', fontsize=14, fontweight='bold')
    plt.ylabel('Loss', fontsize=14, fontweight='bold')
    plt.plot(all_theta_0, loss_vector, '-mo', markersize=8)
    plt.show()

    plt.xlabel('theta_1', fontsize=14, fontweight='bold')
    plt.ylabel('Loss', fontsize=14, fontweight='bold')
    plt.plot(all_theta_1, loss_vector, '-mo', markersize=8)
    plt.show()

    plt.scatter(X, y, color='blue', label='Actual data')
    plt.scatter(X, y_pred, color='black', marker='x')
    plt.plot(X, y_pred, color='red', label='Regression line')
    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.title('Linear Regression')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
