import random

import numpy as np


def initial_A():
    A = np.tril(np.random.randint(-10, 10, (dim, dim)))
    A += A.T

    if np.linalg.det(A) != 0 and np.shape(A)[0] == np.shape(A)[1]:
        print("Матрица A симметрична и невырождена:")
        return A
    else:
        print("Матрица A вырождена, создадим новую.")
        initial_A()


def initial_b():
    b = (10 - (-10)) * np.random.random_sample(dim) - 10
    ctr = 0

    for i in range(dim):
        if b[i] != 0:
            continue
        else:
            ctr += 1
            if ctr == dim:
                print("Вектор b является нулевым, создадим новый.")
                initial_b()

    print("\nВектор b ненулевой:")

    return b


def initial_x0():
    x0 = (10 - (-10)) * np.random.random_sample(dim) - 10
    ctr = 0

    for i in range(dim):
        if x0[i] != 0:
            continue
        else:
            ctr += 1
            if ctr == dim:
                print("Вектор x0 является нулевым, создадим новый.")
                initial_b()

    print("\nВектор x0 ненулевой:")

    return x0


def function(x):
    res = .5 * np.matmul(np.matmul(x.T, A), x) + np.matmul(b.T, x)
    return res[0][0]


def lagrange_slae(x):
    return np.append(np.matmul((A + 2 * np.identity(dim) * y), x) + (b + 2 * y * x0), [[np.linalg.norm(np.power((x - x0), 2)) - np.power(r, 2)]], axis=0)


def jacobian(x):
    J = np.empty([2, 2], dtype=float)
    J[0][0] = A + 2 * np.identity(dim) * y
    J[0][1] = 2 * (x - x0)
    J[1][0] = J[0][1].T
    J[1][1] = 0
    return J


def newton(x_k):
    max_iter = 30
    eps = 1e-6
    x_prev = x_k
    x_cur = x_prev - np.matmul(np.linalg.inv(jacobian(x_prev[0:-1])), lagrange_slae(x_prev[0:-1]))
    iter = 0

    while np.linalg.norm(x_cur[0:-1] - x_prev[0:-1]) > eps and iter < max_iter:
        iter += 1
        x_prev = x_cur
        x_cur = x_prev - np.matmul(np.linalg.inv(jacobian(x_prev[0:-1])), lagrange_slae(x_prev[0:-1]))

    return x_cur


if __name__ == "__main__":
    rand_seed = 10
    np.random.seed(rand_seed)
    random.seed(rand_seed)

    dim = 4
    A = np.array(initial_A(), float)
    print(A)
    b = initial_b()
    print(b)
    x0 = initial_x0()
    print(x0)
    r = random.randint(1, 20)
    print("\nРадиус сферы =", r)

    y = 2
    n = 3
    sign = 1
    x = np.append(x0, [[y]], axis=0)

    x_star = np.matmul(-np.linalg.inv(A), b)
    f_x_star = function(x_star)
    print(f"x* =\n{x_star}")
    print(f"Функция от x* = {f_x_star}")
    print(f"x*-x0 =\n{x_star - x0}")
    print(f"||x*-x0|| = {np.linalg.norm(x_star - x0)}")

    for i in range(8):
        sign = -sign
        x_k = x.copy()
        x_k[i//2][0] += sign * n
        print(f"\nДля {i+1}:\n{x_k[0:-1]}")
        res = newton(x_k)
        print(f"Результат по x = {res[0:-1]}")
        print(f"Результат по y = {res[4][0]}")
        print(f"Результат для функции = {res[0:-1]}")