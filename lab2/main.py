import random

import numpy as np


def initial_A(dim):
    A = np.tril(np.random.randint(-10, 10, (dim, dim)))
    A += A.T

    if np.linalg.det(A) != 0 and np.shape(A)[0] == np.shape(A)[1]:
        print("Матрица A симметрична и невырождена:")
        return A
    else:
        print("Матрица A вырождена, создадим новую.")
        initial_A(dim)


def initial_b(dim):
    b = (10 - (-10)) * np.random.random_sample(dim) - 10
    ctr = 0

    for i in range(dim):
        if b[i] != 0:
            continue
        else:
            ctr += 1
            if ctr == dim:
                print("Вектор b является нулевым, создадим новый.")
                initial_b(dim)

    print("\nВектор b ненулевой:")

    return b


def initial_x0(dim):
    x0 = (10 - (-10)) * np.random.random_sample(dim) - 10
    ctr = 0

    for i in range(dim):
        if x0[i] != 0:
            continue
        else:
            ctr += 1
            if ctr == dim:
                print("Вектор x0 является нулевым, создадим новый.")
                initial_b(dim)

    print("\nВектор x0 ненулевой:")

    return x0


def solve_x(A, b, dim, y):
    return np.linalg.tensorsolve(np.add(A, ((2 * y * np.identity(dim)))), -(np.add(b, (2 * y * x0))))


# def error(x, y, r):
#     vect_sum = 0
#     for elem in x:
#         vect_sum += np.power(elem, 2)
#
#     if y == 0 and np.sqrt(vect_sum) <= r:
#         return np.sqrt(vect_sum)
#     elif y > 0 and vect_sum == np.power(r, 2):
#         return vect_sum
#
#
# def lagrange(A, b, x0, r):
#     return .5 *


if __name__ == "__main__":
    rand_seed = 10
    np.random.seed(rand_seed)
    random.seed(rand_seed)

    dim = 4
    A = np.array(initial_A(dim), float)
    print(A)
    b = initial_b(dim)
    print(b)
    x0 = initial_x0(dim)
    print(x0)
    r = random.randint(1, 20)
    print("\nРадиус сферы =", r)
    y = 0