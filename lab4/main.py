import numpy as np
from tabulate import tabulate


class Solution:
    __m = 6
    __n = 8
    __A = np.array([])
    __b = np.array([])
    __x = np.array([])
    __c = np.array([])
    __support_matrix = 0

    def __init__(self):
        np.random.seed(16)
        self.__c = np.array(np.random.randint(100, size=self.__n))
        self.__b = np.array(np.random.randint(100, size=self.__m))
        self.__A = np.array(np.random.randint(-100, 100, size=(self.__m, self.__n)))

        print("c:\n", self.__c)
        print("b:\n", self.__b)
        print("A:\n", self.__A, "\n")

    def execute(self):
        tmp = []
        for i in range(self.__A[0].size):
            tmp.append(max(self.__A[0:, i]))
        print("Верхняя цена игры:", min(tmp))

        tmp.clear()

        for i in range(self.__A[0, :6].size):
            tmp.append(min(self.__A[i, 0:]))
        print("Нижняя цена игры:", max(tmp))

        beta = min(tmp)
        print("beta =", beta)
        self.__A[0:] += abs(beta)
        print("NEW A:\n", self.__A)


if __name__ == "__main__":
    Solution().execute()
