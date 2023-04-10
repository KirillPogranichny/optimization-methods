import numpy as np
import matplotlib.pyplot as plt

# Проверяем, что все собственные значения матрицы положительны
def is_pos_def(A):
    return np.all(np.linalg.eigvals(A) > 0)


# Умножаем обратную матрицу на вектор b, чтобы получить точное значение
def exact_solution(A, b):
    return -np.linalg.inv(A).dot(b)


# Находим значение функции
def function(x):
    return .5 * np.matmul(np.matmul(x.T, A), x) + np.matmul(b.T, x)


# Находим первую производную функции
def derivative(x):
    return np.matmul(.5 * (np.add(A.T, A)), x) + b


# Находим корень суммы квадратов элементов вектора x
def error(x):
    vect_sum = 0
    for elem in x:
        vect_sum += np.power(elem, 2)

    return np.sqrt(vect_sum)


# Используем метод градиентного спуска
class GradientMethod:
    def __init__(self, A: np.array, x_0: np.array, steps: int, eps: float):
        self.ctr = 1
        self.x_k = x_0
        self.x_k1 = self.x_k - 1e-4 * derivative(self.x_k)

        self.func_sols = []

        if steps == 0 and is_pos_def(A):
            print("Матрица положительно определена")
            print("A:\n", A)
            print(f"lambda = {np.linalg.eigvals(A)}")
            # Пока разность векторов x_k+1 и x_k больше эпсилон, будем спускаться
            while error(self.x_k1 - self.x_k) > eps:
                self.ctr += 1
                self.x_k = self.x_k1
                self.x_k1 = self.x_k - 1e-4 * derivative(self.x_k)
                self.func_sols.append(function(self.x_k))
            print(f"Погрешности метода: {abs(self.x_k - exact_solution(A, b))}")
            print(f"X_m: {self.x_k}")
            print(f"X_точ: {exact_solution(A, b)}")

        elif is_pos_def(A):
            while self.ctr < steps:
                self.ctr += 1
                self.x_k = self.x_k1
                self.x_k1 = self.x_k - 1e-4 * derivative(self.x_k)
                self.func_sols.append(function(self.x_k))

        else:
            print("Матрица не определена положительно")

        print(f"\nТочка минимума: {self.x_k1}, значение функции: {function(self.x_k)}")
        print(f"\nПогрешность функции в x: {abs(function(self.x_k) - function(exact_solution(A, b)))}")
        print(f"Потребовалось итераций: {self.ctr}, точность: {error(self.x_k1 - self.x_k)}")


    def display(self):
        return self.func_sols, self.ctr, self.x_k1


if __name__ == "__main__":
    A = np.array([[2., 3.5, 4.5, 6., 7., 2.],
                  [0., 6., 1.5, 5., 3., 0.5],
                  [6., 6., 0., 9.5, 5., 3.5],
                  [5., 2., 6., 9., 3., 3.],
                  [9., 6.5, 6., 2., 3., 1.],
                  [8., 0.5, 2., 7., 6., 6.]])
    A = np.matmul(A, A.T)

    b = np.array([1., 0.5, 0.5, 1.5, 1.5, 2.])
    x_0 = np.array([1.5, 0.5, 2.5, 0.5, 2., 1.5])

    eps = 1e-7

    func_solves, steps, x = GradientMethod(A, x_0, 0, eps).display()
    plt.plot(func_solves)
    print(f"Результат для 1/4 итераций: {GradientMethod(A, x_0, round(steps / 4), eps).display()[2]}\n")
    print(f"Результат для 1/2 итераций: {GradientMethod(A, x_0, round(steps / 2), eps).display()[2]}\n")
    print(f"Результат для 3/4 итераций: {GradientMethod(A, x_0, round(steps * 3 / 4), eps).display()[2]}\n")


    print(f"Точное решение: {exact_solution(A, b)}")
    print(f"Значение функции в точке x*: {function(exact_solution(A, b))}")

    plt.grid()
    plt.savefig('solution.png', bbox_inches='tight')
    plt.show()