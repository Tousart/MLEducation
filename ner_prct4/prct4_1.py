import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
X = 2*np.random.rand(100, 1)
y = 4 + 3*X + np.random.rand(100, 1)

plt.scatter(X, y)
plt.xlabel("X")
plt.ylabel("y")
plt.title("Сгенерированные данные")
plt.show()

# Функция градиентного спуска
def gradient_descent(X, y, lr=0.1, epochs=100):
    m = len(X)
    w = np.random.rand(1)
    b = np.random.rand(1)

    losses = []

    for epoch in range(epochs):
        y_pred = w*X+b
        loss = (1/m) * np.sum((y-y_pred) ** 2)
        losses.append(loss)
        dw = (-2/m) * np.sum(X * (y-y_pred))
        db = (-2/m) * np.sum(y - y_pred)
        w -= lr*dw
        b -= lr*db
        if epoch % 10 == 0:
            print(f"Эпoxa {epoch}: Incred = {loss:.4f}, w = {w[0]:.4f}, b = {b[0]:.4f}")
    return w, b, losses


# Запуск градиентного спуска
w_opt, b_opt, loss_history = gradient_descent(X, y, lr=0.1, epochs=100)

print("\nОптимизированные параметры:")
print(f"w= {w_opt[0] :.4f}, b = {b_opt[0] :.4f}")


plt.plot(loss_history)
plt.xlabel("Эпoxa")
plt.ylabel("Значение функции потерь")
plt.title("Динамика изменения функции потерь")
plt.show()

plt.scatter(X, y, label="Данные")
plt.plot(X, w_opt * X + b_opt, color="red", label="Линейная модель")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Линейная регрессия с градиентным спуском")
plt.legend()
plt.show()