import numpy as np
import matplotlib.pyplot as plt

# Функции активации
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)


def tanh(z):
    return np.tanh(z)

def tanh_derivative(z):
    return 1 - np.tanh(z) ** 2


# Перцептрон: 1 входной слой, 2 скрытых, 1 выходной
class MLP:
    def __init__(self, n_input, n_hidden1, n_hidden2, n_output, learning_rate=0.05):
        self.learning_rate = learning_rate

        self.w1 = np.random.randn(n_input, n_hidden1) * 0.01
        self.b1 = np.zeros((1, n_hidden1))

        self.w2 = np.random.randn(n_hidden1, n_hidden2) * 0.01
        self.b2 = np.zeros((1, n_hidden2))

        self.w3 = np.random.randn(n_hidden2, n_output) * 0.01
        self.b3 = np.zeros((1, n_output))
    
    # Прямое распростронение входных данных (преобразование по слоям)
    def forward(self, X):
        # Первый скрытый слой
        self.z1 = np.dot(X, self.w1) + self.b1
        self.a1 = relu(self.z1)
        # Второй скрытый слой
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = tanh(self.z2)
        # Выходной слой
        self.z3 = np.dot(self.a2, self.w3) + self.b3
        self.a3 = sigmoid(self.z3)
        return self.a3

    # Функция потерь
    def compute_loss(self, y, output):
        # Бинарная кросс-энтропия
        m = y.shape[0]
        loss = -(1 / m) * np.sum(y * np.log(output + 1e-8) + (1 - y) * np.log(1 - output + 1e-8))
        return loss
    
    # Обратное распространение ошибки для вычисления градиентов
    def backward(self, X, y):
        m = X.shape[0]
        # Градиенты для выходного слоя
        dz3 = self.a3 - y # Производная для кросс-энтропии со сигмоидой
        dw3 = np.dot(self.a2.T, dz3) / m
        db3 = np.sum(dz3, axis=0, keepdims=True) / m

        # Градиенты для второго скрытого слоя
        da2 = np.dot(dz3, self.w3.T)
        dz2 = da2 * tanh_derivative(self.z2)
        dw2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # Градиенты для первого скрытого слоя
        da1 = np.dot(dz2, self.w2.T)
        dz1 = da1 * relu_derivative(self.z1)
        dw1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        grads = {"dw3": dw3, "db3": db3,
        "dw2": dw2, "db2": db2,
        "dw1": dw1, "db1": db1}
        return grads
    
    # Обновление весов и смещений с помощью градиентного спуска
    def update_parameters(self, grads):
        self.w1 -= self.learning_rate * grads["dw1"]
        self.b1 -= self.learning_rate * grads["db1"]
        self.w2 -= self.learning_rate * grads["dw2"]
        self.b2 -= self.learning_rate * grads["db2"]
        self.w3 -= self.learning_rate * grads["dw3"]
        self.b3 -= self.learning_rate * grads["db3"]

    # Функция тренировки
    def train(self, X, y, epochs=1000):
        loss_history = []
        for i in range(epochs):
            output = self.forward(X) # Прямое распространение
            loss = self.compute_loss(y, output) # Вычисление функции потерь
            loss_history.append(loss)
            grads = self.backward(X, y) # Обратное распространение
            self.update_parameters(grads) # Обновление параметров

            if i % 100 == 0:
                print(f"Epoch {i}, Loss: {loss:.4f}")
        return loss_history

    # Функция предсказания
    def predict(self, X, threshold=0.5):
        output = self.forward(X)
        predictions = (output > threshold).astype(int)
        return predictions


# Генерация данных
def generate_data(n_samples=200):
    np.random.seed(42)
    # Класс 0: данные вокруг [-1, -1]
    X0 = np.random.randn(n_samples // 2, 2) + np.array([-1, -1])
    # Класс 1: данные вокруг [1, 1]
    X1 = np.random.randn(n_samples // 2, 2) + np.array([1, 1])
    X = np.vstack((X0, X1))
    y = np.vstack((np.zeros((n_samples // 2, 1)), np.ones((n_samples // 2, 1))))

    # Перемешивание данных
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    return X, y

# Генерация данных
X, y = generate_data(200)
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# Визуализация данных
plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap='bwr', alpha=0.7)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Синтетические данные для бинарной классификации")
plt.show()

# Создание экземпляра MLP
n_input = X.shape[1] # Количество признаков (2)
n_hidden1 = 12 # Нейроны в первом скрытом слое
n_hidden2 = 8 # Нейроны во втором скрытом слое
n_output = 1 # Выходной нейрон для бинарной классификации

mlp = MLP(n_input, n_hidden1, n_hidden2, n_output, learning_rate=0.05)

epochs = 1000
loss_history = mlp.train(X, y, epochs=epochs)

# Визуализация данных
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Динамика функции потерь")
plt.show()

# Оценка точности
predictions = mlp.predict(X)
accuracy = np.mean(predictions == y)
for i in range(len(y)):
    print(f"expected: {y[i]}   predicted: {predictions[i]}")
print("Точность:", accuracy)