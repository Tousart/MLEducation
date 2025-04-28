import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

class Perceptron:
    def __init__(self, lr = 0.1, n_epochs = 10):
        self.lr = lr
        self.n_epochs = n_epochs

    def activation(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        # Кол-во образцов и признаков
        n_samples, n_features = X.shape

        # Инициализируем веса и смещение
        self.weights = np.random.randn(n_features)
        self.bias = np.random.randn()

        # Проходим по датасету несколько эпох
        for epoch in range(self.n_epochs):
            for i in range(n_samples):
                # Шаг 1: вычисляем линейную комбинацию
                z = np.dot(X[i], self.weights) + self.bias

                # Шаг 2: применяем функцию активации
                y_pred = self.activation(z)

                # Шаг 3: вычисляем ошибку
                error = y[i] - y_pred

                # Шаг 4: обновляем веса (правило обучения перцептрона)
                self.weights += self.lr * error * X[i]
                self.bias += self.lr * error
        
    # Возвращает массив предсказаний (0 или 1) для каждого образца
    def predict(self, X):
        # X - входные данные
        z = np.dot(X, self.weights) + self.bias

        # Используем пороговую функцию по каждому элементу
        predictions = [[self.activation(val)] for val in z]
        return np.array(predictions)


# Читаем датасэт
df = pd.read_csv("train.csv")

# Убираем столбцы типа object
df.drop(columns=df.select_dtypes(include=['object']).columns, inplace=True)

# Хитмапа с корреляцией столбцов
corr = df.corr()
plt.figure(figsize=(30, 30))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)

# Удалим столбцы, которые коррелируют менее, чем на 0.5
coef = 0.45
del_col = []

for col in corr:
  if corr["SalePrice"][col] <= coef:
    del_col.append(col)

df.drop(columns=del_col, inplace=True)
print(df.head()) # теперь у нас отобраны необходимые признаки для определения цены

# Считаем пустые значения и заполняем их средними значениями соответствующих столбцов
print(df.isnull().sum())
df.fillna(df.mean(), inplace=True)

print()

# Разбиваем данные на тренировочные и тестовые
X, y = df.drop(columns=["SalePrice"], inplace=False), pd.DataFrame(df["SalePrice"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# Масштабируем данные при помощи scaler
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# Масштабируем X_train и y_train, обучив scaler на тренировочных значениях
X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train)

# Масштабируем тестовые данные X_test
X_test_scaled = scaler_X.transform(X_test)

# Обучаем перцептрон
perceptron = Perceptron()
perceptron.fit(X_train_scaled, y_train_scaled)

# Получаем массив предсказанных цен
y_predicted = scaler_y.inverse_transform(perceptron.predict(X_test_scaled))

# Выводим результат
result = pd.DataFrame(
    {
        "predicted": [i[0] for i in y_predicted],
        "excepted": y_test["SalePrice"]
    }
)
print(result.head())

print()

# Выводим среднию квадратичную ошибку
print("MSE:", mean_squared_error(y_predicted, y_test))

# Выводим абсолютную ошибку
print("MAE:", mean_absolute_error(y_predicted, y_test))

plt.show()