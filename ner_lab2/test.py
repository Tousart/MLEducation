import matplotlib.pyplot as plt
"""
import tensorflow - фреймворк для построения нейросетей
from tensorflow import keras - высокоуровневый API для работы с моделями
"""
from tensorflow.keras.models import Sequential # type: ignore # Последовательная модель нейросети
from tensorflow.keras.layers import Dense # type: ignore # Полносвязные слои нейросети
from tensorflow.keras.layers import Input # type: ignore # Для входного слоя
from tensorflow.keras.optimizers import Adam # type: ignore # для скорости обучения
from sklearn.model_selection import train_test_split # Разделение данных нa train/test
from sklearn.preprocessing import StandardScaler # Масштабирование данных
from sklearn.metrics import mean_squared_error, r2_score # Метрики оценки модели
from sklearn.datasets import fetch_openml # Загрузка датасетов из OpenML
from sklearn.linear_model import LinearRegression # Линейная регрессия
from sklearn.preprocessing import PolynomialFeatures # Полиномиальная регрессия

# Загружаем датасет Boston Housing
boston = fetch_openml(name="boston", version=1, as_frame=True)
x = boston.data # 13 переменных, описывающих дома
y = boston.target # 1 переменная: стоимость дома

# 80 процентов обучающей выборки и 20 тестовой
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Масштабируем признаки [-1:1]
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

print("X train:", x_train.shape)
print("X test:", x_test.shape)

print(x.sample(5, random_state=42))
print(y.sample(5, random_state=42))

print()

# Модель линейной регрессии
lin_reg = LinearRegression()
lin_reg.fit(x_train_scaled, y_train)

# Предсказываем
y_pred_lin = lin_reg.predict(x_test_scaled)

# Оценка модели
mse_lin = mean_squared_error(y_test, y_pred_lin) # Среднеквадратичное отклонение
r2_lin = r2_score(y_test, y_pred_lin) # Коэффициент детерминации
"""
r2 показывает долю изменчивости данных модели (от 0 до 1).
То есть, насколько сильно меняются исходные данные y, 
и насколько хорошо модель учитывает эту изменчивость.
1 - (необъясненная изменчивость)/(общая изменчивость).
Необъясненная - ошибки модели, общая - разброс y вокруг среднего y_mean.
Соответственно, чем меньше необъясненная, тем лучше модель предсказывает.
"""
print(f"Линейная регрессия: MSE - {mse_lin}, R2 - {r2_lin}")

# Модель полиномиальной регрессии (до 2-ой степени)
poly = PolynomialFeatures(degree=2)
x_train_poly = poly.fit_transform(x_train_scaled)
x_test_poly = poly.transform(x_test_scaled)

poly_model = LinearRegression()
poly_model.fit(x_train_poly, y_train)
y_test_pred = poly_model.predict(x_test_poly)

# Оценка модели
mse_poly = mean_squared_error(y_test, y_test_pred)
r2_poly = r2_score(y_test, y_test_pred)
print(f"Полиномиальная регрессия: MSE - {mse_poly}, R2 - {r2_poly}")

print()

# Создаем модель перцептрона
model = Sequential([
    Input(shape=(x_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

# Компилируем и обучаем модель
model.compile(optimizer=Adam(learning_rate=0.1), loss='mse', metrics=['mae'])
history = model.fit(x_train_scaled, y_train, epochs=50, batch_size=32, validation_data=(x_test_scaled, y_test), verbose=0)

# Предсказание и оценка перцептрона
y_pred_per = model.predict(x_test_scaled)
mse_per = mean_squared_error(y_test, y_pred_per)
r2_per = r2_score(y_test, y_pred_per)

print(f"Перцептрон: MSE - {mse_per}, R2 - {r2_per}")

# График изменения MSE во время обучения
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Эпохи')
plt.ylabel('Среднеквадратичная ошибка (MSE)')
plt.legend()
plt.grid()
plt.title('Процесс обучения нейронной сети')
plt.show()