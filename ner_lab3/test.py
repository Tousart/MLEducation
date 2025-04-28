import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore

# Загрузка данных
iris = load_iris()
X, y = iris.data, iris.target

# Разделяем на обучающую и тестовую выборку (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Масштабирование данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print ("Размеры данных: (цветки, признаки)")
print("X train:", X_train.shape)
print("X test: ", X_test.shape)

"""
Бинарная классификация – объект принадлежит одному из двух возможных классов.
"""

# Оставляем два класса для бинарной классификации
binary_mask = (y == 0) | (y == 1)
X_binary, y_binary = X[binary_mask], y[binary_mask]

# Разделим на выборки
X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(X_binary, y_binary, test_size=0.2, random_state=42)

# Масштабирование
scaler_bin = StandardScaler()
X_train_bin_scaled = scaler_bin.fit_transform(X_train_bin)
X_test_bin_scaled = scaler_bin.transform(X_test_bin)

# Обучение модели логистической регрессии
log_reg = LogisticRegression()
log_reg.fit(X_train_bin_scaled, y_train_bin)

# Предсказания
y_pred_bin = log_reg.predict(X_test_bin_scaled)

# Оценка модели
print("Точнсть (Accuracy):", accuracy_score(y_test_bin, y_pred_bin))
print("\nОтчет классификации:\n", classification_report(y_test_bin, y_pred_bin))
print("\nМатрица ошибок:\n", confusion_matrix(y_test_bin, y_pred_bin))

# Функция для визуализации границ классов
def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                        np.arange(x2_min, x2_max, resolution))

    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    color=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')

# Выбираем только два признака (длина и ширина лепестка)
X_binary_2D = X_binary[:, [2, 3]]

X_train_bin_2D, X_test_bin_2D, _, _ = train_test_split(X_binary_2D, y_binary, test_size=0.2, random_state=42)

scaler_2D = StandardScaler()
X_train_bin_2D_scaled = scaler_2D.fit_transform(X_train_bin_2D)
X_test_bin_2D_scaled = scaler_2D.transform(X_test_bin_2D)

# Создаем и обучаем модель логистической регрессии
log_reg_2D = LogisticRegression()
log_reg_2D.fit(X_train_bin_2D_scaled, y_train_bin)

plot_decision_regions(X_train_bin_2D_scaled, y_train_bin, classifier=log_reg_2D)
plt.xlabel('Длина лепестка (стандартиз.)')
plt.ylabel('Ширина лепестка (стандартиз.)')
plt.legend(loc='upper left')
plt.title("Визуализация бинарной классификации")
plt.show()

"""
Многоклассовая классификация – объект принадлежит одному из множества возможных классов.

Многоклассовая классификация с помощью Softmax
Эта функция возвращает вероятности для всех k классов, 
и объект относится к классу с наибольшей вероятностью.
"""

# Обучение многоклассовой логистической регрессии
log_reg_multi = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
log_reg_multi.fit(X_train_scaled, y_train)

y_pred_multi = log_reg_multi.predict(X_test_scaled)

print("Точность многоклассовой классификации:", accuracy_score(y_test, y_pred_multi))
print("\nОтчет классификации:\n", classification_report(y_test, y_pred_multi))
print("\nМатрица ошибок:\n", confusion_matrix(y_test, y_pred_multi))

# Два признака для визуализации
X_train_2D = X_train_scaled[:, [2, 3]]
X_test_2D = X_test_scaled[:, [2, 3]]

log_reg_2D_multi = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
log_reg_2D_multi.fit(X_train_2D, y_train)

plot_decision_regions(X_train_2D, y_train, classifier=log_reg_2D_multi)
plt.xlabel('Длина лепестка (стандартиз.)')
plt.ylabel('Ширина лепестка (стандартиз.)')
plt.legend(loc='upper left')
plt.title("Многоклассовая логистическая регрессия")
plt.show()

"""
Многоклассовая классификация с помощью One-vs-Rest.
Строится несколько бинарных классификаторов, 
каждый из которых отличает один класс от всех остальных.
"""

log_reg_ovr = LogisticRegression(multi_class='ovr', solver='lbfgs', max_iter=200)
log_reg_ovr.fit(X_train_scaled, y_train)

y_pred_ovr = log_reg_ovr.predict(X_test_scaled)

print("Точность многоклассовой логистической регрессии (One-vs-Rest):", accuracy_score(y_test, y_pred_ovr))
print("\nОтчет классификации:\n", classification_report(y_test, y_pred_ovr))
print("\nМатрица ошибок:\n", confusion_matrix(y_test, y_pred_ovr))

"""
Классификация с помощью нейронной сети (перцептрон)
"""

X_train_2D = X_train_scaled[:, [2, 3]]
X_test_2D = X_test_scaled[:, [2, 3]]

model_2D = Sequential([
    Dense(32, activation='relu', input_shape=(X_train_2D.shape[1],)),
    Dense(16, activation='relu'),
    Dense(3, activation='softmax')
])

model_2D.compile(optimizer='adam', 
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])

model_2D.fit(X_train_2D, y_train, 
             epochs=50, 
             batch_size=8,
             validation_data=(X_test_2D, y_test),
             verbose=0)

loss, accuracy = model_2D.evaluate(X_test_2D, y_test)
print(f"Точность нейронной сети: {accuracy:.3f}")

def plot_decision_regions_NN(X, y, model, resolution=0.02):
    markers = ('s', 'x', 'o')
    colors = ('red', 'blue', 'lightgreen')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    Z = model.predict(np.c_[xx1.ravel(), xx2.ravel()])
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    color=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')


plot_decision_regions_NN(X_train_2D, y_train, model_2D)
plt.xlabel('Длина лепестка')
plt.ylabel('Ширина лепестка')
plt.legend(loc='upper left')
plt.title("Классификация для трех классов")
plt.show()