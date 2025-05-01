import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_blobs
from tensorflow.keras import layers, Model # type: ignore


# Генерация простых кластеров
np.random.seed(42)
n_clusters = 4 # Количество кластеров

data, _ = make_blobs(
    n_samples=800,
    centers=n_clusters,
    cluster_std=0.8, # Разброс точек внутри кластера
    random_state=42
)

plt.figure(figsize=(18, 5))
plt.subplot(131)
plt.scatter(data[:, 0], data[:, 1], s=10)
plt.title("Исходные данные")
plt.grid(True)

"""
Применим алгоритм KMeans++, суть которого в зависимости инициализации кластеров 
от расстояния между между ними (чем дальше кластер от предыдущего, тем больше
вероятность, что он будет следующим кластером)
"""

plt.subplot(132)
kmeans = KMeans(
    n_clusters=4,
    init='k-means++',
    n_init=10,
    random_state=42
)

kmeans_labels = kmeans.fit_predict(data)
plt.scatter(data[:, 0], data[:, 1], c=kmeans_labels, s=10, cmap='tab10')
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    c='red', marker='X', s=200,
    label='Центроиды'
)

plt.title("KMeans++")
plt.legend()
plt.grid(True)

"""
Применим алгоритм DBSCAN, чья кластеризация основана на плотности объектов
(группирует точки, тесно расположенные друг к другу). Имеет радиус, внутри
которого должны быть точки и минимальное количество точек в этом радиусе.
Все, что вне этого радиуса, классифицируется как шум.
"""
plt.subplot(133)
dbscan = DBSCAN(eps=0.8, min_samples=10)
dbscan_labels = dbscan.fit_predict(data)
plt.scatter(data[:, 0], data[:, 1], c=dbscan_labels, s=10, cmap='tab10')
plt.title("DBSCAN")
plt.grid(True)

plt.tight_layout()
plt.show()

"""
Автоэнкодер. Алгоритм сжатия данных до латентных (с наиболее четкими признаками
для разделения на кластеры) с помощью энкодера и их реконструкции декодером.
Это сверточная нейронная сеть (со сверточной и разверточной частами).
"""

data, _ = make_blobs(n_samples=1000, centers=4, cluster_std=1.2, random_state=42)

input_dim = data.shape[1]
encoding_dim = 2

input_layer = layers.Input(shape=(input_dim,))

encoder = layers.Dense(128, activation='relu')(input_layer)
encoder = layers.Dense(64, activation='relu')(encoder)
encoder = layers.Dense(encoding_dim, activation='linear')(encoder)

decoder = layers.Dense(64, activation='relu')(encoder)
decoder = layers.Dense(128, activation='relu')(decoder)
decoder = layers.Dense(input_dim, activation='linear')(decoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
encoder_model = Model(inputs=input_layer, outputs=encoder)

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(data, data, epochs=50, batch_size=32, verbose=0)

latent_vectors = encoder_model.predict(data)

# KMeans++
kmeans = KMeans(n_clusters=4, random_state=42)
clusters_km = kmeans.fit_predict(latent_vectors)

plt.figure(figsize=(18, 5))
plt.suptitle("KMeans++")

plt.subplot(131)
plt.scatter(data[:, 0], data[:, 1], s=10)
plt.title("Исходные данные")
plt.grid(True)

plt.subplot(132)
plt.scatter(data[:, 0], data[:, 1], c=clusters_km, cmap='tab10', s=10)
plt.title("Кластеризация через автоэнкодер")
plt.grid(True)

plt.subplot(133)
plt.scatter(latent_vectors[:, 0], latent_vectors[:, 1], c=clusters_km, cmap='tab10', s=10)
plt.title("Латентное пространство автоэнкодера")
plt.grid(True)

plt.tight_layout()
plt.show()

# DBSCAN
dbscan = DBSCAN(eps=1, min_samples=10)
clusters_db = dbscan.fit_predict(latent_vectors)

plt.figure(figsize=(18, 5))
plt.suptitle("DBSCAN")

plt.subplot(131)
plt.scatter(data[:, 0], data[:, 1], s=10)
plt.title("Исходные данные")
plt.grid(True)

plt.subplot(132)
plt.scatter(data[:, 0], data[:, 1], c=clusters_db, cmap='tab10', s=10)
plt.title("Кластеризация через автоэнкодер")
plt.grid(True)

plt.subplot(133)
plt.scatter(latent_vectors[:, 0], latent_vectors[:, 1], c=clusters_db, cmap='tab10', s=10)
plt.title("Латентное пространство автоэнкодера")
plt.grid(True)

plt.tight_layout()
plt.show()