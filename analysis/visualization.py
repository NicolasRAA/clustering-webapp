import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from scipy.cluster.hierarchy import dendrogram, linkage


def plot_raw_distribution(df: pd.DataFrame, title: str = "Исходное распределение данных") -> plt.Figure:
    if df.shape[1] < 2:
        raise ValueError("Необходимо минимум 2 признака для визуализации.")
    
    fig, ax = plt.subplots()
    sns.scatterplot(x=df.iloc[:, 0], y=df.iloc[:, 1], ax=ax, s=60, color='gray')
    ax.set_title(title)
    ax.set_xlabel(df.columns[0])
    ax.set_ylabel(df.columns[1])
    return fig


def plot_kmeans_clusters(df: pd.DataFrame, labels: np.ndarray, model: KMeans, title: str = "Кластеры KMeans") -> plt.Figure:
    if df.shape[1] < 2:
        raise ValueError("Необходимо минимум 2 признака для визуализации.")

    fig, ax = plt.subplots()
    sns.scatterplot(x=df.iloc[:, 0], y=df.iloc[:, 1], hue=labels, palette='Set2', ax=ax, s=60)
    
    centers = model.cluster_centers_
    ax.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.6, marker='X', label='Центроиды')

    ax.set_title(title)
    ax.set_xlabel(df.columns[0])
    ax.set_ylabel(df.columns[1])
    ax.legend()
    return fig


def plot_hierarchical_clusters(df: pd.DataFrame, labels: np.ndarray, title: str = "Кластеры (иерархическая кластеризация)") -> plt.Figure:
    fig, ax = plt.subplots()
    sns.scatterplot(x=df.iloc[:, 0], y=df.iloc[:, 1], hue=labels, palette='Set2', ax=ax, s=60)
    ax.set_title(title)
    ax.set_xlabel(df.columns[0])
    ax.set_ylabel(df.columns[1])
    ax.legend(title="Кластер")
    return fig


def plot_dendrogram(df: pd.DataFrame, method: str = 'average') -> plt.Figure:
    Z = linkage(df, method=method)
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(Z, ax=ax)
    ax.set_title(f"Дендрограмма (метод: {method})")
    ax.set_xlabel("Индекс наблюдения")
    ax.set_ylabel("Расстояние")
    plt.xticks([])  # Убираем метки с оси X
    return fig


def plot_elbow(df: pd.DataFrame, max_k: int = 10) -> plt.Figure:
    inertias = []
    ks = range(1, max_k + 1)

    for k in ks:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df)
        inertias.append(kmeans.inertia_)

    fig, ax = plt.subplots()
    ax.plot(ks, inertias, marker='o')
    ax.set_xlabel("Количество кластеров (k)")
    ax.set_ylabel("Инерция")
    ax.set_title("Метод локтя (Elbow Method)")
    return fig


def plot_silhouette(df: pd.DataFrame, labels: np.ndarray) -> plt.Figure:
    silhouette_vals = silhouette_samples(df, labels)
    n_clusters = len(np.unique(labels))
    y_lower = 10

    fig, ax = plt.subplots()
    for i in range(n_clusters):
        ith_cluster_silhouette_vals = silhouette_vals[labels == i]
        ith_cluster_silhouette_vals.sort()
        size_cluster_i = ith_cluster_silhouette_vals.shape[0]
        y_upper = y_lower + size_cluster_i
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_cluster_silhouette_vals)
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax.set_title("Силуэт-анализ кластеров")
    ax.set_xlabel("Значение силуэта")
    ax.set_ylabel("Кластер")
    ax.axvline(x=np.mean(silhouette_vals), color="red", linestyle="--")
    return fig


def plot_true_labels(df: pd.DataFrame, true_labels: np.ndarray, title="Реальные классы") -> plt.Figure:
    fig, ax = plt.subplots()
    sns.scatterplot(x=df.iloc[:, 0], y=df.iloc[:, 1], hue=true_labels, palette='tab10', ax=ax, s=60)
    ax.set_title(title)
    ax.set_xlabel(df.columns[0])
    ax.set_ylabel(df.columns[1])
    ax.legend(title="Класс")
    return fig