import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from typing import List, Tuple

def calculate_silhouette_score(df: pd.DataFrame, labels: np.ndarray) -> float:
    """
    Вычисляет коэффициент силуэта (silhouette score) для заданных меток кластеров.
    Значение ближе к 1 — лучше.
    """
    if len(set(labels)) < 2:
        return -1  # невозможно рассчитать силуэт, если только один кластер
    return silhouette_score(df, labels)

def calculate_inertia(model: KMeans) -> float:
    """
    Возвращает инерцию обученной модели KMeans.
    Инерция — сумма квадратов расстояний от точек до их центров кластеров.
    """
    return model.inertia_

def suggest_k_by_silhouette(df: pd.DataFrame, k_range: List[int]) -> Tuple[int, List[float]]:
    """
    Автоматически выбирает оптимальное число кластеров (k), используя коэффициент силуэта.
    Возвращает:
    - рекомендуемое значение k
    - список силуэтов для каждого k
    """
    best_score = -1
    best_k = k_range[0]
    silhouette_scores = []

    for k in k_range:
        if k >= len(df):
            break  # нельзя создать больше кластеров, чем точек
        try:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(df)
            score = silhouette_score(df, labels)
            silhouette_scores.append(score)
            if score > best_score:
                best_score = score
                best_k = k
        except Exception:
            silhouette_scores.append(-1)

    return best_k, silhouette_scores
