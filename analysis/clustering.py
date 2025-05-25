import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from typing import Tuple, Optional, List

def run_kmeans(df: pd.DataFrame, n_clusters: int = 3, random_state: int = 42) -> Tuple[np.ndarray, KMeans]:
    """
    Выполняет кластеризацию методом k-средних (k-means).
    Возвращает:
    - метки кластеров для каждой строки
    - обученную модель KMeans
    """
    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = model.fit_predict(df)
    return labels, model

def run_hierarchical(df: pd.DataFrame, n_clusters: int = 3, linkage_method: str = 'ward') -> Tuple[np.ndarray, AgglomerativeClustering]:
    """
    Выполняет иерархическую агломеративную кластеризацию.
    Возвращает:
    - метки кластеров
    - обученную модель AgglomerativeClustering
    Поддерживаемые методы связи: 'ward', 'complete', 'average', 'single'
    """
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    labels = model.fit_predict(df)
    return labels, model
