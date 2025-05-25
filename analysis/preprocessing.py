import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Tuple, Optional, Union, BinaryIO

def load_dataset(source: Union[str, BinaryIO]) -> pd.DataFrame:
    """
    Загрузка набора данных из локального пути или объекта файла.
    Поддерживаются только файлы формата CSV.
    """
    try:
        df = pd.read_csv(source)
        return df
    except Exception as e:
        raise ValueError(f"Ошибка при загрузке набора данных: {e}")

def clean_data(df: pd.DataFrame, exclude_columns: Optional[list] = None) -> pd.DataFrame:
    """
    Очистка данных:
    - Удаление указанных столбцов (например, 'ID', 'Name')
    - Удаление нечисловых признаков
    - Удаление строк с пропущенными значениями (NaN)
    """
    if exclude_columns:
        df = df.drop(columns=exclude_columns, errors='ignore')
    
    df_numeric = df.select_dtypes(include=[np.number])
    df_clean = df_numeric.dropna()

    if df_clean.empty:
        raise ValueError("Набор данных пуст после удаления NaN или нечисловых столбцов.")
    
    return df_clean

def scale_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Масштабирование признаков с помощью StandardScaler.
    Возвращает:
    - масштабированный DataFrame
    - объект scaler для возможного сохранения или обратного преобразования
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(scaled, columns=df.columns)
    return df_scaled, scaler

def apply_pca(df: pd.DataFrame, n_components: int = 2) -> Tuple[pd.DataFrame, PCA, np.ndarray]:
    """
    Применение анализа главных компонент (PCA) к масштабированным данным.
    Возвращает:
    - DataFrame с пониженными измерениями (например, PC1, PC2)
    - объект PCA (можно использовать для графиков или восстановления)
    - массив с долей объяснённой дисперсии по компонентам
    """
    if n_components >= df.shape[1]:
        raise ValueError("Число компонент PCA должно быть меньше числа признаков.")

    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(df)
    df_reduced = pd.DataFrame(reduced, columns=[f"PC{i+1}" for i in range(n_components)])
    variance_ratio = pca.explained_variance_ratio_

    return df_reduced, pca, variance_ratio
