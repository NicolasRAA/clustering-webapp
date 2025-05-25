import gradio as gr
import pandas as pd

from analysis.preprocessing import load_dataset, clean_data, scale_data, apply_pca
from analysis.clustering import run_kmeans, run_hierarchical
from analysis.evaluation import calculate_silhouette_score, calculate_inertia
from analysis.visualization import (
    plot_raw_distribution,
    plot_kmeans_clusters,
    plot_hierarchical_clusters,
    plot_dendrogram,
    plot_elbow,
    plot_silhouette,
    plot_true_labels
)

PREDEFINED_DATASETS = {
    "Wine": "datasets/wine.csv",
    "Wholesale": "datasets/wholesale.csv",
    "Mall Customers": "datasets/mall.csv"
}

def run_pipeline(dataset_choice, uploaded_file, use_pca, n_clusters, linkage_method):
    if dataset_choice != "Загрузить свой CSV":
        df = load_dataset(PREDEFINED_DATASETS[dataset_choice])
    else:
        if uploaded_file is None:
            raise gr.Error("Пожалуйста, загрузите CSV-файл.")
        df = load_dataset(uploaded_file)

    # Попытка найти целевую переменную (target)
    label_col = None
    for col in df.columns:
        if col.lower() in ["target", "label", "class"]:
            label_col = col
            break

    # Предобработка
    df_clean = clean_data(df)
    df_scaled, _ = scale_data(df_clean)

    # PCA
    if use_pca and df_scaled.shape[1] > 2:
        df_pca, pca_model, variance_ratio = apply_pca(df_scaled, n_components=2)
        pca_text = f"Объяснённая дисперсия PCA: {round(sum(variance_ratio)*100, 2)}%"
    else:
        df_pca = df_scaled.copy()
        pca_text = "PCA не применялся (данные уже двумерные или отключено)"

    # График истинных меток (если есть)
    fig_true = plot_true_labels(df_pca, df[label_col]) if label_col else None

    # Распределение
    fig_raw = plot_raw_distribution(df_pca)

    # KMeans
    labels_km, model_km = run_kmeans(df_scaled, n_clusters)
    fig_kmeans = plot_kmeans_clusters(df_pca, labels_km, model_km)
    fig_silhouette = plot_silhouette(df_scaled, labels_km)
    fig_elbow = plot_elbow(df_scaled)
    inertia = calculate_inertia(model_km)
    silhouette_val = calculate_silhouette_score(df_scaled, labels_km)

    # Agglomerative
    labels_hier, _ = run_hierarchical(df_scaled, n_clusters, linkage_method)
    fig_hier = plot_hierarchical_clusters(df_pca, labels_hier)
    fig_dendro = plot_dendrogram(df_scaled, method=linkage_method)

    return (
        pca_text,
        fig_true if fig_true else fig_raw,
        fig_kmeans,
        fig_hier,
        fig_dendro,
        fig_elbow,
        fig_silhouette,
        f"Inertia (KMeans): {round(inertia, 2)}",
        f"Silhouette Score: {round(silhouette_val, 3)}"
    )

with gr.Blocks(title="Кластеризация и визуализация") as demo:
    gr.Markdown("Визуализация кластеризации (KMeans и иерархической)")

    with gr.Row():
        dataset_choice = gr.Radio(
            choices=["Wine", "Wholesale", "Mall Customers", "Загрузить свой CSV"],
            label="Выберите набор данных",
            value="Wine"
        )
        uploaded_file = gr.File(label="Загрузите CSV", file_types=[".csv"])

    with gr.Row():
        use_pca = gr.Checkbox(label="Применить PCA (снижение до 2D)", value=True)
        n_clusters = gr.Slider(2, 10, value=3, step=1, label="Количество кластеров (k)")
        linkage_method = gr.Dropdown(choices=["ward", "complete", "average", "single"], label="Метод связи (иерархическая)", value="ward")

    run_button = gr.Button("▶️ Выполнить кластеризацию")

    pca_output = gr.Textbox(label="Информация о PCA")
    fig1 = gr.Plot(label="Распределение (или реальные классы)")
    fig2 = gr.Plot(label="Кластеры KMeans")
    fig3 = gr.Plot(label="Кластеры иерархические")
    fig4 = gr.Plot(label="Дендрограмма")
    fig5 = gr.Plot(label="Elbow-график")
    fig6 = gr.Plot(label="Силуэт-график")
    inertia_output = gr.Textbox(label="Инерция (KMeans)")
    silhouette_output = gr.Textbox(label="Silhouette Score")

    run_button.click(
        fn=run_pipeline,
        inputs=[dataset_choice, uploaded_file, use_pca, n_clusters, linkage_method],
        outputs=[pca_output, fig1, fig2, fig3, fig4, fig5, fig6, inertia_output, silhouette_output]
    )

if __name__ == "__main__":
    demo.launch()