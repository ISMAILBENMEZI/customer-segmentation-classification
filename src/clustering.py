import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)


def prepare_clustering_data(df_clean):
    df_prepare_clustering = df_clean.copy()

    num_colu = df_prepare_clustering.select_dtypes(include="number").columns

    skew_before = df_prepare_clustering[num_colu].skew()

    df_prepare_clustering[num_colu] = np.log1p(df_prepare_clustering[num_colu])

    skew_after = df_prepare_clustering[num_colu].skew()

    skew_table = pd.DataFrame(
        {"Skewness Before": skew_before, "Skewness After": skew_after}
    )

    scaler = StandardScaler()

    df_scaled = scaler.fit_transform(df_prepare_clustering[num_colu])

    df_prepare_clustering.to_csv(
        "data/processed/df_prepare_clustering.csv",
        index=False
    )
    return df_prepare_clustering, num_colu, df_scaled, skew_table


def apply_pca(df_scaled):
    pca = PCA()

    x_pca = pca.fit_transform(df_scaled)

    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    n_components = np.argmax(cumulative_variance >= 0.80) + 1

    return (x_pca, explained_variance, cumulative_variance, n_components)


def search_kmeans(df_scaled):
    results = []

    for n_components in range(2, 6):
        pca = PCA(n_components=n_components)

        x_pca = pca.fit_transform(df_scaled)

        for k in range(2, 8):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)

            labels = kmeans.fit_predict(x_pca)

            results.append(
                {
                    "pca_components": n_components,
                    "k": k,
                    "inertia": kmeans.inertia_,
                    "silhouette": silhouette_score(x_pca, labels),
                }
            )

    return pd.DataFrame(results)


def apply_final_kmeans(df_scaled, df_clean):
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(df_scaled)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

    clusters = kmeans.fit_predict(x_pca)

    silhouette = silhouette_score(x_pca, clusters)
    davies_bouldin = davies_bouldin_score(x_pca, clusters)
    calinski_harabasz = calinski_harabasz_score(x_pca, clusters)

    df_clean = df_clean.copy()

    df_clean["cluster_kmeans"] = clusters

    return (df_clean, x_pca, kmeans, silhouette, davies_bouldin, calinski_harabasz)


def get_cluster_balance(clusters):
    cluster_balance = pd.Series(clusters).value_counts().reset_index()

    cluster_balance.columns = ["cluster", "count"]

    cluster_balance["percentage"] = cluster_balance["count"] / len(clusters) * 100

    return cluster_balance


def search_dbscan(df_scaled):
    results = []

    for n_components in range(2, 6):
        pca = PCA(n_components=n_components)
        x_pca = pca.fit_transform(df_scaled)

        for eps in [0.3, 0.5, 0.7, 1.0, 1.5]:
            for min_samples in [5, 10, 15]:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)

                labels = dbscan.fit_predict(x_pca)

                n_cluster = len(set(labels) - {-1})

                nois_percentage = (labels == -1).mean() * 100

                if n_cluster >= 2:
                    mask = labels != -1

                    silhouette = silhouette_score(x_pca[mask], labels[mask])
                else:
                    silhouette = np.nan

                results.append(
                    {
                        "pca_components": n_components,
                        "eps": eps,
                        "min_samples": min_samples,
                        "n_clusters": n_cluster,
                        "nois_percentage": nois_percentage,
                        "silhouette": silhouette,
                    }
                )

    return pd.DataFrame(results)


def apply_final_dbscan(df_scaled):
    pca = PCA(n_components=2)

    x_pca = pca.fit_transform(df_scaled)

    dbscan = DBSCAN(eps=0.3, min_samples=15)

    labels = dbscan.fit_predict(x_pca)

    mask = labels != -1

    silhouette = silhouette_score(x_pca[mask], labels[mask])

    davies_bouldin = davies_bouldin_score(x_pca[mask], labels[mask])

    calinski_harabasz = calinski_harabasz_score(x_pca[mask], labels[mask])

    noise_percentage = (labels == -1).mean() * 100

    return (
        x_pca,
        labels,
        silhouette,
        davies_bouldin,
        calinski_harabasz,
        noise_percentage,
    )


def compare_clustering_models(kmeans_matrics, dbscan_metrics):
    comparison = pd.DataFrame(
        {
            "silhoutte": [kmeans_matrics["silhouette"], dbscan_metrics["silhouette"]],
            "Davies_bouldin": [
                kmeans_matrics["davies_bouldin"],
                dbscan_metrics["davies_bouldin"],
            ],
            "Calinski_harabasz": [
                kmeans_matrics["calinski_harabasz"],
                dbscan_metrics["calinski_harabasz"],
            ],
        }
    )

    return comparison


def add_cluster_target(df_clean, clusters):
    df_clean = df_clean.copy()

    df_clean["cluster_final"] = clusters

    cluster_names = {
        0: "Clients à Risque",
        1: "Clients Premium",
        2: "Clients à Faible Activité",
    }

    df_clean["target"] = df_clean["cluster_final"].map(
        cluster_names
    )

    return df_clean
