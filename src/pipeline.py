from preprocessing import load_and_clean_data
from clustering import prepare_clustering_data, apply_final_kmeans, add_cluster_target
from classification import (
    prepare_classification_data,
    create_classification_models,
    optimize_models,
    evaluate_best_models,
    select_best_model,
    save_best_model,
)

from src.classification import(
    setup_mlflow,
    log_model_experiment
)

import mlflow


def main():
    setup_mlflow()
    df_clean = load_and_clean_data()

    df_prepare_clustering , num_coul , df_scaled , skew_table = (
        prepare_clustering_data(df_clean)
    )

    df_clean , x_pca , kmeans , silhoutte , davies_bouldin , calinski_harabaz = (
        apply_final_kmeans(df_scaled , df_clean)
    )

    df_clean = add_cluster_target(
        df_clean,
        df_clean["cluster_kmeans"]
    )

    X_train , X_test , y_train , y_test = prepare_classification_data(
        df_clean,
        num_coul
    )

    models = create_classification_models()

    best_models , best_params = optimize_models(
        models,
        X_train,
        y_train
    )

    results = evaluate_best_models(
        best_models,
        X_test,
        y_test
    )

    for name , model in best_models.items():
        log_model_experiment(
            name = name,
            model = model,
            params=best_params[name],
            metrics=results[name]
        )

    best_name , best_model = select_best_model(
        best_models,
        results
    )

    save_best_model(best_model)

    print("Best Model:", best_name)
    print("Best Parameters:", best_params[best_name])
    print("Results:", results[best_name])
    print("Pipeline saved successfully.")

if __name__ == "__main__":
    main()