import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from data_splitter import DataSplitter
from data_visualizer import DataVisualizer
from config import DATASET_PATH, rp
import pandas as pd
import matplotlib.pyplot as plt
from model_trainer import ModelTrainer

# Impostazioni per Pandas e Matplotlib
pd.set_option('display.max_rows', None)     # Mostra tutte le righe in Pandas
plt.rcParams["figure.figsize"] = (10, 10)   # Imposta la dimensione predefinita delle figure

# Iperparametri manuali per RandomForest
manual_params_rf = {
    "n_estimators": 100,            # Numero di alberi nel modello
    "max_depth": None,              # Profondità massima degli alberi (None per illimitata)
    "min_samples_split": 10,        # Minimo numero di campioni per dividere un nodo
    "min_samples_leaf": 2,          # Minimo numero di campioni per foglia
    "max_features": "sqrt",         # Numero massimo di feature considerate per split
    "n_jobs": -1,                   # Utilizza tutti i core CPU per velocizzare
}

# Iperparametri manuali per XGBoost
manual_params_xgb = {
    "n_estimators": 200,            # Numero di alberi
    "max_depth": 9,                 # Profondità massima degli alberi
    "learning_rate": 0.1,           # Tasso di apprendimento
    "colsample_bytree": 1.0,        # Frazione di feature per albero
    "subsample": 1.0,               # Frazione di campioni per albero
    "n_jobs": -1,                   # Parallel processing
    "verbosity": 1,                 # Mostra progressi
    "use_label_encoder": False      # Evita warning di XGBoost
}

# Iperparametri manuali per LightGBM
manual_params_lgbm = {
    "n_estimators": 100,            # Numero di alberi
    "learning_rate": 0.1,           # Tasso di apprendimento
    "max_depth": 5,                 # Profondità massima
    "num_leaves": 31,               # Numero di foglie per albero
    "subsample": 0.8,               # Frazione di campioni
    "colsample_bytree": 0.8,        # Frazione di feature
    "n_jobs": -1,                   # Utilizza tutti i core
    "objective": "multiclass",      # Classificazione multiclasse
    "metric": "multi_logloss",      # Metrica di perdita
    "verbose": -1                   # Supprime output
}

# Definisce il metodo di gestione dello squilibrio e parametri
IMBALANCE_METHOD = 'smote'  # Opzioni: 'smote' o 'undersample'
IMBALANCE_PARAMS = {'undersampling_ratio': 0.15} if IMBALANCE_METHOD == 'undersample' else {'sampling_strategy': 1.0}

if __name__ == "__main__":

    # ===================== LOAD DATA ===================== #
    loader = DataLoader(DATASET_PATH)       # Inizializza il caricatore del fucile di umberto
    df = loader.load_data()                 # Carica il dataset

    # ===================== PREPROCESS DATA ===================== #
    preprocessor = DataPreprocessor(df)
    df_clean = preprocessor.clean_data()    # Pulisce il dataset

    # ===================== DATA VISUALIZATION (Before Label Mapping) ===================== #
    visualizer = DataVisualizer(df_clean, rp)
    visualizer.plot_label_distribution()    # Plotta la distribuzione delle etichette (prima del label mapping)

    # ===================== LABEL MAPPING ===================== #
    df_clean = preprocessor.map_labels()

    # ===================== SPLIT DATA ===================== #
    splitter = DataSplitter(df_clean, preprocessor)
    splitter.extract_features_and_labels()                          # Estrae feature ed etichette
    X_train, X_test, y_train, y_test = splitter.split_data()        # Divide in training e test
    feature_names = df_clean.drop(columns=['Label']).columns.tolist()       # Ottiene i nomi delle feature

    # ===================== HANDLE IMBALANCE ON TRAINING SET ONLY ===================== #
    X_train_balanced, y_train_balanced = preprocessor.handle_imbalance(X_train, y_train, method=IMBALANCE_METHOD,
                                                                       **IMBALANCE_PARAMS)      # Bilancia i dati

    # Mantiene X_test e y_test invariati
    X_test = X_test.select_dtypes(include=['int64', 'float64', 'int32', 'float32'])


    # Converte in DataFrame per visualizzazione se necessario
    if not isinstance(X_train_balanced, pd.DataFrame):
        X_train_balanced_df = pd.DataFrame(X_train_balanced,
                                           columns=X_train.columns if isinstance(X_train, pd.DataFrame) else [
                                               f'feature_{i}' for i in range(X_train.shape[1])])
        y_train_balanced_series = pd.Series(y_train_balanced, name='Label')
        df_train_balanced = pd.concat([X_train_balanced_df, y_train_balanced_series], axis=1)
    else:
        df_train_balanced = pd.concat([X_train_balanced, y_train_balanced], axis=1)

    # Aggiorna i dati di training per visualizzazione e processamento
    X_train = X_train_balanced
    y_train = y_train_balanced

    # ===================== 📊 DATA VISUALIZATION (After Label Mapping) ===================== #
    visualizer = DataVisualizer(df_clean, rp)
    visualizer.plot_label_distribution()
    visualizer.plot_train_test_distribution(y_train, y_test)


    print("\n✅ Dati pronti!")
    print(f"Forma di X_train: {X_train.shape}")
    print(f"Forma di y_train: {y_train.shape}")
    print(f"Forma di X_test: {X_test.shape}")
    print(f"Forma di y_test: {y_test.shape}")


    # Codifica delle etichette per XGBoost e LightGBM
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Ottimizza l'uso della memoria per le etichette codificate
    y_train_encoded = y_train_encoded.astype(np.int32)
    y_test_encoded = y_test_encoded.astype(np.int32)

    # Converte X_train e X_test in float32 per ridurre la memoria
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # Divide i dati di test per validazione (opzionale, 50% del test set) --> per questioni di efficienza nel plottare learning curve
    X_val, _, y_val, _ = train_test_split(X_test, y_test, test_size=0.5, random_state=42, stratify=y_test)

    # Codifica y_val per XGBoost/LightGBM
    y_val_encoded = label_encoder.transform(y_val)

    # Sottocampiona X_train e y_train per le curve di apprendimento --> per question di efficienza nel plottare learning curve
    n_samples = min(100000, len(X_train))
    subset_indices = np.random.choice(len(X_train), n_samples, replace=False)
    X_train_subset = X_train.iloc[subset_indices]  # Use .iloc[] for row indexing
    y_train_subset = y_train.iloc[subset_indices]  # Use .iloc[] if y_train is a Series


    # ===================== 🔥 TRAIN & COMPARE MODELS ===================== #

    # Train Random Forest
    print("\n🌲 Training Random Forest...")

    # print("\n  Before HYPERPARAMETER TUNING:")
    # trainer_rf = ModelTrainer(X_train, y_train, X_test, y_test, visualizer, model_type="random_forest", n_jobs=-1)
    # rf_model = trainer_rf.train()
    #
    # # Evaluate RANDOM FOREST BEFORE TUNING
    # print("\n🔍 Evaluating Random Forest before tuning of parameters...")
    # trainer_rf.evaluate(X_test, y_test)




    # print("\n  After HYPERPARAMETER TUNING:")
    trainer_rf = ModelTrainer(X_train, y_train, X_test, y_test, visualizer, model_type="random_forest", **manual_params_rf)
    # Perform hyperparameter tuning for Random Forest
    # trainer_rf.hyperparameter_tuning(X_train, y_train)

    # Train the final Random Forest model (now with the best parameters)
    rf_model = trainer_rf.train()


    #print("\n🔍 Evaluating Random Forest after tuning of parameters...")
    trainer_rf.evaluate(X_test, y_test)


    # Perform Cross-Validation for Random Forest
    # cv_score_rf = trainer_rf.cross_validate(X_train, y_train)

    # Learning Curve for Random Forest
    #print("\n📈 Plotting Learning Curves for Random Forest...")
    #visualizer.plot_learning_curve(rf_model, X_train_subset, y_train_subset, X_val, y_val, model_type="random_forest")

    # Feature Importance for Random Forest
    print("\n📊 Plotting Feature Importance for Random Forest...")
    visualizer.plot_feature_importance(rf_model, feature_names, 'Random Forest')






    # Train XGBoost
    print("\n🚀 Training XGBoost...")
    # print("\n Before HYPERPARAMETER TUNING...")
    # trainer_xgb = ModelTrainer(X_train, y_train_encoded, X_test, y_test_encoded, visualizer, model_type="xgboost", n_jobs=-1)
    # xgb_model = trainer_xgb.train()
    # print("\n🔍 Evaluating XGBoost...")
    # trainer_xgb.evaluate(X_test, y_test_encoded)
    #
    # print("\n After HYPERPARAMETER TUNING...")
    trainer_xgb = ModelTrainer(X_train, y_train_encoded, X_test, y_test_encoded, visualizer, model_type="xgboost", **manual_params_xgb)
    # Perform hyperparameter tuning for XGBoost
    # trainer_xgb.hyperparameter_tuning(X_train, y_train_encoded)

    # Train the final XGBoost model (now with the best parameters)
    xgb_model = trainer_xgb.train()

    # Perform Cross-Validation for XGBoost
    # cv_score_xgb = trainer_xgb.cross_validate(X_train, y_train_encoded)

    print("\n🔍 Evaluating XGBoost...")
    trainer_xgb.evaluate(X_test, y_test_encoded)

    # Learning Curve for XGBoost
    #print("\n📈 Plotting Learning Curves for XGBoost...")
    #visualizer.plot_learning_curve(xgb_model, X_train, y_train_encoded, X_val, y_val_encoded, model_type="xgboost")

    # Feature Importance for XGBoost
    print("\n📊 Plotting Feature Importance for XGBoost...")
    visualizer.plot_feature_importance(xgb_model, feature_names, 'XGBoost')






    # Train LightGBM
    print("\n🌟 Training LightGBM...")
    trainer_lgbm = ModelTrainer(X_train, y_train_encoded, X_test, y_test_encoded, visualizer, model_type="lightgbm",
                                **manual_params_lgbm)
    # Perform hyperparameter tuning for LightGBM (optional, uncomment if needed)
    # trainer_lgbm.hyperparameter_tuning(X_train, y_train_encoded)

    # Train the final LightGBM model
    lgbm_model = trainer_lgbm.train()
    # Perform Cross-Validation for LightGBM (optional, uncomment if needed)
    # cv_score_lgbm = trainer_lgbm.cross_validate(X_train, y_train_encoded)

    # Evaluate LightGBM
    print("\n🔍 Evaluating LightGBM...")
    trainer_lgbm.evaluate(X_test, y_test_encoded)

    # Learning Curve for LightGBM
    #print("\n📈 Plotting Learning Curves for LightGBM...")
    #visualizer.plot_learning_curve(lgbm_model, X_train, y_train_encoded, X_val, y_val_encoded, model_type="lightgbm")

    # Feature Importance for LightGBM
    print("\n📊 Plotting Feature Importance for LightGBM...")
    visualizer.plot_feature_importance(lgbm_model, feature_names, 'LightGBM')



    # Compare Training & Prediction Time
    print("\n📊 Comparing Model Performance...")
    visualizer.compare_training_times(trainer_rf, trainer_xgb, trainer_lgbm)
    visualizer.compare_model_performance(trainer_rf, trainer_xgb, trainer_lgbm, X_test, y_test, y_test_encoded)




    # # Print comparison of cross-validation scores (if uncommented above)
    # if cv_score_rf and cv_score_xgb and cv_score_lgbm:
    #     print(f"📊 Random Forest Mean CV Accuracy: {cv_score_rf:.4f}")
    #     print(f"📊 XGBoost Mean CV Accuracy: {cv_score_xgb:.4f}")
    #     print(f"📊 LightGBM Mean CV Accuracy: {cv_score_lgbm:.4f}")

    print("\n✅ All models trained, evaluated, and visualized!")
