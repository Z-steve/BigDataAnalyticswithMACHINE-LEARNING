import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import learning_curve


# Classe per visualizzare i dati e le prestazioni dei modelli
class DataVisualizer:
    # Inizializza l'oggetto con il DataFrame e la palette dei colori
    def __init__(self, df, color_palette):
        self.df = df
        self.color_palette = color_palette
        self.label_distribution = self.df['Label'].value_counts()    # Distribuzione iniziale delle etichette

    # Metodo per plottare la distribuzione delle etichette nel dataset
    def plot_label_distribution(self):
        # Ottiene gli indici e i valori della distribuzione delle etichette
        x_coords = self.label_distribution.index
        y_coords = self.label_distribution.values

        fig, ax = plt.subplots(figsize=(12, 8))    # Crea una figura con dimensioni specificate
        ax.barh(x_coords, y_coords, color=self.color_palette["iris"], edgecolor=self.color_palette["base"])  # Istogramma orizzontale

        for i, value in enumerate(y_coords):
            ax.text(value, i, f" {value:,}", va="center", fontsize=10, color=self.color_palette["base"])   # Aggiunge etichette ai valori

        ax.set_title("Distribution of Network Activity", fontsize=16, color=self.color_palette["base"]) # Titolo
        ax.set_xlabel("Number of Instances", fontsize=12, color=self.color_palette["base"])         # Etichetta su asse x
        ax.set_ylabel("Network Activity Label", fontsize=12, color=self.color_palette["base"])      # Etichetta su asse y

        plt.tight_layout()
        plt.show()

    # Metodo per plottare la distribuzione delle etichette nei set di training e test
    def plot_train_test_distribution(self, y_train, y_test):
        # Calcola la distribuzione delle etichette per training e test
        train_label_distribution = y_train.value_counts()
        test_label_distribution = y_test.value_counts()

        fig, (ax_train, ax_test) = plt.subplots(nrows=1, ncols=2, figsize=(18, 8))  # Due subplot affiancati

        # Training set
        ax_train.barh(train_label_distribution.index, train_label_distribution.values,
                      color=self.color_palette["iris"], edgecolor=self.color_palette["base"])
        for i, value in enumerate(train_label_distribution.values):
            ax_train.text(value, i, f" {value:,}", va="center", fontsize=10, color="black")
        ax_train.set_title("Training Set Distribution", fontsize=16)
        ax_train.set_xlabel("Number of Instances")     # Etichetta x training
        ax_train.set_ylabel("Network Activity Label")   # Etichetta y training

        # Test set
        ax_test.barh(test_label_distribution.index, test_label_distribution.values,
                     color=self.color_palette["iris"], edgecolor=self.color_palette["base"])
        for i, value in enumerate(test_label_distribution.values):
            ax_test.text(value, i, f" {value:,}", va="center", fontsize=10, color="black")
        ax_test.set_title("Test Set Distribution", fontsize=16)
        ax_test.set_xlabel("Number of Instances")       # Etichetta x testing
        ax_test.set_ylabel("Network Activity Label")    # Etichetta y testing

        plt.tight_layout()
        plt.show()

    # Metodo per plottare la matrice di confusione
    def plot_confusion_matrix(self, y_true, y_pred):

        cm = confusion_matrix(y_true, y_pred)       # Calcola la matrice di confusione
        labels = sorted(np.unique(y_true))          # Ottiene le etichette uniche ordinate

        plt.figure(figsize=(12, 10))                # Crea una figura con dimensioni specificate
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted Label", fontsize=8)   # Etichetta asse x
        plt.ylabel("True Label", fontsize=8)        # Etichetta asse y
        plt.title("Confusion Matrix", fontsize=14)   # Titolo
        plt.show()


    def plot_learning_curve(self, model, X_train, y_train, X_val=None, y_val=None, model_type="random_forest"):
        """
        Plotta le curve di apprendimento per Random Forest, XGBoost o LightGBM.
        Parametri:
        - model: istanza del modello
        - X_train, y_train: dati di training
        - X_val, y_val: dati di validazione (opzionali)
        - model_type: tipo di modello
        """
        plt.figure(figsize=(10, 6))

        # Utilizzo di multi_logloss come metrica di valutazione che misura differenza tra predizione del modello ed i veri valori delle labels
        if model_type in ['xgboost', 'lightgbm']:
            # Gestisce le iterazioni per XGBoost e LightGBM
            eval_set = [(X_train, y_train), (X_val, y_val)] if X_val is not None and y_val is not None else [
                (X_train, y_train)]

            if model_type == 'xgboost':
                model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
                results = model.evals_result()
                epochs = len(results['validation_0']['mlogloss'])
            else:  # lightgbm
                model.fit(X_train, y_train, eval_set=eval_set, eval_metric='multi_logloss')
                results = model.evals_result_
                epochs = len(results['training']['multi_logloss'])
                # Plot training data
                plt.plot(range(1, epochs + 1), results['training']['multi_logloss'], label='Train Log Loss')
                if X_val is not None and y_val is not None:
                    epochs_val = len(results['valid_1']['multi_logloss'])
                    plt.plot(range(1, epochs_val + 1), results['valid_1']['multi_logloss'], label='Validation Log Loss')

            x_axis = range(1, epochs + 1)
            plt.xlabel('Iterations')
            plt.ylabel('Log Loss')
            plt.title(f'{model_type.capitalize()} Learning Curve')

        elif model_type == 'random_forest':
            # Usa learning_curve per Random Forest
            train_sizes, train_scores, val_scores = learning_curve(
                model, X_train, y_train, cv=5, scoring="accuracy", train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=1
            )
            plt.plot(train_sizes, train_scores.mean(axis=1), label='Train Accuracy')
            if X_val is not None and y_val is not None:
                val_scores_val = [accuracy_score(y_val, model.predict(X_val)) for _ in
                                  train_sizes]
                plt.plot(train_sizes, val_scores_val, label='Validation Accuracy')
            else:
                plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation Accuracy')
            plt.xlabel('Training Set Size')
            plt.ylabel('Accuracy')
            plt.title('Random Forest Learning Curve')

        else:
            raise ValueError("Invalid model_type. Use 'random_forest', 'xgboost', or 'lightgbm'.")

        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_feature_importance(self, model, feature_names, model_name):
        """
        Metodo per plottare l'importanza delle feature
        Parametri:
        - model: istanza del modello di training (Random Forest, XGBoost, or LightGBM)
        - feature_names: lista dei nomi delle features
        - model_name: nome del modello (e.g., 'Random Forest')
        """
        importances = model.feature_importances_        # Ottiene l'importanza delle feature
        indices = np.argsort(importances)[::-1]         # Ordina gli indici in ordine decrescente
        top_n = min(20, len(feature_names))             # Limita a 20 feature o tutte se meno di 20

        plt.figure(figsize=(12, 6))
        plt.bar(range(top_n), importances[indices[:top_n]], color='purple')         # Crea un grafico a barre
        plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=90, fontsize=8)      # Etichette ruotate
        plt.title(f'{model_name} Feature Importances (Top {top_n})')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # Metodo per confrontare i tempi di addestramento
    def compare_training_times(self, trainer_rf, trainer_xgb, trainer_lgbm):
        times = [trainer_rf.train_time, trainer_xgb.train_time, trainer_lgbm.train_time]
        labels = ["Random Forest", "XGBoost", "LightGBM"]

        plt.figure(figsize=(8, 5))
        plt.bar(labels, times, color=['green', 'blue', 'purple'])
        plt.ylabel("Training Time (seconds)")
        plt.title("Training Time Comparison")
        plt.show()

    def compare_model_performance(self, trainer_rf, trainer_xgb, trainer_lgbm, X_test, y_test, y_test_encoded=None):
        """
        Metodo per confrontare le prestazioni dei modelli

        Parametri:
        - trainer_rf, trainer_xgb, trainer_lgbm: istanze per ogni modello
        - X_test: test features
        - y_test: test labels (per Random Forest)
        - y_test_encoded: test labels codificate (per XGBoost e LightGBM)
        """

        # Accuratezza training
        train_acc = [
            trainer_rf.model.score(trainer_rf.X_train, trainer_rf.y_train),
            trainer_xgb.model.score(trainer_xgb.X_train, trainer_xgb.y_train),
            trainer_lgbm.model.score(trainer_lgbm.X_train, trainer_lgbm.y_train)
        ]

        # Accuratezza testing
        test_acc = [
            trainer_rf.model.score(X_test, y_test),
            trainer_xgb.model.score(X_test, y_test_encoded),  # Usa y_test codificato per XGBoost (non accetta stringhe)
            trainer_lgbm.model.score(X_test, y_test_encoded)  # Usa y_test codificato per LightGBM (non accetta stringhe)
        ]

        labels = ["Random Forest", "XGBoost", "LightGBM"]
        x = np.arange(len(labels))
        width = 0.25

        plt.figure(figsize=(12, 8))
        bars1 = plt.bar(x - width / 2, train_acc, width, color='green', alpha=0.6, label="Train Accuracy")
        bars2 = plt.bar(x + width / 2, test_acc, width, color='blue', alpha=0.6, label="Test Accuracy")

        # Aggiunge le label (minuscole) sulle barre
        for bar in bars1:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.4f}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.4f}', ha='center', va='bottom', fontsize=8)

        plt.xlabel("Models", fontsize=12)
        plt.ylabel("Accuracy", fontsize=12)
        plt.title("Model Accuracy Comparison", fontsize=14)
        plt.xticks(x, labels, rotation=45, ha='right')  # Ruota le labels per questioni di visibilità
        plt.ylim(0.997, 1.002)  # Specifica il range di visualizzazione nel grafico
        plt.legend(fontsize=10)

        # Aggiunge linee di griglia per chiarezza
        plt.grid(True, linestyle='--', alpha=0.5, which='both')

        plt.tight_layout()
        plt.show()
