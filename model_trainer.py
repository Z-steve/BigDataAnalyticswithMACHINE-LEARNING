import time
from xgboost import XGBClassifier  # Import XGBoost
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier  # Import LightGBM
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, RandomizedSearchCV


# Classe per addestrare e valutare i modelli di Machine Learning
class ModelTrainer:
    # Inizializza l'oggetto con dati di training, test, visualizzatore e tipo di modello
    def __init__(self, X_train, y_train, X_test, y_test, visualizer, model_type="random_forest", **kwargs):
        self.X_train = X_train      # Dati di feature per il training
        self.y_train = y_train      # Etichette per il training
        self.X_test = X_test        # Dati di feature per il test
        self.y_test = y_test        # Etichette per il test
        self.visualizer = visualizer    # Riferimento al visualizzatore per grafici
        self.best_params = None     # Memorizza i migliori iperparametri dopo il tuning
        self.model_type = model_type    # Tipo di modello (default: random_forest)
        self.train_time = None      # Tempo di addestramento

        # Inizializza il modello in base al tipo specificato
        if model_type == "random_forest":
            self.model = RandomForestClassifier(random_state=42, **kwargs)  # Inizializza Random Forest
        elif model_type == "xgboost":
            self.model = XGBClassifier(eval_metric="mlogloss", random_state=42, **kwargs)   # Inizializza XGBoost
        elif model_type == "lightgbm":
            self.model = LGBMClassifier(random_state=42, **kwargs)  # Inizializza LightGBM
        else:
            raise ValueError("Invalid model type. Choose 'random_forest', 'xgboost', or 'lightgbm'.")


    # Metodo per addestrare il modello con i migliori iperparametri se disponibili
    def train(self):
        print(f"\n🔍 Training {self.model_type.capitalize()} Model...")

        print(f"🚀 Current model parameters before training: {self.model.get_params()}")  # per debug

        if self.best_params and not self.model.get_params():
            print(f"🚀 Using best hyperparameters: {self.best_params}")
            if self.model_type == "random_forest":
                self.model = RandomForestClassifier(**self.best_params, random_state=42, n_jobs=-1)
            elif self.model_type == "xgboost":
                self.model = XGBClassifier(**self.best_params, eval_metric="mlogloss", random_state=42)
            elif self.model_type == "lightgbm":
                self.model = LGBMClassifier(**self.best_params, random_state=42)

        start_fit = time.time()                             # Inizia il conteggio del tempo
        self.model.fit(self.X_train, self.y_train)          # Addestra il modello
        end_fit = time.time()                               # Termina il conteggio del tempo
        self.train_time = end_fit - start_fit               # Calcola il tempo di addestramento

        print(f"✅ Training time: {end_fit - start_fit:.2f} seconds")
        return self.model


    # Metodo per eseguire la cross validation  (utilizziamo 5 fold)
    def cross_validate(self, X_train, y_train, cv=5):
        print("\n🔄 Performing Cross-Validation...")
        scores = cross_val_score(self.model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)        # Esegue CV
        print(f"✅ Cross-Validation Accuracy Scores: {scores}")      # Stampa i punteggi
        print(f"📈 Mean CV Accuracy: {scores.mean():.4f}")           # Stampa la media dei punteggi
        return scores.mean()


    # Metodo per ottimizzare gli iperparametri con RandomizedSearchCV
    def hyperparameter_tuning(self, X_train, y_train):

        print(f"\n🔧 Performing Hyperparameter Tuning for {self.model_type.capitalize()}...")

        # Definisce la griglia di iperparametri per ogni modello
        if self.model_type == "random_forest":
            param_dist = {
                "n_estimators": [100, 200, 500],
                "max_depth": [10, 20, 30, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2"]
            }
        elif self.model_type == "xgboost":
            param_dist = {
                "n_estimators": [100, 200, 500],
                "learning_rate": [0.01, 0.1, 0.2, 0.3],
                "max_depth": [3, 6, 9],
                "subsample": [0.5, 0.7, 1.0],
                "colsample_bytree": [0.5, 0.7, 1.0]
            }
        elif self.model_type == "lightgbm":
            param_dist = {
                "n_estimators": [100, 200, 500],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7],
                "num_leaves": [31, 63, 127],  # Controlla la complessità degli alberi
                "subsample": [0.5, 0.7, 1.0],
                "colsample_bytree": [0.5, 0.7, 1.0]
            }

        # Inizializza RandomizedSearchCV con la griglia definita
        random_search = RandomizedSearchCV(
            self.model, param_distributions=param_dist, n_iter=10,
            cv=5, verbose=2, n_jobs=-1, random_state=42
        )

        start_search = time.time()                  # Inizia il conteggio del tempo
        random_search.fit(X_train, y_train)         # Esegue la ricerca
        end_search = time.time()                    # Termina il conteggio del tempo

        print(f"✅ Hyperparameter tuning time: {end_search - start_search:.2f} seconds")
        print(f"🏆 Best parameters found: {random_search.best_params_}")

        self.best_params = random_search.best_params_           # Memorizza i parametri
        self.model = random_search.best_estimator_              # Aggiorna il modello
        return self.model


    # Metodo per valutare le prestazioni del modello
    def evaluate(self, X_test, y_test):
        print("\n🏆 Evaluating Model Performance...")

        start_pred = time.time()                                # Inizia il conteggio del tempo di predizione
        y_pred = self.model.predict(X_test)                     # Esegue le predizioni
        end_pred = time.time()                                  # Termina il conteggio
        print(f"✅ Prediction time: {end_pred - start_pred:.2f} seconds")

        # Report dettagliato
        print("\n📊 Model Performance:")
        print(classification_report(y_test, y_pred, digits=4))

        # Confusion Matrix
        print("\n🔲 Confusion Matrix:")
        self.visualizer.plot_confusion_matrix(y_test, y_pred)

        train_accuracy = self.model.score(self.X_train, self.y_train)   # Accuratezza di training
        test_accuracy = self.model.score(X_test, y_test)                # Accuratezza di test (il metodo score, una volta passato X_test,
                                                                        # effettua nuovamente predizione e valutazione, quindi confronto con y_test effettivo (vero))
        print(f"\n✅ Training Accuracy: {train_accuracy:.4f}")
        print(f"✅ Testing Accuracy: {test_accuracy:.4f}")

        # Overfitting Detection
        if train_accuracy - test_accuracy > 0.05:
            print("⚠️ Warning: Possible Overfitting Detected!")

        return y_pred  # Restituisce le predizioni