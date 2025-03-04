from sklearn.model_selection import train_test_split

# Classe per dividere il dataset in set di training e test
class DataSplitter:
    # Inizializza l'oggetto con una copia del DataFrame e il preprocessore
    def __init__(self, df, preprocessor):
        self.df = df.copy()                         # Crea una copia del DataFrame per evitare modifiche al dato originale
        self.preprocessor = preprocessor
        self.X = None                               # Variabile per memorizzare le feature
        self.y = None                               # Variabile per memorizzare le etichette (labels)
        self.X_train = None                         # Variabile per il set di training delle feature
        self.X_test = None                          # Variabile per il set di test delle feature
        self.y_train = None                         # Variabile per le etichette di training
        self.y_test = None                          # Variabile per le etichette di test


    def extract_features_and_labels(self):
        """Estrazione delle features e delle labels dal dataframe."""
        # Seleziona solo le colonne numeriche come feature
        numerical_cols = self.df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
        self.X = self.df[numerical_cols]  # Assegna le feature numeriche a self.X
        self.y = self.df['Label']  # Assegna le etichette a self.y
        print(f"\n✅ Extracted features (numerical only): {self.X.shape}, Labels: {self.y.shape}")

    def split_data(self, test_size=0.4, random_state=42):
        """
        Divide i dati casualmente: 60% per training, 40% per test, con stratificazione per etichette.
        - test_size: proporzione del dataset per il test (0.4)
        - random_state: seme per la riproducibilità (default 42)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        print(f"✅ Dati partizionati: Dimensione del set per il train: {X_train.shape}, Dimensione del set per il test: {X_test.shape}")

        self.X_train = X_train      # Assegna il set di training delle feature
        self.y_train = y_train      # Assegna le etichette di training
        self.X_test = X_test        # Assegna il set di test delle feature
        self.y_test = y_test        # Assegna le etichette di test

        return self.X_train, self.X_test, self.y_train, self.y_test