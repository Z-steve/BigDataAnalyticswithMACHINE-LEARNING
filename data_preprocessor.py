import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample

# Classe per preprocessare il dataset, gestendo pulizia, mappatura etichette e bilanciamento
class DataPreprocessor:
    # Inizializza l'oggetto con una copia del DataFrame di input
    def __init__(self, df):
        self.df = df.copy() # Crea una copia del DataFrame per evitare modifiche al dato originale

    def clean_data(self):
        """Pulizia del dataset rimuovendo NaN, infinite values, duplicates, and non-numerical features."""
        self.df.columns = self.df.columns.str.strip()  # Rimuove spazi bianchi dai nomi delle colonne per evitare errori

        # Mantiene solo le etichette di interesse (attacchi e BENIGN)
        if 'Label' in self.df.columns:
            before = self.df.shape[0]
            self.df = self.df[self.df['Label'].isin(['DDoS', 'BENIGN', 'DoS GoldenEye', 'DoS Hulk', 'DoS Slowhttptest', 'DoS slowloris', 'Bot', 'FTP-Patator', 'SSH-Patator', 'Heartbleed', 'Infiltration', 'PortScan', 'Web Attack � Brute Force', 'Web Attack � Sql Injection', 'Web Attack � XSS'])]
            print(f"\n✅ Removed {before - self.df.shape[0]} entries that were not DDoS, BENIGN, DoS GoldenEye, DoS Hulk, DoS Slowhttptest, DoS slowloris, Botnet, FTP-Patator, SSH-Patator, Heartbleed, Infiltration, PortScan, Web Attack � Brute Force, Web Attack � Sql Injection, Web Attack � XSS")

        # Identifica e rimuove colonne non numeriche, tranne 'Label' (e.g., Flow ID, Source IP, Destination IP)
        numerical_cols = self.df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
        non_numerical_cols = [col for col in self.df.columns if col not in numerical_cols and col != 'Label']
        if non_numerical_cols:
            print(f"\n⚠️ Dropping non-numerical columns: {non_numerical_cols}")
            self.df = self.df.drop(columns=non_numerical_cols)

        # Sostituzione valori infiniti con NaN
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Rimozione valori NaN
        self.df.dropna(inplace=True)

        # Rimozione righe duplicate
        self.df.drop_duplicates(inplace=True)

        print(f"\n✅ Data Preprocessing Completato! Struttura finale: {self.df.shape}")
        return self.df

    # Metodo per mappare le etichette in categorie semplificate
    def map_labels(self):
        """Mapping della colonna 'Label' in categorie."""
        category_mapping = {
            'BENIGN': 'BENIGN',
            'Bot': 'BOTNET',
            'DDoS': 'DOS',
            'DoS GoldenEye': 'DOS',
            'DoS Hulk': 'DOS',
            'DoS Slowhttptest': 'DOS',
            'DoS slowloris': 'DOS',
            'FTP-Patator': 'BRUTE_FORCE',
            'SSH-Patator': 'BRUTE_FORCE',
            'Heartbleed': 'WEB_ATTACK',
            'Infiltration': 'WEB_ATTACK',
            'PortScan': 'RECONNAISSANCE',
            'Web Attack � Brute Force': 'WEB_ATTACK',
            'Web Attack � Sql Injection': 'WEB_ATTACK',
            'Web Attack � XSS': 'WEB_ATTACK'
        }
        # Applicazione del category mapping
        self.df['Label'] = self.df['Label'].map(category_mapping)
        print("✅ Mapping delle Labels eseguito con successo!")
        # Verifica la distribuzione delle nuove etichette
        print(self.df["Label"].value_counts())
        return self.df

    # Metodo per ridurre il numero di istanze BENIGN nel training set
    def undersample_benign(self, X, y, target_count=None, undersampling_ratio=None):
        """
        - X: Features (DataFrame or NumPy array)
        - y: Labels (Series or NumPy array)
        - target_count: numero specifico di entries (e.g., 500,000).
        - undersampling_ratio: percentuale di Benign da mantenere (e.g., 0.5 per la metà).
        """
        # Converte X e y in DataFrame/Series se non lo sono già
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        else:
            X_df = X.copy()
        if isinstance(y, np.ndarray):
            y_series = pd.Series(y, name='Label')
        else:
            y_series = y.copy()

        # Combina X e y in un unico DataFrame per l'undersampling
        df = pd.concat([X_df, y_series], axis=1)

        # Separa le istanze BENIGN da quelle non BENIGN
        benign = df[df['Label'] == 'BENIGN']
        non_benign = df[df['Label'] != 'BENIGN']

        # Determina il numero di istanze BENIGN da mantenere
        if target_count is not None:
            n_benign_retain = target_count
        elif undersampling_ratio is not None:
            n_benign_retain = int(len(benign) * undersampling_ratio)
        else:
            raise ValueError("Specify either target_count or undersampling_ratio")

        # Campiona casualmente le istanze BENIGN
        benign_undersampled = resample(benign,
                                       replace=False,
                                       n_samples=n_benign_retain,
                                       random_state=42)

        # Combina le istanze BENIGN campionate con quelle non BENIGN
        df_undersampled = pd.concat([benign_undersampled, non_benign], axis=0)
        print(f"✅ Reduced BENIGN to {n_benign_retain} entries")
        print(f"New class distribution:\n{df_undersampled['Label'].value_counts()}")

        # Separa nuovamente in X e y
        X_undersampled = df_undersampled.drop(columns=['Label']).select_dtypes(include=['int64', 'float64', 'int32', 'float32'])
        y_undersampled = df_undersampled['Label']

        return X_undersampled, y_undersampled


    # Metodo per bilanciare il dataset con SMOTE
    def balance_data(self, X, y, sampling_strategy='auto'):
        # Verifica la strategia di campionamento e configura SMOTE di conseguenza
        if sampling_strategy == 'auto':
            smote = SMOTE(random_state=42)
            print(f"\n🔍 Applying SMOTE with auto strategy (matching majority class count)...")
        else:
            # Controlla se sampling_strategy è un float (rapporto) o un dizionario
            if isinstance(sampling_strategy, (int, float)):
                # Calcola il conteggio della classe maggioritaria
                unique, counts = np.unique(y, return_counts=True)
                class_counts = dict(zip(unique, counts))
                majority_class = max(class_counts, key=class_counts.get)
                majority_count = class_counts[majority_class]
                # Crea una strategia basata sul rapporto (es. 0.5 = 50% del conteggio maggioritario)
                sampling_strategy_dict = {cls: int(majority_count * sampling_strategy) for cls in class_counts if
                                          cls != majority_class}
                print(f"\n🔍 Applying SMOTE with ratio-based strategy: {sampling_strategy_dict}")
            else:
                # Se è un dizionario, lo usa direttamente
                sampling_strategy_dict = sampling_strategy
                print(f"\n🔍 Applying SMOTE with custom dictionary strategy: {sampling_strategy_dict}")
            smote = SMOTE(random_state=42, sampling_strategy=sampling_strategy_dict)

        # Applica SMOTE per bilanciare i dati
        X_balanced, y_balanced = smote.fit_resample(X, y)
        print(f"\n✅ Dataset Balanced with SMOTE in training set! New shape: {X_balanced.shape}")
        print(f"New class distribution in training set: {np.unique(y_balanced, return_counts=True)[1]}")
        return X_balanced, y_balanced

    # Metodo per gestire lo squilibrio delle classi con SMOTE o undersampling
    def handle_imbalance(self, X, y, method='undersample', **kwargs):
        """
        - X: Features (DataFrame or NumPy array)
        - y: Labels (Series or NumPy array)
        - method: 'smote' per SMOTE, 'undersample' per ridurre le entries di Benign.
        - kwargs: parametri per il metodo scelto (e.g., sampling_strategy per SMOTE, target_count/undersampling_ratio per undersampling).
        """
        if method == 'smote':
            return self.balance_data(X, y, **kwargs)
        elif method == 'undersample':
            return self.undersample_benign(X, y, **kwargs)
        else:
            raise ValueError("Method must be 'smote' or 'undersample'")