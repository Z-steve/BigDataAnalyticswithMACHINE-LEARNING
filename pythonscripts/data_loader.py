import os
import pandas as pd

# Classe per caricare il dataset da file CSV
class DataLoader:
    # Inizializza l'oggetto DataLoader con il percorso del dataset e un DataFrame vuoto
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path  # Percorso del dataset
        self.df = None  # DataFrame per memorizzare i dati caricati

    # Metodo per caricare il dataset da un file CSV o una cartella di file CSV
    def load_data(self):
        # Verifica se il percorso specificato esiste, altrimenti solleva un'eccezione
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Path not found: {self.dataset_path}")

        # Controlla se il percorso è una cartella o un singolo file
        if os.path.isdir(self.dataset_path):
            # Ottiene la lista dei file CSV nella cartella
            csv_files = [f for f in os.listdir(self.dataset_path) if f.endswith('.csv')]

            # Verifica se ci sono file CSV nella cartella, altrimenti solleva un'eccezione
            if not csv_files:
                raise ValueError(f"No CSV files found in directory: {self.dataset_path}")

            print(f"\n📂 Found {len(csv_files)} CSV files. Loading...")  # Stampa il numero di file trovati
            df_list = []  # Lista per accumulare i DataFrame dei singoli file
            for file in csv_files:
                file_path = os.path.join(self.dataset_path, file)  # Costruisce il percorso completo del file
                try:
                    # Prova a caricare il file con codifica UTF-8
                    df = pd.read_csv(file_path, encoding='utf-8')
                except UnicodeDecodeError:
                    # Se UTF-8 fallisce, prova con Windows-1252
                    print(f"\n⚠️ UTF-8 decoding failed for {file}. Trying Windows-1252 encoding...")
                    df = pd.read_csv(file_path, encoding='Windows-1252')
                except UnicodeDecodeError:
                    # Se Windows-1252 fallisce, prova con Latin-1
                    print(f"\n⚠️ Windows-1252 decoding failed for {file}. Trying Latin-1 encoding...")
                    df = pd.read_csv(file_path, encoding='latin1')
                except Exception as e:
                    # Se tutti i tentativi falliscono, solleva un'eccezione con il messaggio di errore
                    raise ValueError(f"Could not load {file} with any encoding: {str(e)}")
                df_list.append(df)  # Aggiunge il DataFrame alla lista
            self.df = pd.concat(df_list, ignore_index=True)  # Concatena tutti i DataFrame in uno solo
        else:
            # Se è un singolo file, carica direttamente con gestione delle codifiche
            try:
                self.df = pd.read_csv(self.dataset_path, encoding='utf-8')
            except UnicodeDecodeError:
                print(f"\n⚠️ UTF-8 decoding failed for {self.dataset_path}. Trying Windows-1252 encoding...")
                self.df = pd.read_csv(self.dataset_path, encoding='Windows-1252')
            except UnicodeDecodeError:
                print(f"\n⚠️ Windows-1252 decoding failed for {self.dataset_path}. Trying Latin-1 encoding...")
                self.df = pd.read_csv(self.dataset_path, encoding='latin1')
            except Exception as e:
                raise ValueError(f"Could not load {self.dataset_path} with any encoding: {str(e)}")

        print("\n✅ Dataset loaded successfully!")  # Conferma il caricamento riuscito
        print(f"Shape: {self.df.shape}")  # Stampa la forma del DataFrame (righe, colonne)
        print(self.df.head())  # Stampa le prime 5 righe del DataFrame per verifica
        return self.df
