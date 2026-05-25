import os
import time
import warnings
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "3600"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")

from chronos import ChronosBoltPipeline, ChronosPipeline, Chronos2Pipeline
import timesfm
from tirex import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_CSV = os.path.join(BASE_DIR, "serietest.csv")

CONTEXT_LENGTH = 150
NUM_PREVISIONI = 10
DEVICE = "cpu"

REPO_HUGGINGFACE = {
    "chronos-bolt-tiny": "amazon/chronos-bolt-tiny",
    "chronos-bolt-mini": "amazon/chronos-bolt-mini",
    "chronos-bolt-small": "amazon/chronos-bolt-small",
    "chronos-bolt-base": "amazon/chronos-bolt-base",
    "chronos-t5-tiny": "amazon/chronos-t5-tiny",
    "chronos-2": "amazon/chronos-2",
    "timesfm-1.0-200m-pytorch": "google/timesfm-1.0-200m-pytorch",
    "TiRex": "NX-AI/TiRex"
}

class BenchmarkDataset(Dataset):
    def __init__(self, file_path, context_length=150, num_samples=10):
        df = pd.read_csv(file_path)
        self.data = torch.tensor(df['Data'].values).float()
        self.context_length = context_length
        self.num_samples = num_samples

    def __len__(self): return self.num_samples
    def __getitem__(self, idx): return self.data[idx : idx + self.context_length]

dataset_test = BenchmarkDataset(FILE_CSV, context_length=CONTEXT_LENGTH, num_samples=NUM_PREVISIONI)
dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

def esegui_inferenza(nome_modello, modello, batch_x):
    if "chronos-2" in nome_modello:
        modello.predict(batch_x.unsqueeze(1), prediction_length=1)
    elif "chronos" in nome_modello:
        modello.predict(batch_x, prediction_length=1)
    elif "timesfm" in nome_modello:
        modello.forecast(list(batch_x.numpy()), freq=[0])
    elif "TiRex" in nome_modello:
        modello.forecast(context=batch_x, prediction_length=1)

if __name__ == "__main__":
    print("==================================================")
    print(" FASE 1: CARICAMENTO DI TUTTI I MODELLI IN MEMORIA")
    print("==================================================")
    
    modelli_caricati = {}
    
    for nome, repo_id in REPO_HUGGINGFACE.items():
        print(f" -> Caricamento: {nome}...")
        try:
            if nome == "chronos-2":
                modelli_caricati[nome] = Chronos2Pipeline.from_pretrained(repo_id, device_map=DEVICE)
            elif "chronos-t5" in nome:
                modelli_caricati[nome] = ChronosPipeline.from_pretrained(repo_id, device_map=DEVICE)
            elif "chronos-bolt" in nome:
                modelli_caricati[nome] = ChronosBoltPipeline.from_pretrained(repo_id, device_map=DEVICE)
            elif "timesfm" in nome:
                hparams = timesfm.TimesFmHparams(backend="cpu", per_core_batch_size=1, horizon_len=1, context_len=512)
                modelli_caricati[nome] = timesfm.TimesFm(
                    hparams=hparams, 
                    checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=repo_id)
                )
            elif "TiRex" in nome:
                modelli_caricati[nome] = load_model(repo_id, backend="torch")
        except Exception as e:
            print(f"[ERRORE] Impossibile caricare {nome}: {e}")

    print("\n==================================================")
    print(" FASE 2: AVVIO BENCHMARK SUI MODELLI CARICATI")
    print("==================================================")
    
    report_finali = []
    primo_batch = next(iter(dataloader_test)) # Usato per il warm-up

    for nome, modello in modelli_caricati.items():
        print(f"\n[TESTING] Modello: {nome.upper()}")
        tempi = []
        
        try:
            with torch.no_grad():
                esegui_inferenza(nome, modello, primo_batch)
        except Exception:
            pass

        for passo, batch_x in enumerate(dataloader_test, 1):
            start_step = time.time()
            
            with torch.no_grad():
                esegui_inferenza(nome, modello, batch_x)
                
            tempo_impiegato = time.time() - start_step
            tempi.append(tempo_impiegato)
            print(f"  Previsione {passo}/{NUM_PREVISIONI} | Tempo: {tempo_impiegato:.4f}s")
            
        media, mediana = np.mean(tempi), np.median(tempi)
        print(f"-> [RISULTATO] Media: {media:.4f}s | Mediana: {mediana:.4f}s")
        
        report_finali.append({
            "Modello": nome.upper(), 
            "Tempo Medio (s)": media, 
            "Tempo Mediano (s)": mediana
        })

    if report_finali:
        df_report = pd.DataFrame(report_finali).sort_values(by="Tempo Mediano (s)").reset_index(drop=True)
        
        print("\n" + "="*60)
        print(" CLASSIFICA FINALE VELOCITÀ MODELLI (Ordinata dal più veloce)")
        print("="*60)
        print(df_report.to_markdown(index=False, floatfmt=".4f"))
        print("="*60)
        
        percorso_output = os.path.join(BASE_DIR, "benchmark_velocita_risultati.csv")
        df_report.to_csv(percorso_output, index=False)
        print(f"\n[INFO] Risultati salvati in '{percorso_output}'")
    else:
        print("\n[INFO] Nessun modello elaborato con successo.")