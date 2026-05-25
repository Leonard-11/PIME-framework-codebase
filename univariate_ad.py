import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from chronos import ChronosBoltPipeline, ChronosPipeline, Chronos2Pipeline
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from vus import metrics
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
import timesfm
from tirex import load_model
import time

class TimeSeriesAnomalyDataset(Dataset):
    def __init__(self, time_series_data, context_length=150):
        self.data = torch.tensor(time_series_data).float()
        self.context_length = context_length

    def __len__(self):
        return len(self.data) - self.context_length

    def __getitem__(self, idx):
        x_storico = self.data[idx : idx + self.context_length]
        y_target = self.data[idx + self.context_length]
        return x_storico, y_target


def prepara_dati_zero_shot(folder_path, colonna_valori='Data', colonna_etichette='Label'):
    dizionari_dati = {}
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            
            dati_grezzi = df[colonna_valori].values
            vere_etichette = df[colonna_etichette].values
            
            dataset = TimeSeriesAnomalyDataset(dati_grezzi, context_length=150)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
            
            dizionari_dati[filename] = (dataloader, vere_etichette)
            
    print(f"Ho elaborato e preparato {len(dizionari_dati)} file CSV.")
    return dizionari_dati


def carica_modelli_zero_shot():
    modelli_pronti = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Sto caricando i modelli su: {device.upper()}")

    print("Caricamento di Chronos 2 in corso...")
    pipeline_chronos_2 = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2",
        device_map=device,
        torch_dtype=torch.bfloat16
    )
    modelli_pronti["chronos2_base"] = pipeline_chronos_2

    taglie_t5 = ["tiny"]
    for taglia in taglie_t5:
        nome_modello = f"chronos_t5_{taglia}"
        print(f"Caricamento di {nome_modello.upper()} in corso...")
        modelli_pronti[nome_modello] = ChronosPipeline.from_pretrained(
            f"amazon/chronos-t5-{taglia}",
            device_map=device,
            torch_dtype=torch.bfloat16
        )

    taglie_bolt = ["tiny", "mini", "small", "base"]
    for taglia in taglie_bolt:
        nome_modello = f"chronos_bolt_{taglia}"
        print(f"Caricamento di {nome_modello.upper()} in corso...")
        modelli_pronti[nome_modello] = ChronosBoltPipeline.from_pretrained(
            f"amazon/chronos-bolt-{taglia}",
            device_map=device,
            torch_dtype=torch.bfloat16 
        )

    print("Caricamento di Moirai 2.0-Small in corso...")
    moirai_module = Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small")
    moirai_module = moirai_module.to(device)
    moirai_module.eval()
    
    pipeline_moirai = Moirai2Forecast(
        module=moirai_module,
        prediction_length=1, 
        context_length=150,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0
    )
    pipeline_moirai = pipeline_moirai.to(device)
    pipeline_moirai.eval()
    modelli_pronti["moirai_small"] = pipeline_moirai
    
    print("Caricamento di TimesFM in corso...")
    hparams = timesfm.TimesFmHparams(
        backend="gpu" if device == "cuda" else "cpu",
        per_core_batch_size=32,
        horizon_len=1,
        context_len=512, 
    )
    checkpoint = timesfm.TimesFmCheckpoint(
        huggingface_repo_id="google/timesfm-1.0-200m-pytorch" 
    )
    tfm = timesfm.TimesFm(hparams=hparams, checkpoint=checkpoint)
    modelli_pronti["timesfm"] = tfm

    print("Caricamento di TiRex in corso...")
    backend_tirex = "cuda" if device == "cuda" else "torch"
    modello_tirex = load_model("NX-AI/TiRex", backend=backend_tirex)
    modelli_pronti["tirex"] = modello_tirex

    print("Modelli caricati e pronti per l'inferenza!")
    return modelli_pronti


def calcola_anomaly_scores(nome_modello, modello, dataloader, device="cuda"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    anomaly_scores = []
    
    with torch.no_grad():
        for batch_x, batch_y in tqdm(dataloader, desc=f"Valutando {nome_modello}"):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            if "chronos2" in nome_modello:
                batch_x_3d = batch_x.cpu().unsqueeze(1) 
                campioni_previsione = modello.predict(batch_x_3d, prediction_length=1)
                
                if isinstance(campioni_previsione, list):
                    if isinstance(campioni_previsione[0], torch.Tensor):
                        campioni_previsione = torch.stack(campioni_previsione).to(device)
                    else:
                        campioni_previsione = torch.tensor(np.array(campioni_previsione), dtype=torch.float32, device=device)
                
                if campioni_previsione.dim() == 4:
                    campioni_previsione = campioni_previsione.squeeze(1)
                    
                y_pred = torch.quantile(campioni_previsione, 0.5, dim=1).squeeze(-1)

            elif "chronos" in nome_modello:
                campioni_previsione = modello.predict(batch_x.cpu(), prediction_length=1)
                y_pred = torch.quantile(campioni_previsione, 0.5, dim=1).squeeze(-1)

            elif "moirai" in nome_modello:
                previsione = modello.predict(past_target=batch_x.cpu().numpy())
                if isinstance(previsione, np.ndarray):
                    mediana_np = np.median(previsione, axis=1).reshape(-1)
                    y_pred = torch.tensor(mediana_np, dtype=torch.float32, device=device)
                else:
                    y_pred = previsione.median.view(-1).to(device)

            elif "timesfm" in nome_modello:
                input_numpy = batch_x.cpu().numpy()
                frequenze = [0] * len(input_numpy)
                point_forecast, _ = modello.forecast(list(input_numpy), freq=frequenze)
                y_pred = torch.tensor(point_forecast[:, 0], dtype=torch.float32, device=device)

            elif "tirex" in nome_modello:
                quantiles, mean = modello.forecast(context=batch_x.to(device), prediction_length=1)
                if mean.dim() > 1:
                    y_pred = mean[:, 0].to(device)
                else:
                    y_pred = mean.to(device)

            mse_batch = (batch_y - y_pred.to(device)) ** 2
            anomaly_scores.extend(mse_batch.cpu().numpy().tolist())

    return np.array(anomaly_scores)


def calcola_metriche_finali(anomaly_scores, vere_etichette, sliding_window_size=150):
    auc_roc = roc_auc_score(vere_etichette, anomaly_scores)
    auc_pr = average_precision_score(vere_etichette, anomaly_scores)
    
    try:
        risultati_vus = metrics.get_metrics(
            anomaly_scores, 
            vere_etichette, 
            slidingWindow=sliding_window_size
        )
        vus_roc = risultati_vus.get('VUS_ROC', None)
        vus_pr = risultati_vus.get('VUS_PR', None)
        
    except Exception as e:
        print(f"Errore durante il calcolo del VUS: {e}")
        vus_roc = None
        vus_pr = None

    print("\n--- RISULTATI VALUTAZIONE ---")
    print(f"AUC-PR:  {auc_pr:.3f}")
    print(f"AUC-ROC: {auc_roc:.3f}")
    print(f"VUS-PR:  {vus_pr:.3f}" if vus_pr else "VUS-PR:  Non calcolato")
    print(f"VUS-ROC: {vus_roc:.3f}" if vus_roc else "VUS-ROC: Non calcolato")
    print("-----------------------------\n")
    
    return {
        "AUC-ROC": auc_roc,
        "AUC-PR": auc_pr,
        "VUS-ROC": vus_roc,
        "VUS-PR": vus_pr
    }


def stampa_tabelle_comparative(tutti_i_risultati):
    dati_piatti = []
    
    for nome_modello, risultati_dataset in tutti_i_risultati.items():
        for nome_dataset, metriche in risultati_dataset.items():
            riga = {
                "Modello": nome_modello.upper(),
                "Dataset": nome_dataset,
                "AUC-PR": metriche["AUC-PR"],
                "AUC-ROC": metriche["AUC-ROC"],
                "VUS-PR": metriche["VUS-PR"],
                "VUS-ROC": metriche["VUS-ROC"],
                "Tempo (s)": metriche.get("Tempo (s)", 0.0) 
            }
            dati_piatti.append(riga)
            
    df_dettaglio = pd.DataFrame(dati_piatti)
    print("\n\n" + "="*80)
    print(" TABELLA 1: DETTAGLIO PER SINGOLO DATASET")
    print("="*80)
    print(df_dettaglio.to_markdown(index=False, floatfmt=".3f"))
    df_dettaglio.to_csv("risultati_dettagliati.csv", index=False)
    
    df_aggregato = df_dettaglio.groupby("Modello").mean(numeric_only=True).reset_index()
    df_aggregato = df_aggregato[["Modello", "AUC-PR", "AUC-ROC", "VUS-PR", "VUS-ROC", "Tempo (s)"]]
    
    print("\n\n" + "="*80)
    print(" TABELLA 2: MEDIE GLOBALI E TEMPO MEDIO PER DATASET")
    print("="*80)
    print(df_aggregato.to_markdown(index=False, floatfmt=".3f"))
    df_aggregato.to_csv("risultati_aggregati_medi.csv", index=False)

from torch.utils.data import TensorDataset, DataLoader
import copy

class MetaAutoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=2):
        super(MetaAutoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.LeakyReLU(0.01)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_ae_from_scratch(input_dim, bottleneck_dim, data, max_epochs=1000, lr=0.005, batch_size=32, patience=15):
    """Addestra il modello stampando la loss, con Early Stopping e Validation Split."""
    
    n_samples = len(data)
    val_size = max(int(n_samples * 0.2), 1)
    
    train_data = data[:-val_size]
    val_data = data[-val_size:]
    
    train_dataset = TensorDataset(torch.FloatTensor(train_data))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_tensor = torch.FloatTensor(val_data)
    
    model = MetaAutoencoder(input_dim, bottleneck_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    print("\n      --- Inizio Addestramento Autoencoder ---")
    
    for epoch in range(max_epochs):
        model.train()
        train_loss_accum = 0.0
        
        for batch_tuple in train_loader:
            batch_x = batch_tuple[0]
            optimizer.zero_grad()
            
            output = model(batch_x)
            
            loss = criterion(output, batch_x)
            loss.backward()
            optimizer.step()
            
            train_loss_accum += loss.item()
            
        avg_train_loss = train_loss_accum / len(train_loader)
            
        model.eval()
        with torch.no_grad():
            val_output = model(val_tensor)
            val_loss = criterion(val_output, val_tensor).item()
            
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"      [Epoca {epoch+1:03d}/{max_epochs}] Train Loss: {avg_train_loss:.5f} | Val Loss: {val_loss:.5f}")
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"      [!] Early Stopping attivato all'epoca {epoch+1}. Miglior Val Loss: {best_val_loss:.5f}")
            break

    if patience_counter < patience:
         print(f"      [!] Addestramento completato ({max_epochs} epoche). Miglior Val Loss: {best_val_loss:.5f}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    return model

if __name__ == "__main__":
    print("=== Avvio Pipeline Incrementale Zero-Shot ===")
    start_time = time.time()

    CARTELLA_DATI = "TSB-AD-U tuning"
    CARTELLA_OUTPUT_ENSEMBLE = "dati_per_ensemble"
    CONTEXT_LENGTH = 150
    
    MODELLI_SELEZIONATI = ["chronos_bolt_mini", "moirai_small", "timesfm", "tirex"] 

    os.makedirs(CARTELLA_OUTPUT_ENSEMBLE, exist_ok=True)

    print("\n--- Fase 1: Preparazione Dati ---")
    dizionari_dati = prepara_dati_zero_shot(CARTELLA_DATI)
    
    print("\n--- Fase 2: Caricamento Foundation Models ---")
    modelli_caricati = carica_modelli_zero_shot()
    
    tutti_i_risultati = {}
    tutti_gli_scores_grezzi = {nome_file: {} for nome_file in dizionari_dati.keys()}
    
    print("\n--- Fase 3: Esecuzione Incrementale ed Estrazione Metriche ---")

    for nome_file, (dataloader, vere_etichette) in dizionari_dati.items():
        print(f"\n--- Elaborazione Dataset: {nome_file} ---")
        
        nome_output = f"ensemble_data_{nome_file}"
        percorso_file_csv = os.path.join(CARTELLA_OUTPUT_ENSEMBLE, nome_output)
        
        df_esistente = None
        modelli_gia_calcolati = []
        
        if os.path.exists(percorso_file_csv):
            df_esistente = pd.read_csv(percorso_file_csv)
            modelli_gia_calcolati = [c for c in df_esistente.columns if c != 'True_Label']
            print(f"File trovato. Modelli già presenti: {modelli_gia_calcolati}")
            
            for m in modelli_gia_calcolati:
                tutti_gli_scores_grezzi[nome_file][m] = df_esistente[m].values
        else:
            print("Nessun salvataggio precedente trovato per questo file.")

        for nome_modello, modello in modelli_caricati.items():
            etichette_allineate = vere_etichette[CONTEXT_LENGTH:]
            
            if nome_modello in modelli_gia_calcolati:
                print(f"Skipping {nome_modello}: già calcolato. Recupero metriche...")
                anomaly_scores = tutti_gli_scores_grezzi[nome_file][nome_modello]
                tempo_impiegato = 0.0 
            else:
                print(f"Esecuzione inferenza per modello MANCANTE: {nome_modello}")
                inizio_inferenza = time.time()
                anomaly_scores = calcola_anomaly_scores(nome_modello, modello, dataloader)
                fine_inferenza = time.time()
                tempo_impiegato = fine_inferenza - inizio_inferenza
                
                tutti_gli_scores_grezzi[nome_file][nome_modello] = anomaly_scores
            
            STEP_BURN_IN = 512 
            
            if len(anomaly_scores) > STEP_BURN_IN:
                scores_per_valutazione = anomaly_scores[STEP_BURN_IN:]
                etichette_per_valutazione = etichette_allineate[STEP_BURN_IN:]
                
                metriche = calcola_metriche_finali(scores_per_valutazione, etichette_per_valutazione, CONTEXT_LENGTH)
                metriche["Tempo (s)"] = tempo_impiegato
                
                if nome_modello not in tutti_i_risultati:
                    tutti_i_risultati[nome_modello] = {}
                tutti_i_risultati[nome_modello][nome_file] = metriche
            else:
                print(f" [SKIP] Serie troppo corta ({len(anomaly_scores)} step) per togliere il burn-in.")

        print(f"Aggiornamento file CSV: {percorso_file_csv}")
        df_nuovo = pd.DataFrame(tutti_gli_scores_grezzi[nome_file])
        df_nuovo['True_Label'] = vere_etichette[CONTEXT_LENGTH:]
        df_nuovo.to_csv(percorso_file_csv, index=False)

    print("\n--- Fase 3.5: Calcolo ENSEMBLE (Autoencoder sui modelli correnti) ---")
    nome_ensemble = "ensemble_autoencoder"
    risultati_ensemble = {}
    
    STEP_BURN_IN = 512
    BOTTLENECK_DIM = 2

    for nome_file, (dataloader, vere_etichette) in dizionari_dati.items():
        print(f"Generando Meta-Ensemble (Autoencoder) per: {nome_file}...")
        inizio_ensemble = time.time()
        
        lista_scores = []
        
        modelli_da_usare = MODELLI_SELEZIONATI if MODELLI_SELEZIONATI else tutti_gli_scores_grezzi[nome_file].keys()
        
        modelli_effettivamente_usati = []
        for m_nome in modelli_da_usare:
            if m_nome in tutti_gli_scores_grezzi[nome_file]:
                lista_scores.append(tutti_gli_scores_grezzi[nome_file][m_nome])
                modelli_effettivamente_usati.append(m_nome)
            else:
                print(f"  [AVVISO] Il modello '{m_nome}' non è presente in questo dataset e verrà ignorato.")
                
        if not lista_scores:
            print("  [SKIP] Nessun modello valido per creare l'ensemble.")
            continue
            
        print(f"  Modelli usati per l'ensemble: {modelli_effettivamente_usati}")
        
        X_scores = np.vstack(lista_scores).T 
        n_steps = X_scores.shape[0]
        
        if n_steps <= STEP_BURN_IN:
            print(f"  [SKIP] Dati insufficienti: {n_steps} righe trovate, ne servono > {STEP_BURN_IN}.")
            continue
        
        X_train_raw = X_scores[:STEP_BURN_IN]
        X_test_raw = X_scores[STEP_BURN_IN:]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)
        
        input_dim = X_scores.shape[1]
        model_ae = train_ae_from_scratch(
            input_dim=input_dim,
            bottleneck_dim=BOTTLENECK_DIM,
            data=X_train_scaled
        )
        
        model_ae.eval()
        with torch.no_grad():
            reconstructed_test = model_ae(torch.FloatTensor(X_test_scaled)).numpy()
            ensemble_anomaly_scores = np.mean((X_test_scaled - reconstructed_test)**2, axis=1)
            
        tempo_ensemble = time.time() - inizio_ensemble
        
        etichette_allineate = vere_etichette[CONTEXT_LENGTH:] 
        etichette_test = etichette_allineate[STEP_BURN_IN:]
        
        metriche_ensemble = calcola_metriche_finali(
            ensemble_anomaly_scores, 
            etichette_test, 
            CONTEXT_LENGTH
        )
        metriche_ensemble["Tempo (s)"] = tempo_ensemble
        risultati_ensemble[nome_file] = metriche_ensemble

    tutti_i_risultati[nome_ensemble] = risultati_ensemble

    print("\n--- Fase 4: Generazione Tabelle Finali ---")
    stampa_tabelle_comparative(tutti_i_risultati)
    
    tempo_totale = (time.time() - start_time) / 60
    print(f"\n⏱️ ESECUZIONE COMPLETATA IN: {tempo_totale:.2f} minuti")