import os
import time
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score
from tirex import load_model


from vus import metrics


from chronos import ChronosBoltPipeline
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
import timesfm


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

def estrai_prima_componente_principale(file_features_path, cartella_labels, burn_in_steps=662):
    """
    Legge i file .txt multivariati, scala i dati e applica la PCA 
    restituendo SOLO la Prima Componente Principale (PC1).
    """
    filename = os.path.basename(file_features_path)
    df_features = pd.read_csv(file_features_path, header=None)
    df_features = df_features.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    dati_raw = df_features.values
    
    label_path = os.path.join(cartella_labels, filename)
    if not os.path.exists(label_path):
        return None, None
        
    df_labels = pd.read_csv(label_path, header=None)
    vere_etichette = df_labels[0].values
    
    if len(dati_raw) != len(vere_etichette) or len(dati_raw) <= burn_in_steps:
        return None, None


    dati_burn_in = dati_raw[:burn_in_steps]
    std_devs = np.std(dati_burn_in, axis=0)
    colonne_valide_idx = np.where(std_devs > 1e-6)[0]
    
    if len(colonne_valide_idx) == 0:
        return None, None
        
    dati_raw_filtrati = dati_raw[:, colonne_valide_idx]
    dati_burn_in_filtrati = dati_raw_filtrati[:burn_in_steps]
    
    scaler = StandardScaler()
    burn_in_scaled = scaler.fit_transform(dati_burn_in_filtrati)
    dati_totali_scaled = scaler.transform(dati_raw_filtrati)
    
    
    pca = PCA(n_components=1)
    pca.fit(burn_in_scaled)
    
    pc1_series = pca.transform(dati_totali_scaled).flatten()
    
    return pc1_series, vere_etichette

def carica_modelli_selezionati():
    modelli_pronti = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nCaricamento dei Foundation Models su: {device.upper()}")

    print(" 1/4 Caricamento di CHRONOS_BOLT_MINI in corso...")
    modelli_pronti["chronos_bolt"] = ChronosBoltPipeline.from_pretrained(
        "amazon/chronos-bolt-mini", device_map=device, torch_dtype=torch.bfloat16 
    )

    print(" 2/4 Caricamento di MOIRAI_SMALL in corso...")
    moirai_module = Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small").to(device)
    moirai_module.eval()
    modelli_pronti["moirai"] = Moirai2Forecast(
        module=moirai_module, prediction_length=1, context_length=150,
        target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0
    ).to(device)
    modelli_pronti["moirai"].eval()

    print(" 3/4 Caricamento di TIMESFM in corso...")
    hparams = timesfm.TimesFmHparams(
        backend="gpu" if device == "cuda" else "cpu",
        per_core_batch_size=64, 
        horizon_len=1,
        context_len=160, 
    )
    checkpoint = timesfm.TimesFmCheckpoint(huggingface_repo_id="google/timesfm-1.0-200m-pytorch")
    modelli_pronti["timesfm"] = timesfm.TimesFm(hparams=hparams, checkpoint=checkpoint)


    print(" 4/4 Caricamento di TiRex in corso...")
    backend_tirex = "cuda" if device == "cuda" else "torch"
    modello_tirex = load_model("NX-AI/TiRex", backend=backend_tirex)
    modelli_pronti["tirex"] = modello_tirex

    return modelli_pronti

def calcola_anomaly_scores(nome_modello, modello, dataloader, device="cuda"):
    anomaly_scores = []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            if "chronos" in nome_modello:
                campioni_previsione = modello.predict(batch_x, prediction_length=1)
                y_pred = torch.quantile(campioni_previsione, 0.5, dim=1).squeeze(-1)
                
            elif "moirai" in nome_modello:
                previsione = modello.predict(past_target=batch_x.cpu().numpy())
                if isinstance(previsione, np.ndarray):
                    y_pred = torch.tensor(np.median(previsione, axis=1).reshape(-1), dtype=torch.float32, device=device)
                else:
                    y_pred = previsione.median.view(-1).to(device)
                    
            elif "timesfm" in nome_modello:
                batch_x_np = batch_x.cpu().numpy()
                freq_list = [0] * batch_x_np.shape[0] 
                point_forecast, _ = modello.forecast(list(batch_x_np), freq=freq_list)
                y_pred = torch.tensor(point_forecast[:, 0], dtype=torch.float32, device=device)

            elif "tirex" in nome_modello:
                quantiles, mean = modello.forecast(context=batch_x.to(device), prediction_length=1)
                if mean.dim() > 1:
                    y_pred = mean[:, 0].to(device)
                else:
                    y_pred = mean.to(device)

            y_pred=y_pred.to(device)


            mse_batch = (batch_y - y_pred) ** 2
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
    except Exception:
        vus_roc = None
        vus_pr = None

    return {
        "AUC-ROC": auc_roc,
        "AUC-PR": auc_pr,
        "VUS-ROC": vus_roc,
        "VUS-PR": vus_pr
    }

class MetaAutoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=2):
        super(MetaAutoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, bottleneck_dim), nn.LeakyReLU(0.01))
        self.decoder = nn.Sequential(nn.Linear(bottleneck_dim, input_dim))

    def forward(self, x):
        return self.decoder(self.encoder(x))

def train_ae_from_scratch(input_dim, bottleneck_dim, data, max_epochs=1000, lr=0.005, batch_size=32, patience=15):
    n_samples = len(data)
    val_size = max(int(n_samples * 0.2), 1)
    
    train_data, val_data = data[:-val_size], data[-val_size:]
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(train_data)), batch_size=batch_size, shuffle=True)
    val_tensor = torch.FloatTensor(val_data)
    
    model = MetaAutoencoder(input_dim, bottleneck_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    best_val_loss, patience_counter, best_model_state = float('inf'), 0, None
    
    for epoch in range(max_epochs):
        model.train()
        for batch_tuple in train_loader:
            batch_x = batch_tuple[0]
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_x)
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(val_tensor), val_tensor).item()
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            
        if patience_counter >= patience: break

    if best_model_state: model.load_state_dict(best_model_state)
    return model

if __name__ == "__main__":
    print("=== MULTI-MODEL ZERO-SHOT & AE ENSEMBLE (PC1 UNIVARIATA) ===")
    start_time = time.time()

    CARTELLA_TEST_FEATURES = "SMD/test"       
    CARTELLA_TEST_LABELS = "SMD/test_label"   
    
    CONTEXT_STEPS = 150       
    CALIBRATION_STEPS = 512   
    PCA_BURN_IN = CONTEXT_STEPS + CALIBRATION_STEPS  # 662
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    modelli_caricati = carica_modelli_selezionati()
    
    tutti_i_risultati = []

    if not os.path.exists(CARTELLA_TEST_FEATURES):
        print("Errore: Cartella mancante.")
    else:
        file_list = [f for f in os.listdir(CARTELLA_TEST_FEATURES) if f.endswith('.txt')]
        print(f"\nTrovati {len(file_list)} file.\n")
        
        for filename in file_list:
            file_features_path = os.path.join(CARTELLA_TEST_FEATURES, filename)
            print(f"\n" + "="*70)
            print(f" Dataset: {filename}")
            
            pc1_series, vere_etichette = estrai_prima_componente_principale(
                file_features_path, CARTELLA_TEST_LABELS, burn_in_steps=PCA_BURN_IN
            )
            
            if pc1_series is None: 
                print(" -> Saltato (Dati invalidi o troppo corti)")
                continue
            
            print("  > PC1 Estratta con successo.")

            dataset = TimeSeriesAnomalyDataset(pc1_series, context_length=CONTEXT_STEPS)
            dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
            
            scores_modelli = {}
            
            for nome_modello, modello in modelli_caricati.items():
                t_inizio = time.time()
                anomaly_scores = calcola_anomaly_scores(nome_modello, modello, dataloader, device)
                scores_modelli[nome_modello] = anomaly_scores
                
                
                etichette_test = vere_etichette[PCA_BURN_IN:]
                scores_test = anomaly_scores[CALIBRATION_STEPS:] 
                
                metriche = calcola_metriche_finali(scores_test, etichette_test, CONTEXT_STEPS)
                metriche.update({
                    "Dataset": filename,
                    "Modello": nome_modello.upper(),
                    "Tempo (s)": time.time() - t_inizio
                })
                tutti_i_risultati.append(metriche)

            print("  > Addestramento Autoencoder Ensemble...")
            t_inizio_ae = time.time()
            
            X_scores = np.vstack(list(scores_modelli.values())).T
            
            X_train_raw = X_scores[:CALIBRATION_STEPS]
            X_test_raw = X_scores[CALIBRATION_STEPS:]
            
            scaler_ae = StandardScaler()
            X_train_scaled = scaler_ae.fit_transform(X_train_raw)
            X_test_scaled = scaler_ae.transform(X_test_raw)
            
            input_dim = X_scores.shape[1]
            bottleneck_dim = 2
            
            ae_model = train_ae_from_scratch(input_dim, bottleneck_dim, X_train_scaled, max_epochs=1000)
            
            ae_model.eval()
            with torch.no_grad():
                reconstructed_test = ae_model(torch.FloatTensor(X_test_scaled)).numpy()
                ensemble_anomaly_scores = np.mean((X_test_scaled - reconstructed_test)**2, axis=1)
                
            metriche_ensemble = calcola_metriche_finali(ensemble_anomaly_scores, etichette_test, CONTEXT_STEPS)
            metriche_ensemble.update({
                "Dataset": filename,
                "Modello": "AUTOENCODER_ENSEMBLE",
                "Tempo (s)": time.time() - t_inizio_ae
            })
            tutti_i_risultati.append(metriche_ensemble)

        if tutti_i_risultati:
            df = pd.DataFrame(tutti_i_risultati)
            
            print("\n\n" + "="*90)
            print(" LEADERBOARD FINALE: MEDIE GLOBALI SUI DATASET")
            print("="*90)
            
            cols = ["Modello", "AUC-PR", "AUC-ROC", "VUS-PR", "VUS-ROC", "Tempo (s)"]
            df_aggregato = df.groupby("Modello").mean(numeric_only=True).reset_index()
            df_aggregato = df_aggregato[cols].sort_values(by="AUC-PR", ascending=False)
            
            print(df_aggregato.to_markdown(index=False, floatfmt=".3f"))
            df_aggregato.to_csv("risultati_leaderboard_PC1.csv", index=False)
            df.to_csv("risultati_dettaglio_PC1.csv", index=False)
            
    print(f"\n⏱️ ESECUZIONE COMPLETATA IN: {(time.time() - start_time) / 60:.2f} minuti")