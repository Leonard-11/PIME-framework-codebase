import os
import time  
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from scipy.stats import gmean
from chronos import ChronosBoltPipeline, Chronos2Pipeline, ChronosPipeline
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
import timesfm
from tirex import load_model

def calcola_scale_factor(data_prior, m):
    """Calcola l'errore stagionale (a_n) dai dati precedenti al periodo di valutazione."""
    if len(data_prior) <= m:
        return 1.0 
    errors = np.abs(data_prior[m:] - data_prior[:-m])
    scale = np.mean(errors)
    return scale if scale > 0 else 1.0

def previsione_seasonal_naive(y_train, h, m):
    """Genera la previsione futura della baseline: copia il valore della stagione precedente."""
    last_season = y_train[-m:]
    repeats = int(np.ceil(h / m))
    return np.tile(last_season, repeats)[:h]

def calcola_mase_per_finestra(y_test_window, y_pred_window, scale_factor):
    """Calcola il MASE per una singola finestra."""
    return np.mean(np.abs(y_test_window - y_pred_window)) / scale_factor


def prepara_dati_benchmark(cartelle_dict, target_column, context_length, prediction_length):
    serie_da_testare = {}
    step_size = prediction_length 
    
    for folder_path, m_corrente in cartelle_dict.items():
        print(f"\nScansione cartella '{folder_path}' (Stagionalità impostata = {m_corrente})...")
        
        if not os.path.exists(folder_path):
            print(f"⚠️ ATTENZIONE: La cartella '{folder_path}' non esiste. Verrà saltata.")
            continue
            
        for filename in os.listdir(folder_path):
            if filename.endswith((".csv", ".parquet")):
                file_path = os.path.join(folder_path, filename)
                df = pd.read_csv(file_path) if filename.endswith(".csv") else pd.read_parquet(file_path)
                
                if target_column not in df.columns:
                    continue
                    
                dati_grezzi = df[target_column].values
                L = len(dati_grezzi)
                
                if L < context_length + prediction_length:
                    print(f"⚠️ Serie {filename} troppo corta. Saltata.")
                    continue
                    
                windows_x = []
                windows_y = []
                
                start_idx = 0
                while start_idx + context_length + prediction_length <= L:
                    x = dati_grezzi[start_idx : start_idx + context_length]
                    y = dati_grezzi[start_idx + context_length : start_idx + context_length + prediction_length]
                    windows_x.append(x)
                    windows_y.append(y)
                    start_idx += step_size 
                    
                train_iniziale = dati_grezzi[:context_length]
                
                nome_univoco_serie = f"{os.path.basename(folder_path)}_{filename}"
                
                serie_da_testare[nome_univoco_serie] = {
                    "windows_x": np.array(windows_x), 
                    "windows_y": np.array(windows_y), 
                    "train_iniziale": train_iniziale,
                    "stagionalita": m_corrente
                }
                
    print(f"\n✅ Preparate {len(serie_da_testare)} serie in totale (Non-Overlapping Windows).")
    return serie_da_testare

def carica_modelli_zero_shot(context_length, prediction_length):
    modelli_pronti = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nCaricamento modelli in corso su: {device.upper()}")

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

    print("Caricamento di CHRONOS2_BASE in corso...")
    pipeline_chronos_2 = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2",
        device_map=device,
        torch_dtype=torch.bfloat16
    )
    modelli_pronti["chronos2_base"] = pipeline_chronos_2

    print("Caricamento di MOIRAI in corso...")
    moirai_module = Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small")
    pipeline_moirai = Moirai2Forecast(
        module=moirai_module.to(device).eval(),
        prediction_length=prediction_length, 
        context_length=context_length,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0
    ).to(device).eval()
    modelli_pronti["moirai_small"] = pipeline_moirai
    
    print("Caricamento di TIMESFM in corso...")
    hparams = timesfm.TimesFmHparams(
        backend="gpu" if device == "cuda" else "cpu",
        per_core_batch_size=32,
        horizon_len=prediction_length,
        context_len=512, 
    )
    checkpoint = timesfm.TimesFmCheckpoint(huggingface_repo_id="google/timesfm-1.0-200m-pytorch")
    tfm = timesfm.TimesFm(hparams=hparams, checkpoint=checkpoint)
    modelli_pronti["timesfm"] = tfm

    print("Caricamento di TIREX in corso...")
    modello_tirex = load_model("NX-AI/TiRex", backend="cuda" if device == "cuda" else "torch")
    modelli_pronti["tirex"] = modello_tirex

    print("✅ Tutti i modelli sono stati caricati in memoria.")
    return modelli_pronti

def valuta_modelli_fev_bench(modelli, serie_dati, prediction_length, batch_size=32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    risultati_mase = {"series_id": list(serie_dati.keys())}
    tempi_esecuzione = {} 
    
    print("\n[Baseline] Calcolo Seasonal Naive...")
    start_base = time.time()
    mase_baseline = []
    
    for nome_serie, dati in serie_dati.items():
        m = dati["stagionalita"]
        scale = calcola_scale_factor(dati["train_iniziale"], m)
        mase_windows = []
        for idx, x_finestra in enumerate(dati["windows_x"]):
            y_true = dati["windows_y"][idx]
            y_pred = previsione_seasonal_naive(x_finestra, prediction_length, m)
            mase_windows.append(calcola_mase_per_finestra(y_true, y_pred, scale))
        mase_baseline.append(np.mean(mase_windows))
        
    tempi_esecuzione["Seasonal_Naive"] = time.time() - start_base
    risultati_mase["Seasonal_Naive"] = mase_baseline

    for nome_modello, modello in modelli.items():
        print(f"\n>>>>> Inferenza in corso: {nome_modello.upper()} <<<<<")
        start_modello = time.time() 
        mase_modello = []
        
        for nome_serie in tqdm(serie_dati.keys()):
            dati = serie_dati[nome_serie]
            
            x_ctx_full = torch.tensor(dati["windows_x"]).float() 
            y_test_matrice = dati["windows_y"]
            
            m = dati["stagionalita"]
            scale = calcola_scale_factor(dati["train_iniziale"], m)
            num_finestre = x_ctx_full.shape[0]
            
            tutte_le_predizioni = []
            
            with torch.no_grad():
                
                for i in range(0, num_finestre, batch_size):
                    
                    batch_x = x_ctx_full[i : i + batch_size]
                    
                    batch_x_gpu = batch_x.to(device) if "tirex" in nome_modello else batch_x
                    
                    
                    if "chronos2" in nome_modello or "chronos" in nome_modello:
                        
                        x_input = batch_x.unsqueeze(1) if "chronos2" in nome_modello else batch_x
                        
                        campioni = modello.predict(x_input, prediction_length=prediction_length)
                        if isinstance(campioni, list):
                            campioni = torch.stack(campioni).to(device)
                        if campioni.dim() == 4:
                            campioni = campioni.squeeze(1)
                        y_pred_batch = torch.quantile(campioni, 0.5, dim=1).cpu().numpy()


                    elif "moirai" in nome_modello:
                        previsione = modello.predict(past_target=batch_x.numpy())
                        
                        if isinstance(previsione, np.ndarray):
                            if previsione.ndim == 3:
                                y_pred_batch = np.median(previsione, axis=1)
                            else:
                                y_pred_batch = previsione
                                
                        elif hasattr(previsione, 'samples'):
                            samples = previsione.samples
                            if torch.is_tensor(samples):
                                samples = samples.cpu().numpy()
                            y_pred_batch = np.median(samples, axis=1)
                            
                        else:
                            med = previsione.median
                            if torch.is_tensor(med):
                                med = med.cpu().numpy()
                            y_pred_batch = med


                    elif "timesfm" in nome_modello:
                        freq_list = [0] * batch_x.shape[0] 
                        point_forecast, _ = modello.forecast(list(batch_x.numpy()), freq=freq_list)
                        y_pred_batch = point_forecast


                    elif "tirex" in nome_modello:
                        
                        _, mean = modello.forecast(context=batch_x_gpu, prediction_length=prediction_length)
                        y_pred_batch = mean.cpu().numpy()
                        
                    tutte_le_predizioni.append(y_pred_batch)


            y_pred_completo = np.concatenate(tutte_le_predizioni, axis=0)


            mase_windows = []
            for w in range(num_finestre):
                mase_w = calcola_mase_per_finestra(y_test_matrice[w], y_pred_completo[w], scale)
                mase_windows.append(mase_w)
            
            mase_modello.append(np.mean(mase_windows))
            
            
            if device == "cuda":
                torch.cuda.empty_cache()
            
        tempi_esecuzione[nome_modello] = time.time() - start_modello 
        risultati_mase[nome_modello] = mase_modello

    return pd.DataFrame(risultati_mase), tempi_esecuzione

def calcola_tabelle_benchmark(df_scores, tempi_modelli, baseline_col="Seasonal_Naive"):
    modelli = [col for col in df_scores.columns if col not in [baseline_col, "series_id"]]
    tutti_i_competitor = modelli + [baseline_col]
    
    matrice_punteggi = df_scores[tutti_i_competitor].values
    col_idx = {nome: i for i, nome in enumerate(tutti_i_competitor)}
    idx_base = col_idx[baseline_col]
    
    R = len(df_scores) 
    M = len(tutti_i_competitor) 
    
    tabella_finale = []
    

    for mod in tutti_i_competitor:
        idx_mod = col_idx[mod]
        punteggi_mod = matrice_punteggi[:, idx_mod]
        punteggi_base = matrice_punteggi[:, idx_base]
        
        
        mase_medio = np.mean(punteggi_mod)
        
        
        if mod == baseline_col:
            skill_score = 0.0 
        else:
            rapporto_clipped = np.clip(punteggi_mod / punteggi_base, 0.01, 100)
            geom_mean = gmean(rapporto_clipped)
            skill_score = 1 - geom_mean
        
        
        vittorie_totali = 0
        for r in range(R):
            for k in range(M):
                if k == idx_mod:
                    continue
                e_rj = matrice_punteggi[r, idx_mod]
                e_rk = matrice_punteggi[r, k]
                
                if e_rj < e_rk:
                    vittorie_totali += 1
                elif e_rj == e_rk:
                    vittorie_totali += 0.5
                    
        avg_win_rate = vittorie_totali / (R * (M - 1))
        
        tabella_finale.append({
            "Model": mod,
            "MASE Medio": mase_medio, 
            "Avg. Win Rate": avg_win_rate * 100, 
            "Skill Score": skill_score * 100,
            "Time (s)": tempi_modelli.get(mod, 0.0) 
        })
        
    df_finale = pd.DataFrame(tabella_finale).sort_values(by="Avg. Win Rate", ascending=False)
    return df_finale


if __name__ == "__main__":
    start_script_totale = time.time() 
    
    print("="*60)
    print(" 🚀 AVVIO PIPELINE BENCHMARK FEV-BENCH (DOMINIO ENERGY) ")
    print("="*60)
    
    COLONNA_TARGET = "target" 
    CONTEXT_LENGTH = 150 
    PREDICTION_LENGTH = 4 
    
    CARTELLE_DA_ANALIZZARE = {
        "Dataset forecasting/stagionalita_24": 24,
        "Dataset forecasting/stagionalita_96": 96
    }


    serie_da_testare = prepara_dati_benchmark(
        cartelle_dict=CARTELLE_DA_ANALIZZARE,
        target_column=COLONNA_TARGET, 
        context_length=CONTEXT_LENGTH, 
        prediction_length=PREDICTION_LENGTH
    )
    
    
    modelli_pronti = carica_modelli_zero_shot(CONTEXT_LENGTH, PREDICTION_LENGTH)
    
    
    df_mase, tempi_esecuzione = valuta_modelli_fev_bench(modelli_pronti, serie_da_testare, PREDICTION_LENGTH, batch_size=32)
    
    
    df_mase.to_csv("mase_per_dataset.csv", index=False)
    print("\n✅ FILE SALVATO: 'mase_per_dataset.csv' (Contiene il MASE di ogni modello per ogni dataset).")
    
    
    tab_finale = calcola_tabelle_benchmark(df_mase, tempi_esecuzione, baseline_col="Seasonal_Naive")
    
    
    tab_finale.to_csv("metriche_riassuntive.csv", index=False)
    print("✅ FILE SALVATO: 'metriche_riassuntive.csv' (Contiene MASE Medio, Win Rate, Skill Score e Tempi).")
    
    
    print("\n" + "="*80)
    print(" TABELLA FINALE: METRICHE RIASSUNTIVE")
    print("="*80)
    print(tab_finale.to_markdown(index=False, floatfmt=".3f"))
    
    
    fine_script_totale = time.time()
    tempo_trascorso = fine_script_totale - start_script_totale
    ore = int(tempo_trascorso // 3600)
    minuti = int((tempo_trascorso % 3600) // 60)
    secondi = tempo_trascorso % 60
    
    print("\n" + "="*80)
    print(f"🏁 Esperimento concluso con successo!")
    print(f"⏱️ Tempo totale di esecuzione: {ore}h {minuti}m {secondi:.2f}s")
    print("="*80)