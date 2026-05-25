import os
import time
import threading
import queue
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import warnings
warnings.filterwarnings("ignore")

from chronos import ChronosBoltPipeline
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
import timesfm
import onnxruntime as ort
from huggingface_hub import hf_hub_download


def safe_extract(val):
    """Estrae un singolo float da qualsiasi output dei modelli (array, tensor, scalare)."""
    val = np.array(val)
    if val.ndim == 0:
        return float(val)
    return float(val.flatten()[0])


FILE_SERIE_GREZZA = "machine-1-1.txt"
FILE_LABEL = "machine-1-1_label.txt"
FILE_TRAIN_PRECALCOLATO = "train_precalcolati.csv"

CONTEXT_STEPS = 150
CALIBRATION_STEPS = 512
PCA_BURN_IN = CONTEXT_STEPS + CALIBRATION_STEPS  # 662

SIMULATION_SCENARIOS = {
    "1. Normal Phase (Right after training)": PCA_BURN_IN + 1,
    "2. Anomaly 1 Simulation": 15840,
    "3. Anomaly 2 Simulation": 16955,
    "4. Anomaly 3 Simulation": 18062
}

class MetaAutoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=2):
        super(MetaAutoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, bottleneck_dim), nn.LeakyReLU(0.01))
        self.decoder = nn.Sequential(nn.Linear(bottleneck_dim, input_dim))

    def forward(self, x):
        return self.decoder(self.encoder(x))

def carica_foundation_models():
    print("\n[SYSTEM] Loading Foundation Models on CPU (Strict Offline Mode)...")
    start_total = time.time()
    modelli = {}
    device = "cpu"
    
    print(" 1/4 Loading CHRONOS_BOLT_MINI...")
    t0 = time.time()
    modelli["chronos_bolt"] = ChronosBoltPipeline.from_pretrained("amazon/chronos-bolt-mini", device_map=device, local_files_only=True)
    print(f"     -> Done in {time.time() - t0:.2f} seconds")
    
    print(" 2/4 Loading MOIRAI_SMALL...")
    t0 = time.time()
    moirai_module = Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small", local_files_only=True).to(device).eval()
    modelli["moirai"] = Moirai2Forecast(module=moirai_module, prediction_length=1, context_length=CONTEXT_STEPS, target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0).to(device).eval()
    print(f"     -> Done in {time.time() - t0:.2f} seconds")
    
    print(" 3/4 Loading TIMESFM...")
    t0 = time.time()
    hparams = timesfm.TimesFmHparams(backend="cpu", per_core_batch_size=1, horizon_len=1, context_len=160)
    checkpoint = timesfm.TimesFmCheckpoint(huggingface_repo_id="google/timesfm-1.0-200m-pytorch")
    modelli["timesfm"] = timesfm.TimesFm(hparams=hparams, checkpoint=checkpoint)
    print(f"     -> Done in {time.time() - t0:.2f} seconds")
    
    print(" 4/4 Loading TIREX (ONNX)...")
    t0 = time.time()
    percorso_onnx = hf_hub_download(repo_id="NX-AI/TiRex", filename="tirex.onnx", local_files_only=True)
    modelli["tirex"] = ort.InferenceSession(percorso_onnx, providers=['CPUExecutionProvider'])
    print(f"     -> Done in {time.time() - t0:.2f} seconds")
    
    print(f"[SYSTEM] All models ready! Total loading time: {time.time() - start_total:.2f} seconds\n")
    return modelli

def calcola_predizioni_step(modelli, contesto_pca):
    with torch.no_grad():
        x_torch = torch.tensor(contesto_pca).float().unsqueeze(0)
        y_chronos = safe_extract(torch.quantile(modelli["chronos_bolt"].predict(x_torch, prediction_length=1), 0.5, dim=1).squeeze())
        
        pred_timesfm, _ = modelli["timesfm"].forecast([contesto_pca], freq=[0])
        y_timesfm = safe_extract(pred_timesfm)
        
        x_numpy_tirex = np.array(contesto_pca, dtype=np.float32).reshape(1, -1)
        pred_tirex = modelli["tirex"].run(None, {modelli["tirex"].get_inputs()[0].name: x_numpy_tirex})
        y_tirex = safe_extract(pred_tirex[0])
        
        pred_moirai = modelli["moirai"].predict(past_target=np.array(contesto_pca).reshape(1, -1))
        y_moirai = safe_extract(pred_moirai.median) if hasattr(pred_moirai, 'median') else safe_extract(np.median(pred_moirai, axis=1))

    return [y_chronos, y_timesfm, y_tirex, y_moirai]

class AnomalyDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Industrial Simulator - Anomaly Detection")
        
        try:
            self.root.state('zoomed') 
        except:
            try:
                self.root.attributes('-zoomed', True) 
            except:
                self.root.geometry("1024x768") 
        
        self.dati_raw = None
        self.dati_label = None
        self.colonne_valide_idx = None
        
        self.modelli = None
        self.autoencoder = None
        
        self.pca_scaler = StandardScaler()
        self.pca_model = PCA(n_components=1)
        self.ae_scaler = StandardScaler()
        
        self.is_running_live = False
        self.queue_dati = queue.Queue()
        
        self.plot_tempi = []
        self.plot_anomaly_scores = []
        
        self._setup_ui()
        self._load_data()
        
        threading.Thread(target=self._init_models, daemon=True).start()

    def _setup_ui(self):
        frame_controlli = tk.Frame(self.root, bg="#2c3e50", pady=10)
        frame_controlli.pack(fill=tk.X)
        
        tk.Label(frame_controlli, text="PCA-Informed Multivariate Ensemble Framework (PIME)", font=("Arial", 16, "bold"), fg="white", bg="#2c3e50").pack(pady=5)
        
        frame_bottoni = tk.Frame(frame_controlli, bg="#2c3e50")
        frame_bottoni.pack()
        
        self.btn_train = tk.Button(frame_bottoni, text="1. Train Models", font=("Arial", 12), command=self.avvia_addestramento, state=tk.DISABLED)
        self.btn_train.pack(side=tk.LEFT, padx=10)
        
        self.combo_scenari = ttk.Combobox(frame_bottoni, values=list(SIMULATION_SCENARIOS.keys()), font=("Arial", 12), width=40, state="readonly")
        self.combo_scenari.current(0)
        self.combo_scenari.pack(side=tk.LEFT, padx=10)
        
        self.btn_live = tk.Button(frame_bottoni, text="2. Start Live", font=("Arial", 12), command=self.avvia_live, state=tk.DISABLED, bg="#27ae60", fg="white")
        self.btn_live.pack(side=tk.LEFT, padx=10)
        
        self.btn_stop = tk.Button(frame_bottoni, text="Stop", font=("Arial", 12), command=self.ferma_live, state=tk.DISABLED, bg="#c0392b", fg="white")
        self.btn_stop.pack(side=tk.LEFT, padx=10)
        
        self.frame_progresso = tk.Frame(frame_controlli, bg="#2c3e50")
        
        self.label_stato = tk.Label(self.frame_progresso, text="Initializing...", font=("Arial", 10, "italic"), fg="lightgray", bg="#2c3e50")
        self.label_stato.pack(pady=(5,0))
        
        self.progressbar = ttk.Progressbar(self.frame_progresso, orient="horizontal", length=400, mode="determinate")
        self.progressbar.pack(pady=5)
        
        self.fig = Figure(figsize=(10, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def aggiorna_stato_ui(self, messaggio, mostra_barra=True, mode="determinate", max_val=100, valore=0):
        """Gestisce in modo fluido barre di caricamento."""
        def update():
            self.label_stato.config(text=messaggio)
            if mostra_barra:
                self.frame_progresso.pack()
                self.progressbar.config(mode=mode, maximum=max_val)
                if mode == "indeterminate":
                    self.progressbar.start(10)
                else:
                    self.progressbar.stop()
                    self.progressbar['value'] = valore
            else:
                self.progressbar.stop()
                self.frame_progresso.pack_forget()
        self.root.after(0, update)

    def _load_data(self):
        try:
            df_grezzo = pd.read_csv(FILE_SERIE_GREZZA, header=None)
            df_grezzo = df_grezzo.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
            self.dati_raw = df_grezzo.values
            
            if os.path.exists(FILE_LABEL):
                self.dati_label = pd.read_csv(FILE_LABEL, header=None)[0].values
            else:
                self.dati_label = np.zeros(len(self.dati_raw))
            print(f"[SYSTEM] Data loaded successfully: {self.dati_raw.shape[0]} steps.")
        except Exception as e:
            messagebox.showerror("Data Error", f"Cannot read txt files:\n{e}")

    def _init_models(self):
        self.aggiorna_stato_ui("Loading AI models...", mode="indeterminate")
        try:
            self.modelli = carica_foundation_models()
            self.root.after(0, lambda: self.btn_train.config(state=tk.NORMAL))
            self.aggiorna_stato_ui("Models loaded. Ready for training.", mostra_barra=False)
        except Exception as e:
            self.aggiorna_stato_ui(f"Error loading models: {e}", mostra_barra=False)
            print(f"[ERROR] Failed to load models: {e}")
        
    def avvia_addestramento(self):
        self.btn_train.config(state=tk.DISABLED)
        threading.Thread(target=self._worker_addestramento, daemon=True).start()

    def _worker_addestramento(self):
        start_training = time.time()
        print("\n[TRAINING] Starting training pipeline...")
        
        self.aggiorna_stato_ui("Extracting features (PCA)...", mode="indeterminate")
        t0 = time.time()
        dati_burn_in = self.dati_raw[:PCA_BURN_IN]
        std_devs = np.std(dati_burn_in, axis=0)
        self.colonne_valide_idx = np.where(std_devs > 1e-6)[0]
        
        dati_burn_in_filtrati = self.dati_raw[:PCA_BURN_IN, self.colonne_valide_idx]
        burn_in_scaled = self.pca_scaler.fit_transform(dati_burn_in_filtrati)
        self.pca_model.fit(burn_in_scaled)
        print(f" -> PCA Feature Extraction completed in {time.time() - t0:.2f} seconds")
        
        if not os.path.exists(FILE_TRAIN_PRECALCOLATO):
            print("[TRAINING] Pre-calculating CSV file (this might take a while)...")
            t0 = time.time()
            dati_per_csv = []
            for i in range(CONTEXT_STEPS, PCA_BURN_IN):
                contesto_multi = self.dati_raw[i - CONTEXT_STEPS : i, self.colonne_valide_idx]
                valore_multi = self.dati_raw[i, self.colonne_valide_idx].reshape(1, -1)
                
                contesto_scaled = self.pca_scaler.transform(contesto_multi)
                valore_scaled = self.pca_scaler.transform(valore_multi)
                contesto_pca = self.pca_model.transform(contesto_scaled).flatten()
                vero_valore_pca = self.pca_model.transform(valore_scaled).flatten()[0]
                
                predizioni = calcola_predizioni_step(self.modelli, contesto_pca)
                dati_per_csv.append([vero_valore_pca] + predizioni)
                
                passi_fatti = i - CONTEXT_STEPS + 1
                if passi_fatti % 5 == 0:
                    self.aggiorna_stato_ui(f"Calibration Simulation: {passi_fatti}/{CALIBRATION_STEPS}", 
                                           mode="determinate", max_val=CALIBRATION_STEPS, valore=passi_fatti)

            df_creato = pd.DataFrame(dati_per_csv, columns=["Target_PC1", "Chronos", "TimesFM", "TiRex", "Moirai"])
            df_creato.to_csv(FILE_TRAIN_PRECALCOLATO, index=False)
            print(f" -> CSV Calculation completed in {time.time() - t0:.2f} seconds")

        try:
            self.aggiorna_stato_ui("Reading data and scaling...", mode="indeterminate")
            df_pre = pd.read_csv(FILE_TRAIN_PRECALCOLATO)
            mse_chronos = (df_pre['Target_PC1'] - df_pre['Chronos'])**2
            mse_timesfm = (df_pre['Target_PC1'] - df_pre['TimesFM'])**2
            mse_tirex = (df_pre['Target_PC1'] - df_pre['TiRex'])**2
            mse_moirai = (df_pre['Target_PC1'] - df_pre['Moirai'])**2
            
            matrice_mse = np.column_stack([mse_chronos, mse_timesfm, mse_tirex, mse_moirai])
            mse_scalati = self.ae_scaler.fit_transform(matrice_mse)
        except Exception as e:
            self.aggiorna_stato_ui("CSV Error.", mostra_barra=False)
            self.root.after(0, lambda: messagebox.showerror("Error", f"{e}"))
            return

        print("[TRAINING] Training MetaAutoencoder...")
        t0 = time.time()
        self.aggiorna_stato_ui("Autoencoder Training...", mode="determinate", max_val=200, valore=0)
        self.autoencoder = MetaAutoencoder(matrice_mse.shape[1], bottleneck_dim=2)
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=0.005)
        criterion = nn.MSELoss()
        
        x_tensor = torch.FloatTensor(mse_scalati)
        for epoch in range(200):
            optimizer.zero_grad()
            loss = criterion(self.autoencoder(x_tensor), x_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                self.aggiorna_stato_ui(f"Training Neural Network: Epoch {epoch}/200", 
                                       mode="determinate", max_val=200, valore=epoch)
                
        print(f" -> Autoencoder training completed in {time.time() - t0:.2f} seconds")
        print(f"[TRAINING] Total pipeline completed in {time.time() - start_training:.2f} seconds\n")
        
        self.autoencoder.eval()
        self.aggiorna_stato_ui("Training completed!", mostra_barra=False)
        self.root.after(0, lambda: self.btn_live.config(state=tk.NORMAL))

    def avvia_live(self):
        indice_partenza = SIMULATION_SCENARIOS[self.combo_scenari.get()]
        
        self.plot_tempi.clear()
        self.plot_anomaly_scores.clear()
        
        while not self.queue_dati.empty():
            try:
                self.queue_dati.get_nowait()
            except queue.Empty:
                break
        
        self.is_running_live = True
        self.btn_live.config(state=tk.DISABLED); self.btn_train.config(state=tk.DISABLED)
        self.combo_scenari.config(state=tk.DISABLED); self.btn_stop.config(state=tk.NORMAL)
        
        self.aggiorna_stato_ui("Live Inference...", mode="indeterminate")
        print(f"\n[LIVE] Starting live simulation from step {indice_partenza}...")
        threading.Thread(target=self._worker_live, args=(indice_partenza,), daemon=True).start()
        self._aggiorna_gui_periodicamente()

    def ferma_live(self):
        self.is_running_live = False
        self.btn_live.config(state=tk.NORMAL); self.btn_train.config(state=tk.NORMAL)
        self.combo_scenari.config(state="readonly"); self.btn_stop.config(state=tk.DISABLED)
        self.aggiorna_stato_ui("Inference Stopped.", mostra_barra=False)
        print("[LIVE] Simulation stopped by user.")

    def _worker_live(self, indice_partenza):
        indice_corrente = indice_partenza
        while self.is_running_live and indice_corrente < len(self.dati_raw):
            inizio_ciclo = time.time()
            
            contesto_multi = self.dati_raw[indice_corrente - CONTEXT_STEPS : indice_corrente, self.colonne_valide_idx]
            valore_multi = self.dati_raw[indice_corrente, self.colonne_valide_idx].reshape(1, -1)
            
            contesto_scaled = self.pca_scaler.transform(contesto_multi)
            valore_scaled = self.pca_scaler.transform(valore_multi)
            
            contesto_pca = self.pca_model.transform(contesto_scaled).flatten()
            vero_valore_pca = safe_extract(self.pca_model.transform(valore_scaled))
            
            predizioni = calcola_predizioni_step(self.modelli, contesto_pca)
            
            mse_array = np.array([(vero_valore_pca - p)**2 for p in predizioni])
            mse_array_scalato = self.ae_scaler.transform(mse_array.reshape(1, -1))
            
            with torch.no_grad():
                ricostruzione = self.autoencoder(torch.FloatTensor(mse_array_scalato)).numpy()
                anomaly_score = safe_extract(np.mean((mse_array_scalato - ricostruzione)**2))
                
            self.queue_dati.put({
                "step": indice_corrente, 
                "anomaly_score": anomaly_score
            })
            
            tempo_impiegato = time.time() - inizio_ciclo
            print(f"[LIVE TIMING] Step {indice_corrente} processed in {tempo_impiegato:.4f} seconds")
            
            indice_corrente += 1
            if tempo_impiegato < 1.0: 
                time.sleep(1.0 - tempo_impiegato)

    def _aggiorna_gui_periodicamente(self):
        while not self.queue_dati.empty():
            dato = self.queue_dati.get()
            self.plot_tempi.append(dato["step"])
            self.plot_anomaly_scores.append(dato["anomaly_score"])
            
            if len(self.plot_tempi) > 8: 
                self.plot_tempi.pop(0)
                self.plot_anomaly_scores.pop(0)
                
            self._disegna_grafici()

        if self.is_running_live: 
            self.root.after(500, self._aggiorna_gui_periodicamente)

    def _disegna_grafici(self):
        self.ax.clear()
        
        self.ax.plot(self.plot_tempi, self.plot_anomaly_scores, color='blue', linewidth=2, marker='o', markersize=6)
        
        if len(self.plot_tempi) > 0:
            current_step = self.plot_tempi[-1]
            
            if len(self.plot_tempi) < 8:
                x_min = self.plot_tempi[0]
                x_max = self.plot_tempi[0] + 11
            else:
                x_min = current_step - 7
                x_max = current_step + 4
                
            self.ax.set_xlim(x_min, x_max)
            
            self.ax.set_xticks(np.arange(int(x_min), int(x_max) + 1, 1))
            
            max_y_corrente = max(self.plot_anomaly_scores)
            limite_y = max(1.5, max_y_corrente * 1.1) 
            self.ax.set_ylim(0, limite_y)
            
            idx_start = max(0, int(np.floor(x_min)))
            idx_end = min(len(self.dati_label), int(np.ceil(x_max)) + 1)
            
            for step_idx in range(idx_start, idx_end):
                if self.dati_label[step_idx] == 1:
                    self.ax.axvspan(step_idx - 0.5, step_idx + 0.5, color='red', alpha=0.3)
        
        self.ax.set_title("Ensemble Anomaly Score in Real-Time", fontsize=14, fontweight='bold')
        self.ax.set_ylabel("Anomaly Score", fontsize=12)
        self.ax.set_xlabel("Time Step", fontsize=12)
        self.ax.grid(True, linestyle='--', alpha=0.5)
        
        self.fig.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = AnomalyDashboard(root)
    root.mainloop()