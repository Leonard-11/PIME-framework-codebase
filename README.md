# PIME-framework-codebase
This repository contains the official codebase and datasets for the thesis "PIME: A PCA-Informed Multivariate Ensemble Framework for Industrial Anomaly Detection at the Edge".

## Overview
Anomaly detection in real-world industrial systems requires the simultaneous evaluation of complex temporal and spatial dependencies. This repository implements a robust, scalable, and zero-shot anomaly detection pipeline leveraging the Time Series Foundation Models Chronos, Moirai, TimesFM, and TiRex. 

To mitigate the architectural biases of individual predictors, this work introduces a multi-model meta-learning ensemble based on a **Meta-Autoencoder**. Furthermore, to handle high-dimensional multivariate industrial data, a strict spatiotemporal decoupling strategy via Principal Component Analysis (PCA) is implemented. This culminates in the **PCA-Informed Multivariate Ensemble (PIME)** framework. Finally, the repository provides the complete software pipeline designed to deploy this architecture on resource-constrained Edge hardware (Raspberry Pi 4) for real-time, asynchronous industrial monitoring.

## Repository Structure

The repository is organized as follows:

### Datasets
* `Dataset forecasting/`: Raw time series data utilized for the zero-shot forecasting benchmark (fev-bench subset).
* `TSB-AD-U tuning/`: Tuning partition of the univariate anomaly detection benchmark.
* `SMD/`: Server Machine Dataset for multivariate anomaly detection evaluation.
* `machine-1-1` & `machine-1-1_label`: Specific SMD sequence and ground-truth labels utilized for the real-time Edge simulation.
* `serietest`: Sample time series sequence utilized specifically for hardware inference benchmarking on the Raspberry Pi.

### Standard Environment Scripts
* `requirements`: Dependencies required to execute the pipeline on a standard workstation/server.
* `fevbench.py`: Replicates the zero-shot point forecasting evaluation across the selected TSFMs (Next-step prediction, $H=1$).
* `univariate_ad.py`: Executes the zero-shot univariate anomaly detection pipeline and trains the Meta-Autoencoder ensemble.
* `multivariate_ad.py`: Implements the PIME framework, performing spatial decorrelation via PCA prior to zero-shot inference and meta-learning ensemble aggregation.

### Edge Deployment Scripts (Raspberry Pi 4)
* `requirements_raspberry`: Specific dependencies required for the embedded Edge environment.
* `raspberry_time.py`: Benchmarking script to evaluate the exact CPU inference latency of the foundation models on the Edge hardware.
* `simulation_first.py`: Initial execution script for the live simulation. It initializes the environment and downloads the pre-trained foundation models locally.
* `simulation_offline.py`: The core asynchronous real-time simulation framework. Operating completely offline, it manages data ingestion, spatial decorrelation, ensemble inference, and the dynamic Graphical User Interface (GUI) rendering.

---
