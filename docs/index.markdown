---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: post
title: Acoustic Traffic Monitoring
---

## Introduction

Roadside traffic monitoring is used in many cities to monitor the usage and condition of 
roadways. This can be performed using a range of sensors; in this project we will build a 
prototype acoustic traffic monitoring system designed for low-power operation, which 
will detect and distinguish between cars and commercial vehicles. we will design and 
deploy our application using an ultra-low-power audio processing device Xylo 
Audio from SynSense. 

The system will be designed using the acoustic-based traffic monitoring dataset and 
augmentation system from DCASE2024. When a vehicle is detected, the system would 
provide a sparse positive output indicating either a car or commercial vehicle. The system 
would provide no output at all other times. It is not necessary to use the multi
microphone array data, but instead we may use monaural data. Likewise it is not 
necessary to detect the direction of travel; simply detecting and classifying the presence 
of vehicles is sufficient. 

we will implement a live roadside demo, based on a laptop, as the final phase of the 
project. This will include a python notebook or web API front-end. 

Xylo is a new audio inference processor, which encodes audio as sparse temporal events, 
and performs ML inference using networks of low-bit-depth, temporally sparse 
neurons (sometimes known as Spiking Neural Networks). In this project we will train an 
application to classify real-time audio input using Xylo, and deploy it to prototype 
hardware. Xylo supports simple neural network architectures (dense feed-forward; 
recurrent; residual).

### Some Constraints and considerations
Here are some project constraints: 

1. The system's functionality will be demonstrated in a roadside setting using the Xylo Development Kit connected to a laptop 

2. The vehicle classification model must be a Spiking Neural Network (SNN). 

3. The Xylo Development Kit must be the hardware platform for the system.

4. The system must use an acoustic-based traffic monitoring system to classify vehicles.

## Technical Part
### 1. Environment Setup (Python & Rockpool)

- Python version should be between 3.8 and 3.11.

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.10
sudo apt install python3.10-venv
python3.10 -m venv <myenvpath>
```

Activate the virtual environment:

```bash
source <myenvpath>/bin/activate
pip install -r requirements.txt
```

---

### 2. Data Preprocessing

- Download [IDMT-TRAFFIC - Fraunhofer IDMT](https://www.idmt.fraunhofer.de/en/publications/datasets/traffic.html) (The basic dataset we use)
- Download [DCASE2024 Acoustic Dataset](https://dcase.community/challenge2024/task-acoustic-based-traffic-monitoring) (We use some commercial vehicle samples from this dataset as supplementary)
- Place `loc*` directories into `Datapreprocessing/`
- Run the following notebooks:
  - `data_preprocessing.ipynb`: to extract segments
  - `spike_test.ipynb`: to convert audio into spike signals

**Note:** This process may take 1–2 days.

---

### 3. Training the Spiking Neural Network Model

- Requirements:
  - NVIDIA GPU
  - CUDA toolkit

```bash
# Install Rockpool
pip install "rockpool[sinabs, exodus]"
```

```bash
# Install CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tensorflow[and-cuda]
```

Run the model:

```bash
source venv/bin/activate
python3 model.py
```

Reduce `epochs` to shorten training time (may reduce accuracy).

---

### 4. Running the Backend (FastAPI)

Ensure dependencies installed: FastAPI, Rockpool, OS, numpy, matplotlib

```bash
uvicorn main:app --reload --port 3000
```

---

### 5. Running the Frontend (React + Tailwind)

Make sure Node.js and npm are installed.

```bash
cd FrontEnd/traffic-ui
npm install
npm audit fix --force
npm start
```

Visit the IP address shown in terminal (usually http://localhost:3000).

---

### 6. Blog Site Deployment

- Blog: [https://haxinator.github.io/AcousticTrafficMonitoring/](https://haxinator.github.io/AcousticTrafficMonitoring/)
- Edit markdown files in `docs/`
- Install [Jekyll](https://jekyllrb.com/docs/installation/) to preview locally

---

### 7. Rockpool Overview

Rockpool is a simulation and deployment framework for SNNs:

- Define neurons with `Rate`, `LIF`, etc.
- Inspect states:

```python
print(mod.state())
print(mod.parameters())
print(mod.simulation_parameters())
```

---

### 8. Xylo Audio Hardware

- Analog front-end for ultra-low-power audio
- Real-time inference
- Spiking neural network optimized

[Xylo deployment quickstart](https://rockpool.ai/devices/quick-xylo/deploy_to_xylo.html)

---

### 9. Optional: Installing Rockpool

```bash
python3 -m venv ./venv
source venv/bin/activate
pip install rockpool
```

or with conda:

```bash
conda create -n rockpool-env python=3.8
conda activate rockpool-env
conda install -c conda-forge rockpool
```

## The process of the research and development
At the beginning of the project, we started training our model using a dataset recommended by the client. This dataset served as the foundation for our initial experimentation. However, before training could begin, we performed manual filtering to eliminate low-quality samples, ensuring that only high-quality audio segments were used for model input. While this process improved data reliability, it drastically reduced the available sample size.

To compensate for this shortage, we implemented data augmentation by segmenting each sample into multiple shorter samples. These segments were then further processed into different feature channels, where each channel represented a specific acoustic characteristic. All channels were merged and transformed into spiking neuron inputs, which would eventually produce binary outputs — 0 or 1, representing vehicle class.

We trained the model using the Rockpool framework, which allowed us to simulate and evolve spiking neural networks with temporal dynamics. However, no matter how we adjusted the parameters or increased the number of training epochs, the model’s performance remained stagnant. The validation accuracy consistently hovered around 50%, far below our goal of 90%, indicating possible data bottlenecks rather than model flaws.

Realizing this, we began searching for an alternative dataset with cleaner labels and more consistent audio quality. Eventually, we found a higher-quality dataset that significantly improved baseline accuracy. However, this new dataset lacked sufficient truck samples. To address this, we merged selected high-quality truck audio segments from the original dataset with the new one, creating a hybrid dataset that offered both balance and accuracy.

This decision proved crucial. On the very first training run using the hybrid dataset, accuracy immediately exceeded 50%, and continued to improve with parameter tuning. After several iterations, the model finally surpassed 90% accuracy, achieving our project target.

To accelerate training time and improve resource efficiency, we also leveraged CUDA-based GPU acceleration using the Exodus backend in Rockpool. This significantly reduced the training duration and allowed us to experiment more rapidly with hyperparameters.