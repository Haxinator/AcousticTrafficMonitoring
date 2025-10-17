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

- Download [IDMT-TRAFFIC - Fraunhofer IDMT](https://www.idmt.fraunhofer.de/en/publications/datasets/traffic.html) : This is the basic dataset we use, all data samples have high quality recorded by high-quality sE8 microphones and meduim-quality MEMS microphones. The dataset includes recordings from 4 different locations, 4 different vehicle types, three different tempo limit areas, as well as dry and wet weather/road conditions. However, this dataset is highly unbalanced, the number of commercial vehicle samples are not enough for our training and testing, so we need another dataset as a supplementary.


- Download [DCASE2024 Acoustic Dataset](https://dcase.community/challenge2024/task-acoustic-based-traffic-monitoring): This is the supplementary dataset we use for the project. The audio data were collected using Bosch urban traffic monitoring sensors. A linear microphone array installed on the side of the road parallel to the direction of travel provides audio input to the traffic monitoring system. Real-world data is collected in urban environments with various traffic densities, from country roads with at most 5 vehicles per minute, to intercity roads with as many as 30 vehicles per direction per minute. Although the quality of this dataset is not very high, we still find some available commercial vehicle sample to use.


- The total sample size for training is 3000. We used 1000 car samples and 300 CV samples from `DCASE2024 Acoustic Dataset`, and 700 CV samples and 1000 background noise samples from `IDMT-TRAFFIC - Fraunhofer IDMT`.


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

![Model output](assets/images/output.png)  
Example of the model output. The `[1 0 0]`s are spikes for the final output classification. For example, `[1 0 0]` means the output of this spike is 0, `[0 1 0]` means the output of this spike is 1, and `[0 0 1]` means the output of this spike is 2. The final prediction will be based on the summarization of all the spike outputs. In the example output, there are 100 `0`s, 18 `1`s and 0 `2`s, so the final result would be `0`. We are using 0 for car, 1 for CV and 2 for background noise, so the prediction says it is a car.

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

![Frontend](/assets/images/frontend.png)

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

For more information about Rockpool, please go to [Rockpool](https://rockpool.ai/about.html) 

---

### 8. Xylo Audio Hardware

- Analog front-end for ultra-low-power audio
- Real-time inference
- Spiking neural network optimized

[Xylo deployment quickstart](https://rockpool.ai/devices/quick-xylo/deploy_to_xylo.html)

![Xylo](/assets/images/Xylo.jpg)
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

### 10. Roadside Demo
For the project, our ultimate goal is doing a roadside demo to check if all the functions and performance are working as expected. We made it able to classify some cars and commercial vehicles.

However, we also faced issues: 
- Lacked robustness to wind and talking.
- Detected same vehicle multiple times.
- Power occasionally returned NaN.
- Detected vehicles far away.
- Speed of vehicle affected classification.

![Roadside Demo](assets/images/Roadside.png)

## Model architecture
The implemented model is the [SynNet architecture](https://rockpool.ai/tutorials/synnet/synnet_architecture.html) that based on a Spiking Neural Network (SNN) architecture, comprising an input layer, three hidden layers, and an output layer. The input layer consists of 16 neurons, while each of the three hidden layers contains 24 neurons. The output layer includes 3 neurons responsible for the final classification or decision output (one for car, one for commercial vehicle, and one for background noise). All neurons in the network are Leaky Integrate-and-Fire (LIF) neurons, which encode temporal dynamics by accumulating membrane potential (vmem) over time and emitting spikes once the threshold potential is exceeded.

To enhance computational efficiency, the network employs the LIF Exodus neuron model from Rockpool, which provides optimized support for CUDA acceleration, thereby significantly improving the training performance on compatible GPU hardware.


## The Process of the Research and Development
At the beginning of the project, we started training our model using a dataset recommended by the client. This dataset served as the foundation for our initial experimentation. However, before training could begin, we performed manual filtering to eliminate low-quality samples, ensuring that only high-quality audio segments were used for model input. While this process improved data reliability, it drastically reduced the available sample size.

To compensate for this shortage, we implemented data augmentation. These segments were then further processed into different feature channels, where each channel represented a specific acoustic characteristic. All channels were merged and transformed into spiking neuron inputs, which would eventually produce binary outputs — 0 or 1, representing vehicle class.

We trained the model using the Rockpool framework, which allowed us to simulate and evolve spiking neural networks with temporal dynamics. However, no matter how we adjusted the parameters or increased the number of training epochs, the model’s performance remained stagnant. The validation accuracy consistently hovered around 50%, far below our goal of 90%, indicating possible data bottlenecks rather than model flaws.

Realizing this, we began searching for an alternative dataset with cleaner labels and more consistent audio quality. Eventually, we found a higher-quality dataset that significantly improved baseline accuracy. However, this new dataset lacked sufficient truck samples. To address this, we merged selected high-quality truck audio segments from the original dataset with the new one, creating a hybrid dataset that offered both balance and accuracy with the sample size around 30K.

This decision proved crucial. On the very first training run using the hybrid dataset, accuracy immediately exceeded 50%, and continued to improve with parameter tuning. After several iterations, the model finally surpassed 90% accuracy, achieving our project target.

To accelerate training time and improve resource efficiency, we also leveraged CUDA-based GPU acceleration using the Exodus backend in Rockpool. This significantly reduced the training duration and allowed us to experiment more rapidly with hyperparameters.

![ROC0](/assets/images/ROC0.png)  
ROC curve before quantisation

![ROC](/assets/images/ROC.png)  
ROC curve after quantisation

The first two images show the ROC curves before quantisation, based on both membrane potential outputs and spike counts. The third plot illustrates ROC curves after quantisation, with AUC values still exceeding 0.90 for all three classes (Car, Commercial, Background), indicating that quantisation preserved most of the model's discriminative power.

![accuracy](/assets/images/Accuracy.png)  
Accuracy Curve and Training Loss Curve

The accuracy curve shows rapid early improvement and eventual convergence at around 90%, while the training loss curve decreases steadily, confirming effective learning and stable training.

![metrics](/assets/images/metrics.png)  
Confusion matrix for testing

This plot presents a confusion matrix on the test set. The model achieved an overall test accuracy of 90.62%, correctly identifying most samples across all classes, with especially strong performance on cars and background noise. Some misclassifications occurred between trucks and other classes, consistent with class imbalance during training.