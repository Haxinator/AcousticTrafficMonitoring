---
layout: post
title:  "Technical Blog For Our Project: Acoustic-Traffic-Monitoring"
date:   2025-04-21 19:20:00 +0800
categories: acousticTrafficMonitoring update
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


## The process of the research and development
