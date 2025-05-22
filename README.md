# AcousticTrafficMonitoring

## Setup

### Check Python version, and ensure that Python 3.8 or later is installed:

```
python3 --version
```

### Installing Project Dependencies using requirements.txt file
These instructions were tested using WSL on Ubuntu 24.04LTS

First create a venv:
```
python3 -m venv <myenvpath>
```

Source venv:
```
source venv/bin/activate
```

Install requirements:
```
pip3 install -r requirements.txt
```

Then you should be good to go!

### Data Preprocessing
Download and follow the instructions for extracting the [Acoustic Based Traffic Monitoring Dataset](https://dcase.community/challenge2024/task-acoustic-based-traffic-monitoring)

Place the `loc` directories into the Datapreprocessing folder. Run the `data_preprocessing.ipynb` notebook to start extracting audio segments of cars, commerical vehicles and background noise.

Once complete run the `spike_test.ipynb` file to convert the segements into spikes which the SNN model can use.

This process can take may take a day or two to complete.

### Running the Model
Ensure that the data preprocessing has been completed and the audio segements have been converted to spikes by refering to the instructions above before running the model. 

To run the model first ensure that you have activated your virtual environment by running
```
source venv/bin/activate
```

Then run
```
Python3 model.py
```

The model will then train on the spike data previously generated. Note that training takes a long time, to reduce training time reduce the number of epochs, however, this will reduce the accuracy of the model.

### Running the Front end

Make sure npm and Node.js are [installed](https://askubuntu.com/questions/1502744/how-to-install-node-js-latest-version-on-ubuntu-22-04).

Once installed run ```npm install ``` in the ```FrontEnd/traffic-ui``` directory.

Followed by ```npm audit fix --force```

Once you run ```npm start``` the webpage should open automatically in your browser, otherwise place the ip address provided in the terminal output in a web browser.

### Viewing the project website
The website summarising the project is deployed [here](https://haxinator.github.io/AcousticTrafficMonitoring/)

If you wish to make changes you can directly change the md files inside the `docs` subdirectory. To view the changes locally install [Jeykll](https://jekyllrb.com/docs/installation/). Otherwise after you push to Git the project website will update automatically.
 
### Installing Rockpool (Unnecessary if the above instructions were followed):

Create venv and install rockpool in project directory:
``` bash
python3 -m venv /path/to/new/virtual/environment
source .venv/bin/activate
pip install rockpool
python3 -c "import rockpool; print(rockpool.__version__)"
```
Alternatively using Conda:
``` bash
conda create -n rockpool-env python=3.8 #or later
conda activate rockpool-env
conda install -c conda-forge rockpool
python3 -c "import rockpool; print(rockpool.__version__)"
```

### Dependencies

Optional dependencies includes:
- [scipy](https://www.scipy.org) for [scipy](https://www.scipy.org)-backed modules
- [numba](https://numba.pydata.org) for [numba](https://numba.pydata.org)-backed modules
- [Jax](https://github.com/google/jax) and [Jaxlib](https://github.com/google/jax) for [Jax](https://github.com/google/jax)-backed modules
- [PyTorch](https://pytorch.org/) for [Torch](https://pytorch.org/)-backed modules
- [Brian2](https://github.com/brian-team/brian2) for [Brian](https://github.com/brian-team/brian2)-backed modules
- [Sinabs](https://pypi.org/project/sinabs/) for [Sinabs](https://pypi.org/project/sinabs/)-backed modules
- [Samna](https://pypi.org/project/samna/), [Xylosim](https://pypi.org/project/xylosim/), [Bitstruct](https://pypi.org/project/bitstruct/) for building and deploying modules to the Xylo hardware family
- [Matplotlib](https://matplotlib.org) or [HoloViews](http://holoviews.org) for plotting [`TimeSeries`](https://rockpool.ai/reference/_autosummary/timeseries.TimeSeries.html#timeseries.TimeSeries "timeseries.TimeSeries")
- [PyTest](https://github.com/pytest-dev/pytest) for running tests
- [Sphinx](http://www.sphinx-doc.org), [pandoc](https://pandoc.org), [recommonmark](https://pypi.org/project/sphinx-rtd-theme/), [NBSphinx](https://github.com/spatialaudio/nbsphinx), [sphinx-rtd-theme](https://pypi.org/project/sphinx-rtd-theme/) and [Sphinx-autobuild](https://github.com/GaretJax/sphinx-autobuild) for building documentation

Installing optional dependecies:
``` bash
pip install "rockpool[all]" #installs all optional dependencies
#or
#pip install "rockpool[numba, jax, torch, brian, sinabs, exodus, xylo, dynapse, tests, docs]" #installs specific optional dependencies
```

## Rockpool

Purpose of Rockpool is to allow user to design simulate, train and test dynamical neural networks. In Rockpool, each neuron is a dynamical object that evolves over time. Neurons and their states are encapsulated within the `Modules` class.

Rockpool modules typically require a `shape` parameter. This can be:

- An integer: for modules with an equal number of input and output channels.
- A tuple `(N_in, N_out)`: for modules with different input and output channel counts.

Here's an example defining a population of non-spiking rate neurons using the `Rate` class:
```
# - Define a feed-forward module with `N` neurons
N = 4
mod = Rate(N)
```

Rockpool modules provide methods for inspecting their internal structure:

- `state()`: Returns a dictionary of internal variables that influence the module's dynamics during and between evolution steps.
- `parameters()`: Returns a dictionary of model configurations that can be modified for training, such as weights, neuron time constants, and biases.
- `simulation_parameters()`: Returns a dictionary of configuration elements required for simulation but not trained, such as time-step duration and noise levels.
```
print(mod.state())
```
`{'x': array([0., 0., 0., 0.])}`

```
print(mod.parameters())
```
`{'tau': array([0.02, 0.02, 0.02, 0.02]), 'bias': array([0., 0., 0., 0.]), 'threshold': array([0., 0., 0., 0.])}`

```
print(mod.simulation_parameters())
```
`{'dt': 0.001, 'noise_std': 0.0, 'act_fn': <function H_ReLU at 0x7f9a35e6ede0>}`

## Xylo Audio

Xylo™ Audio integrates an analog audio front-end with spiking neural network inference core, enabling sub-mW audio processing for always-on applications

Rockpool bridges the gap between software-defined neural networks and the ultra-low-power capabilites of the Xylo™ Audio hardware. 

- **Low-Power Design:** Optimized for minimal power consumption, making it ideal for battery-powered and always-on devices.
- **Analog Audio Front-End:** Preprocesses audio signals directly in the analog domain, reducing the need for power-hungry digital conversion.
- **Spiking Neural Network (SNN) Inference Core:** Leverages the efficiency of SNNs for audio processing tasks.
- **Rockpool Integration:** Seamlessly interfaces with Rockpool, allowing users to design, simulate, and deploy SNNs on the Xylo™Audio platform.
- **Power Measurement:** Tools are provided to measure the power consumption of the Xylo™Audio HDK, and to optimize the networks for low power.
- **Real time processing:** The hardware is able to run in real time.

[Xylo Audio quick-start guide](https://rockpool.ai/devices/quick-xylo/deploy_to_xylo.html)
