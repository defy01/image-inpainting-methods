# Comparison of Methods for Image Inpainting based on Deep Learning
The following repository contains the implementation part of my Bachelor's Thesis.

# Prerequisites
* Python 3.10.12
* [PyTorch](https://pytorch.org/) 

Before running the associated scripts for training and evaluating the models, follow these steps:


```
cd src
# Optionally create a Python venv
python -m venv env
source env/bin/activate
# Install the required libraries
pip install -r requirements.txt
```

## Datasets
The used datasets are:
* [Places365](http://places2.csail.mit.edu/)
* [NVIDIA Irregular Mask Dataset](https://nv-adlr.github.io/publication/partialconv-inpainting)

# Running the scripts
The very first run requires the argument `-d` or `--download` to be set to `True`. This will automatically download the Places365 dataset. However, in order to run the scripts, the NVIDIA Irregular Mask Dataset needs to be downloaded manually from the official webpage [here](https://nv-adlr.github.io/publication/partialconv-inpainting).

```
cd scripts
# Traning the network
python3 main.py --train model.pt --download True
# Or running the evaluation of it
python3 main.py --eval models/ResUNet.pt --download True
```

Visualization of the training process is periodically saved in the `scripts/training` folder. The evaluation plots can be likewise found in `scripts/evaluation`.

# Running the web application
To run the demo web application locally on http://127.0.0.1:5000:
```
cd app
python3 app.py
```

# Pretrained Models
Due to size constraints, the pretrained models are not included in this repository. Please do not hesitate to contact me if you are interested in them: xrajsi01@stud.fit.vutbr.cz.
