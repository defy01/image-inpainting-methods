# Comparison of Methods for Image Inpainting based on Deep Learning

To run the associated scripts for training and evaluating the models, follow these steps:

```
cd src
# Optionally create a Python venv
python -m venv env
source env/bin/activate
# Install the required libraries
pip install -r requirements.txt
```

# Running the scripts
Once the libraries are installed the scripts are ready to be executed:

```
cd scripts
# The very first run needs to be with the flag '-d' or '--download' set to 'True'
python3 main.py --train model.pt --download True
# Or for the evaluation script like so
python3 main.py --eval models/ResUNet.pt --download True
```

Visualization of the training process is periodically saved in the `scripts/training` folder. Likewise the evaluation plots can be found in `scripts/evaluation`.

# Running the web application
To run the demo web application locally on http://127.0.0.1:5000:
```
cd app
python3 app.py
```
