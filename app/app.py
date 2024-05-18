"""
Author: Tomáš Rajsigl
Email: xrajsi01@stud.fit.vutbr.cz
Filename: app.py

The backend implementation of the image inpainting web application. 
"""

import sys
import torch
import json
import base64
import io
import os
import time

import numpy as np

from flask import Flask, render_template, request, jsonify, send_from_directory
from torchvision import transforms
from PIL import Image


sys.path.append('../scripts/')
from model import *
from aotgan import *
from utils import remove_module_prefix

app = Flask(__name__)


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(("static"), "favicon.ico", mimetype="image/vnd.microsoft.icon")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/survey")
def survey():
    return render_template("survey.html")


@app.route("/process", methods=["POST"])
def process():
    """
    Process the uploaded image and mask data.

    Returns:
        flask.Response: A JSON response containing URLs of the inpainted images.
    """
    # Load the uploaded image from JSON payload
    data = request.json
    image_data = data.get("image", None)
    mask_data = data.get("mask", None)
    canvas_width = data.get("canvasWidth", 0)
    canvas_height = data.get("canvasHeight", 0)

    if image_data is None:
        return jsonify({"error": "Image data not provided"}), 400

    # Convert base64 image to PIL image
    image_pil = Image.open(io.BytesIO(base64.b64decode(image_data.split(",")[1]))).convert("RGB")

    image_path = "static/images/input.png"
    image_pil.save(image_path)

    binary_mask = torch.tensor(mask_data).view(1, canvas_height, canvas_width)

    # Convert PIL image to PyTorch tensor
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    img_tensor = transform(image_pil)
    masked_tensor = (img_tensor * (1 - binary_mask)) + binary_mask

    # Instantiate the models
    aotgan = InpaintGenerator()
    model = ResidualUNet()

    # Load the pretrained models
    aotgan.load_state_dict(torch.load("../scripts/models/AOTGAN.pt", map_location="cpu"))
    model.load_state_dict(torch.load("../scripts/models/ResUNet.pt", map_location="cpu"))

    aotgan.eval()
    model.eval()

    with torch.no_grad():
        aotgan_inpainted = aotgan(masked_tensor.unsqueeze(0), binary_mask.unsqueeze(0))
        model_inpainted = model(masked_tensor.unsqueeze(0), binary_mask.unsqueeze(0))

    aotgan_inpainted = img_tensor * (1 - binary_mask) + aotgan_inpainted * binary_mask
    model_inpainted = img_tensor * (1 - binary_mask) + model_inpainted * binary_mask

    aotgan_result = aotgan_inpainted[0]
    model_result = model_inpainted[0]

    # Convert the output tensor to a PIL image
    aotgan_image = transforms.ToPILImage()(aotgan_result)
    model_image = transforms.ToPILImage()(model_result)

    # Save the inpainted images
    aotgan_image_path = "static/images/aotgan_result.png"
    model_image_path = "static/images/resunet_result.png"
    aotgan_image.save(aotgan_image_path)
    model_image.save(model_image_path)

    mask_array = binary_mask.squeeze(0).numpy()
    mask_array = np.where(mask_array > 0, 255, 0).astype(np.uint8)
    binary_mask_pil = Image.fromarray(mask_array)
    mask_path = "static/images/mask.png"
    binary_mask_pil.save(mask_path)

    return jsonify({"aotgan_image": aotgan_image_path, "model_image": model_image_path})


@app.route("/submit-ranking", methods=["POST"])
def submit_ranking():
    """Process and store ranking data received from the survey."""
    # Get JSON data from request
    ranking_data = request.get_json()
    ranking_per_method = ranking_data.get("rankingPerMethod")
    ranking_per_image = ranking_data.get("rankingPerImage")

    # Create a directory for results if it doesn't exist
    results_directory = "static/survey/results"
    os.makedirs(results_directory, exist_ok=True)

    # Generate unique filenames based on current timestamp
    timestamp = int(time.time())
    method_filename = f"ranking_per_method_{timestamp}.json"
    image_filename = f"ranking_per_image_{timestamp}.json"

    # Store ranking data
    with open(os.path.join(results_directory, method_filename), "w") as method_file:
        json.dump(ranking_per_method, method_file, indent=4)

    with open(os.path.join(results_directory, image_filename), "w") as image_file:
        json.dump(ranking_per_image, image_file, indent=4)

    return "Ranking data received successfully!", 200


if __name__ == "__main__":
    app.run(debug=True)
