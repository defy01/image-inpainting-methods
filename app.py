import torch
import io
import base64

from PIL import Image
from torchvision import transforms
from flask import Flask, render_template, request, jsonify

from model import *

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    # Load the uploaded image from JSON payload
    data = request.json
    image_data = data.get("image", None)
    mask_data = data.get("mask", None)
    canvas_width = data.get("canvasWidth", 0)
    canvas_height = data.get("canvasHeight", 0)

    if image_data is None:
        return jsonify({"error": "Image data not provided"}), 400

    # Convert base64 image to PIL image
    image_pil = Image.open(io.BytesIO(base64.b64decode(
        image_data.split(",")[1]))).convert("RGB")

    image_path = "static/image.png"
    image_pil.save(image_path)

    # Convert PIL image to PyTorch tensor
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    image_tensor = transform(image_pil)

    # Apply the binary mask to the input image
    binary_mask = torch.tensor(mask_data).view(1, canvas_height, canvas_width)

    model = Autoencoder()
    checkpoint = torch.load("model_mse.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        inpainted_image = model(image_tensor)

    print("image_tensor size:", image_tensor.size())
    print("inpainted_image size:", inpainted_image.size())
    print("binary_mask size:", binary_mask.size())

    inpainted_image = image_tensor + inpainted_image * (1 - binary_mask)

    # Convert the output tensor to a PIL image
    inpainted_image = transforms.ToPILImage()(inpainted_image)

    # Save the inpainted image
    inpainted_image_path = "static/inpainted_image.png"
    inpainted_image.save(inpainted_image_path)

    return jsonify({"inpainted_image": inpainted_image_path})


if __name__ == "__main__":
    app.run(debug=True)
