let selectedImage;
let binaryMask = [];
let canvas = document.getElementById('image-canvas');
let context = canvas.getContext('2d');
let isDrawing = false;
let binaryMaskWidth;
let binaryMaskHeight;

document
  .getElementById('file-input')
  .addEventListener('change', function (event) {
    selectedImage = event.target.files[0];
    showImage(selectedImage);
  });

function showImage(imageFile) {
  const reader = new FileReader();
  reader.onload = function (e) {
    const img = new Image();
    img.onload = function () {
      context.drawImage(img, 0, 0, canvas.width, canvas.height);
      binaryMaskWidth = canvas.width;
      binaryMaskHeight = canvas.height;
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(imageFile);
  binaryMask = Array(canvas.width * canvas.height).fill(1);
}

canvas.addEventListener('mousedown', function (event) {
  isDrawing = true;
  draw(event);
});

canvas.addEventListener('mousemove', function (event) {
  if (isDrawing) {
    draw(event);
  }
});

canvas.addEventListener('mouseup', function () {
  isDrawing = false;
  context.beginPath();
});

function draw(event) {
  if (!isDrawing) return;
  const brushSize = 20;

  context.lineWidth = brushSize;
  context.lineCap = 'square';
  context.strokeStyle = 'rgba(0, 0, 0, 1)';
  context.lineTo(
    event.clientX - canvas.offsetLeft,
    event.clientY - canvas.offsetTop
  );
  context.stroke();
  context.beginPath();
  context.moveTo(
    event.clientX - canvas.offsetLeft,
    event.clientY - canvas.offsetTop
  );

  // Update binary mask in a square region
  const centerX = Math.floor(
    ((event.clientX - canvas.offsetLeft) / canvas.width) * binaryMaskWidth
  );
  const centerY = Math.floor(
    ((event.clientY - canvas.offsetTop) / canvas.height) * binaryMaskHeight
  );

  for (let i = -brushSize / 2; i < brushSize / 2; i++) {
    for (let j = -brushSize / 2; j < brushSize / 2; j++) {
      const x = Math.min(Math.max(centerX + i, 0), binaryMaskWidth);
      const y = Math.min(Math.max(centerY + j, 0), binaryMaskHeight);
      binaryMask[y * binaryMaskWidth + x] = 0; // Set the pixel to 0 (black)
    }
  }
}

function inpaintImage() {
  // Get the base64 representation of the masked image
  const imageDataURL = canvas.toDataURL('image/png');

  // Send the masked image and binary mask to the Flask backend
  fetch('/process', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      image: imageDataURL,
      mask: binaryMask,
      canvasWidth: canvas.width,
      canvasHeight: canvas.height,
    }),
  })
    .then((response) => response.json())
    .then((data) => {
      // Display the inpainted image on the canvas
      const img = new Image();
      img.onload = function () {
        context.drawImage(img, 0, 0, canvas.width, canvas.height);
      };
      img.src = data.inpainted_image;
    })
    .catch((error) => {
      console.error('Error:', error);
      console.error(image);
    });
}
