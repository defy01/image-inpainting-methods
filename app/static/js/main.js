// Define variables
let selectedImage;
let binaryMaskWidth;
let binaryMaskHeight;
let binaryMask = [];
let canDraw = false;
let isDrawing = false;
let brushSize = 20;

// Variables to store the history of drawn lines and binary masks
let imageHistoryStack = [];
let maskHistoryStack = [];
let undoCounter = 0;

// DOM elements
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const canvas = document.getElementById('image-canvas');
const changeImageButton = document.getElementById('change-image');
const nextButton = document.getElementById('next');
const undoButton = document.getElementById('undo');
const brushSlider = document.getElementById('brush-size');

// Get 2D rendering context for the canvas
let context = canvas.getContext('2d');
// Hide the canvas initially
canvas.style.display = 'none';

// Initialize Dropzone
Dropzone.autoDiscover = false;
const myDropzone = new Dropzone('#dropzone', {
  url: '/',
  acceptedFiles: 'image/jpeg, image/png',
  maxFiles: 1,
  clickable: true,
  previewsContainer: false,
  autoProcessQueue: false,
  async init() {
    this.on('addedfile', async function (file) {
      selectedImage = file;
      await renderImage(selectedImage);
    });
    this.on('dragenter', () => this.element.classList.add('dragover'));
    this.on('dragleave', () => this.element.classList.remove('dragover'));
  },
});

// Event listeners
fileInput.addEventListener('change', async (event) => {
  const file = event.target.files[0];
  if (file) {
    context.clearRect(0, 0, canvas.width, canvas.height);
    await renderImage(file);
  }
});

brushSlider.addEventListener('input', function (event) {
  let newSize = event.target.value;
  newSize = Math.floor(newSize / 2) * 2;
  brushSize = newSize;
  updateBrushCursorSize();
});

canvas.addEventListener('mousemove', updateCursorPosition);
canvas.addEventListener(
  'mouseenter',
  () => (circleCursor.style.display = 'block')
);
canvas.addEventListener(
  'mouseleave',
  () => (circleCursor.style.display = 'none')
);

canvas.addEventListener('mousedown', handleMouseDown);
canvas.addEventListener('mousemove', handleMouseMove);
canvas.addEventListener('mouseup', handleMouseUp);

function changeImage() {
  fileInput.click();
}

/**
 * Renders the selected image onto the canvas.
 * @param {File} file - The image file to render.
 * @returns {Promise<void>}
 */
async function renderImage(file) {
  const reader = new FileReader();
  await new Promise((resolve, reject) => {
    reader.onload = function (e) {
      const img = new Image();
      img.onload = function () {
        context.drawImage(img, 0, 0, canvas.width, canvas.height);
        binaryMaskWidth = canvas.width;
        binaryMaskHeight = canvas.height;
        canvas.style.display = 'block';
        dropzone.style.display = 'none';
        changeImageButton.style.display = 'inline-block';
        nextButton.style.display = 'inline-block';
        resolve();
      };
      img.src = e.target.result;
      selectedImage.dataURL = e.target.result;
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
  // Update the stacks and mask every time a new image is rendered
  imageHistoryStack = [];
  maskHistoryStack = [];
  imageHistoryStack.push(canvas.toDataURL());
  binaryMask = Array(canvas.width * canvas.height).fill(0);
}

// Canvas mouse event handlers
function handleMouseDown(event) {
  if (canDraw) {
    isDrawing = true;
    // Store current mask state for undo
    maskHistoryStack.push([...binaryMask]);
    undoButton.disabled = false;
    draw(event);
  }
}

function handleMouseMove(event) {
  if (canDraw && isDrawing) {
    draw(event);
  }
}

function handleMouseUp() {
  if (canDraw) {
    isDrawing = false;
    context.beginPath();
    saveCanvasState();
  }
}

/**
 * Reverts the last drawing action.
 */
function undoLastStroke() {
  if (undoCounter > 0) {
    undoCounter--;
    let img = new Image();
    img.onload = function () {
      context.drawImage(img, 0, 0, canvas.width, canvas.height);
    };
    img.src = imageHistoryStack[undoCounter];
    binaryMask = maskHistoryStack.pop();
    if (maskHistoryStack.length == 0) {
      undoButton.disabled = true;
    }
  }
}

/**
 * Saves the current canvas state.
 */
function saveCanvasState() {
  undoCounter++;
  if (undoCounter < imageHistoryStack.length) {
    imageHistoryStack.length = undoCounter;
  }
  imageHistoryStack.push(canvas.toDataURL());
}

/**
 * Updates the brush cursor size.
 */
function updateBrushCursorSize() {
  const cursorSize = brushSize;
  circleCursor.style.width = cursorSize + 'px';
  circleCursor.style.height = cursorSize + 'px';
}

/**
 * Updates the cursor position.
 * @param {MouseEvent} event - Mouse move event object.
 */
function updateCursorPosition(event) {
  circleCursor.style.left =
    event.clientX - circleCursor.offsetWidth / 2 + 1 + 'px';
  circleCursor.style.top =
    event.clientY - circleCursor.offsetHeight / 2 + 1 + 'px';
}

/**
 * Draws on the canvas and handles the binary mask update.
 * @param {MouseEvent} event - Mouse event object.
 */
function draw(event) {
  if (!isDrawing) return;

  let brushRadius = brushSize / 2;
  let expandedRadius = brushRadius + 2;
  context.lineWidth = brushSize;
  context.lineCap = 'round';
  context.strokeStyle = 'rgba(255, 255, 255, 1)';

  // Draw on canvas
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

  const centerX = Math.floor(
    ((event.clientX - canvas.offsetLeft) / canvas.width) * binaryMaskWidth
  );
  const centerY = Math.floor(
    ((event.clientY - canvas.offsetTop) / canvas.height) * binaryMaskHeight
  );

  // Update mask values affected by the brush stroke
  for (let i = -expandedRadius; i <= expandedRadius; i++) {
    for (let j = -expandedRadius; j <= expandedRadius; j++) {
      // Calculate distance from brush center
      const distance = Math.sqrt(i * i + j * j);
      if (distance <= expandedRadius) {
        const x = Math.min(Math.max(centerX + i, 0), binaryMaskWidth - 1);
        const y = Math.min(Math.max(centerY + j, 0), binaryMaskHeight - 1);
        binaryMask[y * binaryMaskWidth + x] = 1;
      }
    }
  }
}

/**
 * Initiates the mask drawing step.
 */
function startMaskingStep() {
  updateAppState('Mask step');

  undoCounter = 0;
  if (maskHistoryStack.length == 0) {
    undoButton.disabled = true;
  }
}

/**
 * Starts the process over by returning to the first step of choosing an image.
 */
function returnToImageSelectionStep() {
  updateAppState('Start over');

  // Reset brush size
  brushSlider.value = 20;
  brushSize = 20;
  // Reset canvas
  context.clearRect(0, 0, canvas.width, canvas.height);
  canvas.style.display = 'none';
  canvas.style.cursor = 'default';
  canDraw = false;
}

/**
 * Hides given elements.
 * @param {string[]} elements - IDs of elements to hide.
 */
function hideElements(elements) {
  elements.forEach((id) => {
    const element = document.getElementById(id);
    if (element) {
      element.style.display = 'none';
    }
  });
}

/**
 * Shows given elements.
 * @param {string[]} elements - IDs of elements to show.
 */
function showElements(elements) {
  elements.forEach((id) => {
    const element = document.getElementById(id);
    if (element) {
      element.style.display = 'inline-block';
    }
  });
}

/**
 * Updates the application user interface based on the current state.
 * @param {string} state - The state to update the UI to.
 *                         Possible values: 'Mask step', 'Start over', 'Inpaint image', 'Display results'.
 */
function updateAppState(state) {
  const uiState = {
    'Mask step': () => {
      const elementsToHide = ['next', 'change-image'];
      const elementsToShow = ['undo', 'start-over', 'inpaint', 'brush-size'];
      hideElements(elementsToHide);
      showElements(elementsToShow);

      // Adjust button margins
      document.querySelectorAll('button:not(#undo)').forEach((button) => {
        button.style.margin = '20px auto';
      });

      circleCursor = document.createElement('div');
      circleCursor.className = 'circle-cursor';
      document.body.appendChild(circleCursor);
      circleCursor.style.display = 'none';

      // Update titles and instructions
      const h1 = document.querySelector('h1');
      const h3 = document.querySelector('h3');
      h1.textContent = 'Step two';
      h3.textContent = 'Mask out any parts of the image using your cursor.';
      h3.appendChild(document.createElement('br'));
      h3.append('Brush size can be adjusted with the slider below.');

      // Adjust cursor and button state
      const startOverButton = document.getElementById('start-over');
      startOverButton.classList.add('change');
      canvas.style.cursor = 'none';
      canDraw = true;
    },
    'Start over': () => {
      const elementsToHide = [
        'canvas',
        'next',
        'start-over',
        'brush-size',
        'undo',
        'inpaint',
      ];
      hideElements(elementsToHide);

      // Remove redundant canvas
      const canvasContainer = document.getElementById('canvas-container');
      const canvasElements = canvasContainer.querySelectorAll('div');
      if (canvasElements.length > 1) {
        canvasContainer.removeChild(canvasElements[1]);
        canvasContainer.removeChild(canvasElements[2]);
      }

      // Adjust button margins
      document.querySelectorAll('button:not(#undo)').forEach((button) => {
        button.style.margin = '40px auto';
      });

      dropzone.style.display = 'flex';
      dropzone.classList.remove('dragover');
      const paragraph = document.querySelector('#canvas-container p');
      paragraph.style.display = 'none';

      // Update titles and instructions
      const h1 = document.querySelector('h1');
      const h3 = document.querySelector('h3');
      h1.textContent = 'Step one';
      h3.textContent =
        'Upload your image, the image will be scaled by default.';
      const br = document.createElement('br');
      h3.appendChild(br);
      h3.append('The supported format is either JPG or PNG.');

      // Remove brush circle cursor
      const circleCursor = document.querySelector('.circle-cursor');
      if (circleCursor) {
        document.body.removeChild(circleCursor);
      }
    },
    'Inpaint image': () => {
      brushSlider.style.display = 'none';
      document.getElementById('spinner').style.display = 'block';
      document.querySelectorAll('button').forEach((button) => {
        button.style.display = 'none';
      });
    },
    'Display results': () => {
      const elementsToHide = ['brush-size', 'undo', 'spinner'];

      hideElements(elementsToHide);

      document.getElementById('start-over').style.display = 'block';
      document.querySelectorAll('button:not(#undo)').forEach((button) => {
        button.style.margin = '40px auto';
      });
    },
  };

  if (uiState[state]) {
    uiState[state]();
  }
}

/**
 * Inpaints the image using the provided mask.
 * @returns {Promise<void>}
 */
async function inpaintImage() {
  updateAppState('Inpaint image');
  // Get the base64 representation of the masked image
  const imageDataURL = canvas.toDataURL('image/png');
  try {
    // Send the masked image and binary mask to the backend
    const response = await fetch('/process', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image: imageDataURL,
        mask: binaryMask,
        canvasWidth: canvas.width,
        canvasHeight: canvas.height,
      }),
    });
    if (!response.ok) {
      throw new Error('Network response was not ok');
    }
    const data = await response.json();
    // Handle response data
    showResults(data);
  } catch (error) {
    console.error('Error:', error);
  }
}

/**
 * Displays the inpainting results.
 * @param {Object} data - Inpainting results data.
 */
function showResults(data) {
  updateAppState('Display results');
  imageHistoryStack = [];
  undoCounter = 0;

  // Display the original image
  const img = new Image();
  img.onload = function () {
    context.drawImage(img, 0, 0, canvas.width, canvas.height);
  };
  img.src = selectedImage.dataURL;

  const paragraph = document.querySelector('#canvas-container p');
  paragraph.style.display = 'block';
  const ResUNetResultDiv = createResultDiv('Inpainted by ResUNet',data.model_image);
  const AOTGANResultDiv = createResultDiv('Inpainted by AOT-GAN',data.aotgan_image);
  const canvasContainer = document.getElementById('canvas-container');
  canvasContainer.appendChild(ResUNetResultDiv);
  canvasContainer.appendChild(AOTGANResultDiv);

  // Update the UI text
  document.querySelector('h1').textContent = 'And just like that!';
  document.querySelector('h3').textContent =
    'Inpainting complete. You can now compare the results by the two models.';
  document.getElementById('inpaint').style.display = 'none';
  document.getElementById('start-over').classList.remove('change');
  document.body.removeChild(circleCursor);
  canvas.style.cursor = 'default';
  canDraw = false;
}

/**
 * Creates a result div with the provided title and image URL.
 * @param {string} title - Title of the result.
 * @param {string} imageUrl - URL of the image.
 * @returns {HTMLDivElement} - Result div element.
 */
function createResultDiv(title, imageUrl) {
  const div = document.createElement('div');
  const text = document.createElement('p');
  text.textContent = title;
  const newCanvas = document.createElement('canvas');
  newCanvas.width = canvas.width;
  newCanvas.height = canvas.height;
  const newContext = newCanvas.getContext('2d');

  div.appendChild(newCanvas);
  div.appendChild(text);

  const img = new Image();
  img.onload = function () {
    newContext.drawImage(img, 0, 0, newCanvas.width, newCanvas.height);
  };
  img.src = imageUrl;

  return div;
}
