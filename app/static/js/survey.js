// Constants
const methods = ['AOTGAN', 'MINE', 'PCONV'];
const numGroups = 9;
const numImagesPerGroup = 3;

// Initialize ranking data
const images = Array.from({ length: numGroups }, () => {
  return Array.from({ length: numImagesPerGroup }, (_, index) => {
    return { label: methods[index], rank: 0 };
  });
});

const rankingData = {
  AOTGAN: {
    ranking: [0, 0, 0], // First, Second, Last
    noticeability: [0, 0, 0, 0], // Noticeability levels
  },
  MINE: {
    ranking: [0, 0, 0],
    noticeability: [0, 0, 0, 0],
  },
  PCONV: {
    ranking: [0, 0, 0],
    noticeability: [0, 0, 0, 0],
  },
};

// Counter for image group iteration
let counter = 0;

// DOM elements
const container = document.getElementById('survey-container');
const confirmButton = document.getElementById('confirm-ranking');
const nextButton = document.getElementById('next-group');

// Event listeners
confirmButton.addEventListener('click', function () {
  disableSortable();
  addRankingOptions();
  updateRanking(images);

  // Update UI elements for the next step
  this.style.display = 'none';
  nextButton.style.display = 'block';
  const h3Element = document.querySelector('h3');
  h3Element.textContent = 'Now, please rate one image at a time based on how ';
  const strongElement = document.createElement('strong');
  strongElement.textContent = 'noticeable';
  h3Element.appendChild(strongElement);
  h3Element.appendChild(document.createTextNode(' the object'));
  h3Element.appendChild(document.createElement('br'));
  h3Element.append(
    'removal/the changes made to the image are on the displayed scale.'
  );

  // Disable the Next button if radiobuttons aren't checked
  const radioButtons = document.querySelectorAll('input[type="radio"]');
  radioButtons.forEach((radioButton) => {
    radioButton.addEventListener('change', function () {
      // Check if all required radio buttons are checked
      const requiredRadiosChecked =
        document.querySelector('input[name="rank-0-0"]:checked') &&
        document.querySelector('input[name="rank-1-1"]:checked') &&
        document.querySelector('input[name="rank-2-2"]:checked');
      if (requiredRadiosChecked) {
        nextButton.disabled = false;
      }
    });
  });
});

nextButton.addEventListener('click', function () {
  this.style.display = 'none';
  this.disabled = true;
  counter++;
  updateNoticeabilityRanking();
  renderImages(counter);
  enableSortable();

  // Update step instructions
  const h3Element = document.querySelector('h3');
  h3Element.textContent =
    'An object or multiple objects have been removed from these images. Your task is to';
  h3Element.appendChild(document.createElement('br'));
  h3Element.append(
    'drag the images to sort them from best (left) to worst (right) by visual quality.'
  );
});

/**
 * Renders images for the given group index
 * @param {number} groupIndex - Index of the currently displayed image group
 */
function renderImages(groupIndex) {
  counter = groupIndex;
  container.innerHTML = '';
  let rank = 1;
  // Check whether all images have already been ranked
  if (groupIndex >= numGroups) {
    sendDataToBackend();
    return;
  }
  // Shuffle images to avoid bias
  const indices = [0, 1, 2];
  const shuffledIndices = shuffleArray(indices);
  const reorderedImages = [];
  for (let i = 0; i < shuffledIndices.length; i++) {
    const index = shuffledIndices[i];
    const imageContainer = document.createElement('div');
    imageContainer.classList.add('image-container');
    container.appendChild(imageContainer);

    const img = document.createElement('img');
    img.classList.add('survey-images');
    // Set image src attribute based on the fixed labels and group index
    img.src = `/static/survey/dataset/image${groupIndex + 1}_${index + 1}.png`;
    img.alt = `Image ${groupIndex + 1}_${index + 1}`;
    img.width = 512;
    img.height = 512;
    imageContainer.appendChild(img);
    confirmButton.style.display = 'block';

    // Assign the corresponding image label
    const label = methods[index];
    img.dataset.label = label;

    // Update the rank for the current image
    images[groupIndex][index].label = label;
    images[groupIndex][index].rank = rank;
    reorderedImages.push(images[groupIndex][index]);
    rank++;
  }

  // Update images array with reordered images
  images[groupIndex] = reorderedImages;
}

/**
 * Shuffles the given array
 * @param {Array} array - The array to shuffle
 * @returns {Array} - The resulting array
 */
function shuffleArray(array) {
  const shuffled = [...array];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
}

/**
 * Enables sorting functionality for image containers
 */
function enableSortable() {
  sortable = new Sortable(container, {
    animation: 200,
    ghostClass: 'sortable-ghost',
    onEnd: function (evt) {
      const startIndex = evt.oldIndex;
      const endIndex = evt.newIndex;

      const movedImages = images[counter].splice(startIndex, 1);
      images[counter].splice(endIndex, 0, ...movedImages);
      // Update ranks for all images in the group
      for (let i = 0; i < images[counter].length; i++) {
        images[counter][i].rank = i + 1;
      }
    },
  });
}

/**
 * Updates the ranking data based on the current image rankings.
 * @param {Array<Array<{label: string, rank: number}>>} images - Array of images with labels and ranks.
 */
function updateRanking(images) {
  // Get the current group based on counter
  const group = images[counter];
  for (let i = 0; i < group.length; i++) {
    const { rank, label } = group[i];
    const position = getPosition(rank);
    // Increment the corresponding method ranking based on position
    rankingData[label].ranking[position - 1]++;
  }
}

/**
 * Updates the noticeability ranking data based on user selections.
 */
function updateNoticeabilityRanking() {
  const radioButtons = container.querySelectorAll('input[type="radio"]');
  radioButtons.forEach((radioButton) => {
    if (radioButton.checked) {
      // Extract index and value from radio button name
      const [index, value] = radioButton.id.split('-').slice(1);
      const label = images.at(counter - 1).at(index).label;
      rankingData[label].noticeability[parseInt(value)]++;
    }
  });
}

function getPosition(rank) {
  return rank === 1 ? 1 : rank === 2 ? 2 : 3;
}

/**
 * Adds ranking options (radio buttons) for each image container.
 */
function addRankingOptions() {
  // Add radio buttons to each image container
  const imageContainers = container.querySelectorAll('.image-container');
  imageContainers.forEach((imageContainer, index) => {
    let num = 0;
    const radioContainer = document.createElement('div');
    radioContainer.classList.add('radio-container');
    // Add radio buttons to the first row
    const radioRow = document.createElement('div');
    radioRow.classList.add('radio-row');
    for (let j = 1; j <= 4; j++) {
      const radioDiv = document.createElement('div');
      const radio = document.createElement('input');
      radio.type = 'radio';
      radio.id = `rank-${index}-${num++}`;
      radio.name = `rank-${index}-${index}`;
      radio.value = j;
      radio.classList.add('radio');
      radioDiv.appendChild(radio);
      radioRow.appendChild(radioDiv);
    }
    radioContainer.appendChild(radioRow);

    // Add radio button labels
    const labelsRow = document.createElement('div');
    labelsRow.classList.add('label-row');
    const labelNames = [
      'Very Noticeable',
      'Slightly Noticeable',
      'Almost Unnoticeable',
      'Unnoticeable',
    ];
    labelNames.forEach((labelName) => {
      const labelDiv = document.createElement('div');
      labelDiv.textContent = labelName;
      labelsRow.appendChild(labelDiv);
    });
    radioContainer.appendChild(labelsRow);
    // Insert radio buttons and labels after existing images
    imageContainer.appendChild(radioContainer);
  });
}

function disableSortable() {
  sortable.destroy();
}

/**
 * Sends the ranking data to the backend.
 */
async function sendDataToBackend() {
  try {
    const response = await fetch('/submit-ranking', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        rankingPerMethod: rankingData,
        rankingPerImage: images,
      }),
    });
    // Check if request was successful
    if (response.ok) {
      const h3Element = document.querySelector('h3');
      h3Element.textContent = 'Thank you for your time, that is all.';
    } else {
      console.error('Failed to send data.');
    }
  } catch (error) {
    console.error('Error:', error);
  }
}

renderImages(0);
