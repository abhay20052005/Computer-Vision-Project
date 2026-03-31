// Elements
const uploadSection = document.getElementById('uploadSection');
const fileInput = document.getElementById('fileInput');
const previewSection = document.getElementById('previewSection');
const imagePreview = document.getElementById('imagePreview');
const predictBtn = document.getElementById('predictBtn');
const resultSection = document.getElementById('resultSection');
const predictedClass = document.getElementById('predictedClass');
const confidenceText = document.getElementById('confidenceText');
const confidenceBar = document.getElementById('confidenceBar');
const resultMessage = document.getElementById('resultMessage');
const errorAlert = document.getElementById('errorAlert');
const errorMessage = document.getElementById('errorMessage');

let selectedFile = null;

// Drag and Drop Events
uploadSection.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadSection.classList.add('dragover');
});

uploadSection.addEventListener('dragleave', () => {
    uploadSection.classList.remove('dragover');
});

uploadSection.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadSection.classList.remove('dragover');
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFileSelection(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
        handleFileSelection(e.target.files[0]);
    }
});

function handleFileSelection(file) {
    if (!file.type.match('image.*')) {
        showError("Please select a valid image file (JPG, PNG, etc).");
        return;
    }

    // Hide error if visible
    errorAlert.classList.add('d-none');
    
    // Read and display image
    selectedFile = file;
    const reader = new FileReader();
    
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        uploadSection.classList.add('d-none');
        previewSection.classList.remove('d-none');
        resultSection.classList.add('d-none'); // Hide results if re-selecting
    };
    
    reader.readAsDataURL(file);
}

function clearSelection() {
    selectedFile = null;
    fileInput.value = '';
    previewSection.classList.add('d-none');
    uploadSection.classList.remove('d-none');
    resultSection.classList.add('d-none');
    errorAlert.classList.add('d-none');
}

function resetForm() {
    clearSelection();
    // Reset progress bar
    confidenceBar.style.width = '0%';
}

async function uploadAndPredict() {
    if (!selectedFile) return;

    // Create form data
    const formData = new FormData();
    formData.append('file', selectedFile);

    // Update UI for loading state
    predictBtn.disabled = true;
    const originalBtnText = predictBtn.innerHTML;
    predictBtn.innerHTML = '<span class="loader me-2"></span> Analyzing...';
    errorAlert.classList.add('d-none');
    resultSection.classList.add('d-none');

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            displayResults(data.result, data.confidence);
        } else {
            showError(data.error || "An unknown error occurred during prediction.");
        }
    } catch (error) {
        showError("Network error: Could not reach the server.");
        console.error("Prediction Error:", error);
    } finally {
        predictBtn.disabled = false;
        predictBtn.innerHTML = originalBtnText;
    }
}

function displayResults(result, confidence) {
    // Reveal results section
    resultSection.classList.remove('d-none');
    
    // Set text
    predictedClass.innerHTML = `<i class="bi bi-${result.toLowerCase() === 'cat' ? 'chat-right-heart' : 'gitlab'} me-2"></i> It's a ${result}!`;
    
    // Determine color based on confidence
    let colorClass, accentColor;
    if (result.toLowerCase() === 'cat') {
        colorClass = 'bg-warning';
        accentColor = '#ffc107'; // subtle yellow/warning color for cat
    } else {
        colorClass = 'bg-info';
        accentColor = '#0dcaf0'; // light blue for dog
    }
    
    // Update progress bar class
    confidenceBar.className = `progress-bar progress-bar-striped progress-bar-animated ${colorClass}`;
    predictedClass.className = `fs-4 fw-bold text-${result.toLowerCase() === 'cat' ? 'warning' : 'info'}`;
    
    // Animate progress bar filling
    setTimeout(() => {
        const confNum = parseFloat(confidence).toFixed(1);
        confidenceBar.style.width = `${confNum}%`;
        confidenceText.textContent = `${confNum}%`;
        
        // Detailed message based on confidence
        if (confNum > 95) {
            resultMessage.innerHTML = `I'm highly confident this is a <strong>${result.toLowerCase()}</strong>!`;
        } else if (confNum > 80) {
            resultMessage.innerHTML = `This looks quite like a <strong>${result.toLowerCase()}</strong>.`;
        } else {
            resultMessage.innerHTML = `I think this is a <strong>${result.toLowerCase()}</strong>, but I'm not entirely sure.`;
        }
    }, 100);
    
    // Scroll to results if needed
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function showError(msg) {
    errorMessage.textContent = msg;
    errorAlert.classList.remove('d-none');
}
