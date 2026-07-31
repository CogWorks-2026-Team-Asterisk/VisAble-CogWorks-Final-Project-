const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const previewSection = document.getElementById('previewSection');
const imagePreview = document.getElementById('imagePreview');
const generateBtn = document.getElementById('generateBtn');
const loadingSection = document.getElementById('loading');
const outputSection = document.getElementById('outputSection');
const captionText = document.getElementById('captionText');
const resetBtn = document.getElementById('resetBtn');

let selectedFile = null;

dropZone.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
    }
});

function handleFile(file) {
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        dropZone.classList.add('hidden');
        previewSection.classList.remove('hidden');
    };
    reader.readAsDataURL(file);
}

generateBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    previewSection.classList.add('hidden');
    loadingSection.classList.remove('hidden');

    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
        const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        captionText.textContent = data.caption || 'failed';
    } catch (error) {
        captionText.textContent = 'error connecting to backend';
    } finally {
        loadingSection.classList.add('hidden');
        outputSection.classList.remove('hidden');
    }
});

resetBtn.addEventListener('click', () => {
    selectedFile = null;
    fileInput.value = '';
    outputSection.classList.add('hidden');
    dropZone.classList.remove('hidden');
});