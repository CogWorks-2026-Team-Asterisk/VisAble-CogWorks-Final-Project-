const uploadModeBtn = document.getElementById('uploadModeBtn');
const cameraModeBtn = document.getElementById('cameraModeBtn');

const uploadPanel = document.getElementById('uploadPanel');
const cameraPanel = document.getElementById('cameraPanel');
const previewPanel = document.getElementById('previewPanel');
const resultPanel = document.getElementById('resultPanel');

const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');

const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const captureBtn = document.getElementById('captureBtn');
const cameraError = document.getElementById('cameraError');

const previewImage = document.getElementById('previewImage');
const submitBtn = document.getElementById('submitBtn');
const clearBtn = document.getElementById('clearBtn');

const resultText = document.getElementById('resultText');
const audioBtn = document.getElementById('audioBtn');
const stopAudioBtn = document.getElementById('stopAudioBtn');
const audioNote = document.getElementById('audioNote');
const player = document.getElementById('player');

const loading = document.getElementById('loading');
const loadingText = document.getElementById('loadingText');

let selectedBlob = null;
let previewUrl = null;
let stream = null;

/* modes */

function setMode(mode) {
    const camera = mode === 'camera';

    uploadModeBtn.classList.toggle('is-active', !camera);
    cameraModeBtn.classList.toggle('is-active', camera);
    uploadPanel.classList.toggle('hidden', camera);
    cameraPanel.classList.toggle('hidden', !camera);

    clearSelection();

    if (camera) {
        startCamera();
    } else {
        stopCamera();
    }
}

uploadModeBtn.addEventListener('click', () => setMode('upload'));
cameraModeBtn.addEventListener('click', () => setMode('camera'));

/* upload */

dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        fileInput.click();
    }
});

dropZone.addEventListener('dragover', (event) => {
    event.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));

dropZone.addEventListener('drop', (event) => {
    event.preventDefault();
    dropZone.classList.remove('dragover');

    const file = event.dataTransfer.files[0];
    if (file) {
        selectImage(file);
    }
});

fileInput.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        selectImage(file);
    }
});

/* camera */

async function startCamera() {
    cameraError.classList.add('hidden');

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        showCameraError('This browser does not support camera access.');
        return;
    }

    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment' },
        });
        video.srcObject = stream;
    } catch (error) {
        showCameraError(
            'Could not open the camera. Allow camera access in your browser, and make sure ' +
            'the page is served over http://localhost rather than opened as a file.'
        );
    }
}

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach((track) => track.stop());
        stream = null;
    }
    video.srcObject = null;
}

function showCameraError(message) {
    cameraError.textContent = message;
    cameraError.classList.remove('hidden');
    captureBtn.disabled = true;
}

captureBtn.addEventListener('click', () => {
    if (!video.videoWidth) {
        showCameraError('The camera is still starting up. Try again in a moment.');
        return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);

    canvas.toBlob((blob) => {
        if (blob) {
            selectImage(blob);
        }
    }, 'image/jpeg', 0.92);
});

/* selection */

function selectImage(blob) {
    clearPreviewUrl();

    selectedBlob = blob;
    previewUrl = URL.createObjectURL(blob);
    previewImage.src = previewUrl;

    previewPanel.classList.remove('hidden');
    resultPanel.classList.add('hidden');
    stopAudio();
}

function clearPreviewUrl() {
    if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
        previewUrl = null;
    }
}

function clearSelection() {
    clearPreviewUrl();
    selectedBlob = null;
    fileInput.value = '';
    previewPanel.classList.add('hidden');
    resultPanel.classList.add('hidden');
    stopAudio();
}

clearBtn.addEventListener('click', clearSelection);

/* describe */

submitBtn.addEventListener('click', async () => {
    if (!selectedBlob) return;

    showLoading('Generating description…');
    submitBtn.disabled = true;

    const formData = new FormData();
    formData.append('image', selectedBlob, 'diagram.png');

    try {
        const response = await fetch('/api/describe', { method: 'POST', body: formData });
        const data = await response.json();

        resultText.textContent = response.ok
            ? data.caption
            : data.error || 'The description could not be generated.';
    } catch (error) {
        resultText.textContent = 'Could not reach the server. Make sure server.py is running.';
    } finally {
        hideLoading();
        submitBtn.disabled = false;
        resultPanel.classList.remove('hidden');
        resultText.focus();
    }
});

/* audio */

audioBtn.addEventListener('click', async () => {
    const text = resultText.textContent.trim();
    if (!text) return;

    audioNote.classList.add('hidden');
    audioBtn.disabled = true;
    showLoading('Creating audio…');

    try {
        const response = await fetch('/api/speak', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text }),
        });

        if (!response.ok) {
            const data = await response.json().catch(() => ({}));
            speakInBrowser(text, data.error);
            return;
        }

        const blob = await response.blob();
        player.src = URL.createObjectURL(blob);
        player.play();
        stopAudioBtn.classList.remove('hidden');
    } catch (error) {
        speakInBrowser(text, 'Could not reach the audio service.');
    } finally {
        hideLoading();
        audioBtn.disabled = false;
    }
});

function speakInBrowser(text, reason) {
    if (!('speechSynthesis' in window)) {
        showAudioNote(reason || 'Audio is not available.');
        return;
    }

    showAudioNote(
        (reason ? reason + ' ' : '') +
        'Using the built-in browser voice instead. Set OPENAI_API_KEY before starting ' +
        'server.py to use the higher quality voice from text_to_audio.py.'
    );

    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(new SpeechSynthesisUtterance(text));
    stopAudioBtn.classList.remove('hidden');
}

function showAudioNote(message) {
    audioNote.textContent = message;
    audioNote.classList.remove('hidden');
}

stopAudioBtn.addEventListener('click', stopAudio);

function stopAudio() {
    player.pause();

    if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
    }

    stopAudioBtn.classList.add('hidden');
}

player.addEventListener('ended', () => stopAudioBtn.classList.add('hidden'));

/* loading */

function showLoading(message) {
    loadingText.textContent = message;
    loading.classList.remove('hidden');
}

function hideLoading() {
    loading.classList.add('hidden');
}
