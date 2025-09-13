// main.js
import {
  HandLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

// -------------------------------
// Elements (must match your HTML)
const VIDEO = document.getElementById("webcam");
const PREDICTED_WORD = document.getElementById("predictedWord");
const CONFIDENCE = document.getElementById("confidence");
const STACKED_WORDS = document.getElementById("stacked-words");
const INTERPRETED_TEXT = document.getElementById("interpreted-text");

const startBtn = document.getElementById("start-btn");
const startDetectBtn = document.getElementById("start-detect-btn");
const pauseDetectBtn = document.getElementById("pause-detect-btn"); // Diganti dari 'stop-btn'
const deleteBtn = document.getElementById("delete-btn");
const stopCameraBtn = document.getElementById("stop-camera-btn"); // Tombol baru
const interpretControls = document.getElementById("interpret-controls");
const interpretBtn = document.getElementById("interpret-btn");
const clearBtn = document.getElementById("clear-btn");

// -------------------------------
// Config / state
const SEQ_LEN = 20;
const CONF_THRESHOLD = 0.7;

const PREDICT_API = "https://blaziooon-kygb-sft.hf.space/predict";
const SENTENCE_API = "https://blaziooon-kygb-sft.hf.space/sentence";

let handLandmarker = null;
let running = false;
let frameBuffer = [];
let detectedWords = [];

// prevent overlapping predict requests
let predictPending = false;

// -------------------------------
// normalize + feature builder
function normalizeHand(landmarks, w, h) {
  if (!landmarks || landmarks.length === 0) {
    return new Array(63).fill(0.0);
  }
  const pts = landmarks.map((lm) => [lm.x, lm.y, lm.z]);
  const cx = pts[0][0],
    cy = pts[0][1];
  for (let i = 0; i < pts.length; i++) {
    pts[i][0] -= cx;
    pts[i][1] -= cy;
  }
  const px = pts.map((p) => p[0] * w);
  const py = pts.map((p) => p[1] * h);
  const pxMax = Math.max(...px),
    pxMin = Math.min(...px);
  const pyMax = Math.max(...py),
    pyMin = Math.min(...py);
  const scaleRaw = Math.max(pxMax - pxMin, pyMax - pyMin, 1e-3);
  const denom = scaleRaw / Math.max(w, h);
  const flat = [];
  for (let i = 0; i < pts.length; i++) {
    const nx = pts[i][0] / denom;
    const ny = pts[i][1] / denom;
    const nz = pts[i][2];
    flat.push(nx, ny, nz);
  }
  return flat; // length 63
}

function buildFeatFromHands(result, w, h) {
  let leftLm = null,
    rightLm = null;

  const landmarksArr = result.landmarks || result.multiHandLandmarks || [];
  const handednessArr = result.handedness || result.multiHandedness || [];

  if (handednessArr && handednessArr.length > 0) {
    for (let i = 0; i < handednessArr.length; i++) {
      const hd = handednessArr[i];
      const label = (hd.label || hd.categoryName || "")
        .toString()
        .toLowerCase();
      const lmset = landmarksArr[i] || null;
      if (!lmset) continue;
      if (label.includes("left")) leftLm = lmset;
      else if (label.includes("right")) rightLm = lmset;
    }
  }

  if ((!leftLm || !rightLm) && landmarksArr.length > 0) {
    const meanXs = landmarksArr
      .map((lmset) => {
        const sum = lmset.reduce((s, p) => s + (p.x || p[0]), 0);
        return sum / lmset.length;
      })
      .map((v, i) => ({ v, i }));
    meanXs.sort((a, b) => a.v - b.v);
    if (meanXs.length === 1) {
      leftLm = landmarksArr[meanXs[0].i];
    } else if (meanXs.length >= 2) {
      leftLm = landmarksArr[meanXs[0].i];
      rightLm = landmarksArr[meanXs[1].i];
    }
  }

  const L = normalizeHand(leftLm, w, h);
  const R = normalizeHand(rightLm, w, h);
  const presL = leftLm && leftLm.length > 0 ? 1.0 : 0.0;
  const presR = rightLm && rightLm.length > 0 ? 1.0 : 0.0;

  const feat = L.concat(R);
  feat.push(presL, presR);
  return feat; // length 128
}

// -------------------------------
// Initialize MediaPipe HandLandmarker
async function initHandLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
  );

  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    },
    runningMode: "VIDEO",
    numHands: 2,
  });

  console.log("HandLandmarker ready");
}

// -------------------------------
// Start webcam
async function startWebcam() {
  const s = await navigator.mediaDevices.getUserMedia({ video: true });
  VIDEO.srcObject = s;
  await new Promise((resolve) => {
    if (VIDEO.readyState >= 2) resolve();
    else VIDEO.onloadeddata = () => resolve();
  });
}

// -------------------------------
// Stop webcam (fungsi baru)
function stopWebcam() {
  running = false;
  if (VIDEO.srcObject) {
    VIDEO.srcObject.getTracks().forEach((track) => track.stop());
    VIDEO.srcObject = null;
  }
  startBtn.classList.remove("hidden");
  startDetectBtn.classList.add("hidden");
  pauseDetectBtn.classList.add("hidden");
  deleteBtn.classList.add("hidden");
  stopCameraBtn.classList.add("hidden");
  interpretControls.classList.add("hidden");
}

// -------------------------------
// Main detection loop
async function detectLoop() {
  if (!running) return;
  const now = performance.now();
  const res = handLandmarker.detectForVideo(VIDEO, now);

  const w = VIDEO.videoWidth || VIDEO.clientWidth || 640;
  const h = VIDEO.videoHeight || VIDEO.clientHeight || 480;

  const wrapper = {
    landmarks: res.landmarks || res.multiHandLandmarks || [],
    handedness: res.handedness || res.multiHandedness || [],
  };

  const feat = buildFeatFromHands(wrapper, w, h);
  frameBuffer.push(feat);
  if (frameBuffer.length > SEQ_LEN) frameBuffer.shift();

  if (frameBuffer.length === SEQ_LEN && !predictPending) {
    const seqCopy = frameBuffer.slice();
    frameBuffer = [];
    await sendPredictRequest(seqCopy);
  }

  requestAnimationFrame(detectLoop);
}

// -------------------------------
// send /predict to backend
async function sendPredictRequest(seq20x128) {
  predictPending = true;
  try {
    const resp = await fetch(PREDICT_API, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ data: [seq20x128] }),
    });
    const data = await resp.json();

    if (data.predicted_word && typeof data.confidence === "number") {
      console.log(
        `/predict -> {predicted_word: '${data.predicted_word}', confidence: ${data.confidence}}`
      );
    } else {
      console.log(`/predict -> Error: ${data.error || "Unknown error"}`);
    }

    if (data.error) {
      CONFIDENCE.innerText = "-";
      PREDICTED_WORD.innerText = "-";
      return;
    }

    const word = data.predicted_word ?? null;
    const conf = typeof data.confidence === "number" ? data.confidence : null;

    if (conf != null && conf >= CONF_THRESHOLD) {
      CONFIDENCE.innerText = (conf * 100).toFixed(1) + "%";
      PREDICTED_WORD.innerText = word || "-";

      // Tambahkan kata ke dalam daftar jika akurasi > 70% dan kata tidak sama dengan yang terakhir
      if (
        detectedWords.length === 0 ||
        detectedWords[detectedWords.length - 1] !== word
      ) {
        detectedWords.push(word);
        STACKED_WORDS.innerText = detectedWords.join(" ");
      }
    } else {
      CONFIDENCE.innerText = "-";
      PREDICTED_WORD.innerText = "-";
    }
  } catch (e) {
    console.error("Predict request failed:", e);
  } finally {
    predictPending = false;
  }
}

// -------------------------------
// Request sentence from /sentence
async function requestSentence() {
  if (detectedWords.length === 0) {
    INTERPRETED_TEXT.innerText = "(Tidak ada kata terdeteksi)";
    return;
  }
  try {
    const res = await fetch(SENTENCE_API, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ words: detectedWords }),
    });
    const data = await res.json();
    console.log("/sentence ->", data);
    if (data.sentence) {
      INTERPRETED_TEXT.innerText = data.sentence;
    } else if (data.error) {
      INTERPRETED_TEXT.innerText = "(Error: " + data.error + ")";
    } else {
      INTERPRETED_TEXT.innerText = "(Tidak ada kalimat)";
    }
  } catch (e) {
    INTERPRETED_TEXT.innerText = "(Error menghasilkan kalimat)";
  }
}

// -------------------------------
// Buttons wiring
startBtn.addEventListener("click", async () => {
  await initHandLandmarker();
  await startWebcam();
  startBtn.classList.add("hidden");
  startDetectBtn.classList.remove("hidden");
  stopCameraBtn.classList.remove("hidden"); // Tampilkan tombol Matikan Kamera
});

startDetectBtn.addEventListener("click", () => {
  running = true;
  detectLoop();
  startDetectBtn.classList.add("hidden");
  pauseDetectBtn.classList.remove("hidden"); // Tampilkan tombol Jeda Deteksi
  deleteBtn.classList.remove("hidden");
});

pauseDetectBtn.addEventListener("click", () => {
  running = false;
  pauseDetectBtn.classList.add("hidden");
  startDetectBtn.classList.remove("hidden");
});

stopCameraBtn.addEventListener("click", () => {
  stopWebcam();
});

deleteBtn.addEventListener("click", () => {
  if (detectedWords.length > 0) detectedWords.pop();
  STACKED_WORDS.innerText =
    detectedWords.length > 0 ? detectedWords.join(" ") : "-";
});

// Kalimat button
interpretBtn.addEventListener("click", async () => {
  await requestSentence();
});

// Clear button
clearBtn.addEventListener("click", () => {
  detectedWords = [];
  frameBuffer = [];
  PREDICTED_WORD.innerText = "-";
  CONFIDENCE.innerText = "-";
  STACKED_WORDS.innerText = "-";
  INTERPRETED_TEXT.innerText = "-";
  interpretControls.classList.add("hidden");
  console.log("Clear -> reset all");
});

console.log("main.js loaded");
