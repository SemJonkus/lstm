let model;
let wordIndex = {};
let indexWord = {};
let vocabularySize = 0;
let sequenceLength = 5;
let isAutoRunning = false;
let autoInterval;
let modelLoaded = false;
const inputElement = document.getElementById("text-eingabe");
const predictionDiv = document.getElementById("naechste-woerter");
const trainBtn = document.getElementById("btn-train");
const trainingSection = document.getElementById("training-status");
const visDiv = document.getElementById("vis-training");

function showStatus(msg, timeout = 2000) {
  let overlay = document.getElementById("status-overlay");
  if (!overlay) {
    overlay = document.createElement("div");
    overlay.id = "status-overlay";
    overlay.style.position = "fixed";
    overlay.style.top = "20px";
    overlay.style.left = "50%";
    overlay.style.transform = "translateX(-50%)";
    overlay.style.background = "#2a3a5e";
    overlay.style.color = "#fff";
    overlay.style.padding = "12px 32px";
    overlay.style.borderRadius = "8px";
    overlay.style.zIndex = "9999";
    overlay.style.fontSize = "1.2em";
    document.body.appendChild(overlay);
  }
  overlay.textContent = msg;
  overlay.style.display = "block";
  setTimeout(() => { overlay.style.display = "none"; }, timeout);
}

function updateTrainingUI() {
  if (modelLoaded) {
    trainBtn.style.display = "none";
    trainingSection.style.display = "none";
    visDiv.style.display = "none";
  } else {
    trainBtn.style.display = "inline-block";
    trainingSection.style.display = "block";
    visDiv.style.display = "block";
  }
}

async function loadTextData() {
  const response = await fetch("Final.txt");
  const text = await response.text();
  const tokens = tokenize(text);
  const uniqueWords = [...new Set(tokens)];
  vocabularySize = uniqueWords.length;
  uniqueWords.forEach((w, i) => {
    wordIndex[w] = i;
    indexWord[i] = w;
  });
  return tokens;
}

function tokenize(text) {
  return text
    .toLowerCase()
    .replace(/[\r\n\-]+/g, " ")
    .replace(/\b\d+\b/g, "")
    .replace(/[^\wäöüß ]+/g, "")
    .split(/\s+/)
    .filter(w => w.length > 1);
}

function createSequences(tokens) {
  const inputs = [], labels = [];
  for (let i = 0; i < tokens.length - sequenceLength; i++) {
    const seq = tokens.slice(i, i + sequenceLength);
    const label = tokens[i + sequenceLength];
    if (seq.every(w => wordIndex[w] !== undefined) && wordIndex[label] !== undefined) {
      inputs.push(seq.map(w => wordIndex[w]));
      labels.push(wordIndex[label]);
    }
  }
  return { inputs, labels };
}

async function createModel() {
  model = tf.sequential();
  model.add(tf.layers.embedding({ inputDim: vocabularySize, outputDim: 50, inputLength: sequenceLength }));
  model.add(tf.layers.lstm({ units: 100, returnSequences: true }));
  model.add(tf.layers.lstm({ units: 100 }));
  model.add(tf.layers.dense({ units: vocabularySize, activation: "softmax" }));

  model.compile({
    loss: "sparseCategoricalCrossentropy",
    optimizer: tf.train.adam(0.001),
    metrics: ["accuracy"]
  });
}

async function trainModel(inputs, labels) {
  trainBtn.disabled = true;
  showStatus("Training läuft ...");
  const xs = tf.tensor2d(inputs, [inputs.length, sequenceLength], 'float32');
  const ys = tf.tensor1d(labels);

  await model.fit(xs, ys, {
    epochs: 60,
    batchSize: 32,
    validationSplit: 0.1,
    callbacks: tfvis.show.fitCallbacks(
      visDiv,
      ["loss", "val_loss", "acc", "val_acc"],
      { callbacks: ["onEpochEnd"] }
    )
  });

  xs.dispose();
  ys.dispose();
  trainBtn.disabled = false;
  showStatus("Training abgeschlossen!");
  updateTrainingUI();
}

function predictNextWords(prompt, k = 10) {
  if (!model) {
    showStatus("Modell ist nicht geladen oder trainiert!", 2500);
    return [];
  }
  const tokens = tokenize(prompt);
  let inputTokens = tokens.slice(-sequenceLength).map(w => wordIndex[w]);
  inputTokens = inputTokens.map(idx => idx === undefined ? 0 : idx);

  while (inputTokens.length < sequenceLength) inputTokens.unshift(0);

  const inputTensor = tf.tensor2d([inputTokens], [1, sequenceLength], 'int32');
  const prediction = model.predict(inputTensor);
  const probs = prediction.dataSync();
  inputTensor.dispose();

  const topIndices = Array.from({ length: probs.length }, (_, i) => [i, probs[i]])
    .filter(([i, p]) => typeof p === 'number' && !isNaN(p))
    .sort((a, b) => b[1] - a[1])
    .slice(0, k);

  const sorted = topIndices.map(([i, p]) => ({ word: indexWord[i] || `?(${i})`, prob: p }));
  return sorted;
}

function updatePredictionDisplay(predictions) {
  predictionDiv.innerHTML = "";
  if (predictions.length === 0) {
    const info = document.createElement("div");
    info.textContent = `Keine sinnvollen Vorschläge möglich.`;
    info.style.color = "#b71c1c";
    predictionDiv.appendChild(info);
    return;
  }
  predictions.forEach(pred => {
    const btn = document.createElement("button");
    btn.textContent = `${pred.word} (${(pred.prob * 100).toFixed(2)}%)`;
    btn.className = "suggestion-btn";
    btn.onclick = () => {
      inputElement.value += " " + pred.word;
      handlePrediction();
    };
    predictionDiv.appendChild(btn);
  });
}

function handlePrediction() {
  const predictions = predictNextWords(inputElement.value);
  updatePredictionDisplay(predictions);
}

function handleWeiter() {
  const predictions = predictNextWords(inputElement.value);
  if (predictions.length > 0) {
    inputElement.value += " " + predictions[0].word;
    handlePrediction();
  }
}

function handleAuto() {
  if (isAutoRunning) return;
  isAutoRunning = true;
  showStatus("Automatische Vervollständigung läuft ...");
  autoInterval = setInterval(() => {
    handleWeiter();
    if (inputElement.value.split(" ").length > 100) handleStop();
  }, 700);
}

function handleStop() {
  clearInterval(autoInterval);
  isAutoRunning = false;
  showStatus("Automatik gestoppt.", 1200);
}

function handleReset() {
  inputElement.value = "";
  predictionDiv.innerHTML = "";
  handleStop();
}

async function main() {
  showStatus("Lade Daten ...");
  const tokens = await loadTextData();
  const { inputs, labels } = createSequences(tokens);

try {
  const res = await fetch('mein-lstm-modell.json');
  if (!res.ok) throw new Error("Nicht gefunden");
  model = await tf.loadLayersModel('mein-lstm-modell.json');
  modelLoaded = true; // <--- HIER
  showStatus("Modell geladen!", 1800);
} catch (err) {
  await createModel();
  modelLoaded = false; // <--- HIER
  showStatus("Neues Modell erstellt – bitte trainieren!", 1800);
}

  updateTrainingUI();

  trainBtn.onclick = async () => {
    trainBtn.disabled = true;
    await trainModel(inputs, labels);
    await model.save('downloads://mein-lstm-modell');
    showStatus("Training abgeschlossen! Modell gespeichert.", 2000);
    trainBtn.disabled = false;
    updateTrainingUI();
  };

  document.getElementById("btn-vorhersage").onclick = handlePrediction;
  document.getElementById("btn-weiter").onclick = handleWeiter;
  document.getElementById("btn-auto").onclick = handleAuto;
  document.getElementById("btn-stop").onclick = handleStop;
  document.getElementById("btn-reset").onclick = handleReset;
}



main();
