let model;
let wordIndex = {}; // Wort → Index
let indexWord = {}; // Index → Wort
let vocabularySize = 0;
let sequenceLength = 5;
let isAutoRunning = false;
let autoInterval;

const inputElement = document.getElementById("text-eingabe");
const predictionDiv = document.getElementById("naechste-woerter");

// Lade Text und verarbeite Vokabular
async function loadTextData() {
  const response = await fetch("haeufige_saetze_variiert.txt");
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
  // Ersetze Bindestriche und Zeilenumbrüche durch Leerzeichen
  return text
    .toLowerCase()
    .replace(/[\r\n\-]+/g, " ")           // Zeilenumbrüche und Bindestriche zu Leerzeichen
    .replace(/[^\wäöüß ]+/g, "")          // Entferne sonstige Satzzeichen
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
    optimizer: tf.train.adam(0.01),
    metrics: ["accuracy"]
  });
}

async function trainModel(inputs, labels) {
  const xs = tf.tensor(inputs);
  const ys = tf.tensor(labels);

  await model.fit(xs, ys, {
    epochs: 10,
    batchSize: 32,
    validationSplit: 0.1,
    callbacks: tfvis.show.fitCallbacks(
      document.getElementById("vis-training"),
      ["loss", "val_loss", "acc", "val_acc"],
      { callbacks: ["onEpochEnd"] }
    )
  });

  xs.dispose();
  ys.dispose();
}
function predictNextWords(prompt, k = 5) {
  if (!model) {
    alert("Modell ist nicht geladen oder trainiert!");
    return [];
  }
  const tokens = tokenize(prompt);
  let inputTokens = tokens.slice(-sequenceLength).map(w => wordIndex[w]);
  // Ersetze unbekannte Wörter durch 0 (oder ein anderes häufiges Wort)
  inputTokens = inputTokens.map(idx => idx === undefined ? 0 : idx);
  console.log("Eingabe-Tokens:", tokens);
  console.log("Input-Token-Indices:", inputTokens);

  if (inputTokens.length < sequenceLength) {
    // Fülle mit 0 auf, falls weniger als sequenceLength
    while (inputTokens.length < sequenceLength) inputTokens.unshift(0);
  }

  const inputTensor = tf.tensor2d([inputTokens], [1, sequenceLength], 'int32');
  const prediction = model.predict(inputTensor);
  const probs = prediction.dataSync();
  inputTensor.dispose();
  prediction.dispose();

  const sorted = Array.from(probs.map((p, i) => ({ word: indexWord[i], prob: p })))
    .filter(item => item.word !== undefined)
    .sort((a, b) => b.prob - a.prob)
    .slice(0, k);
  console.log("Vorhersage:", sorted);
  return sorted;
}
function updatePredictionDisplay(predictions) {
  predictionDiv.innerHTML = "";
  // Entferne die Prüfung auf predictions.length === 0
  predictions.forEach(pred => {
    const btn = document.createElement("button");
    btn.textContent = `${pred.word} (${(pred.prob * 100).toFixed(1)}%)`;
    btn.onclick = () => {
      inputElement.value += " " + pred.word;
      handlePrediction(); // Vorschläge nach Klick aktualisieren
    };
    predictionDiv.appendChild(btn);
  });
  // Optional: Zeige einen Hinweis, wenn alle Vorschläge gleich sind (z.B. nur das Dummy-Wort)
  if (predictions.length === 0) {
    const info = document.createElement("div");
    info.textContent = `Keine sinnvollen Vorschläge möglich.`;
    predictionDiv.appendChild(info);
  }
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
  autoInterval = setInterval(() => {
    handleWeiter();
    if (inputElement.value.split(" ").length > 100) handleStop();
  }, 700);
}

function handleStop() {
  clearInterval(autoInterval);
  isAutoRunning = false;
}

function handleReset() {
  inputElement.value = "";
  predictionDiv.innerHTML = "";
  handleStop();
}

async function main() {
  const tokens = await loadTextData();
  const { inputs, labels } = createSequences(tokens);

  // Zeige das Vokabular an der Seite an
  const wortlisteDiv = document.getElementById("wortliste");
  wortlisteDiv.innerHTML = "<b>Erlaubte Wörter:</b><br>" + Object.keys(wordIndex).join(", ");

  // Versuche, das Modell aus Datei zu laden
  try {
    model = await tf.loadLayersModel('mein-lstm-modell.json');
    alert("Modell aus Datei geladen!");
  } catch (err) {
    await createModel();
    alert("Neues Modell erstellt. Bitte trainieren!");
  }

  const trainBtn = document.getElementById("btn-train");
  trainBtn.onclick = async () => {
    trainBtn.disabled = true;
    try {
      await trainModel(inputs, labels);
      await model.save('downloads://mein-lstm-modell');
      alert("Training abgeschlossen! Modell wurde als Download gespeichert. Bitte die Dateien ins Projektverzeichnis legen.");
    } catch (err) {
      alert("Fehler beim Training: " + err.message);
    }
    trainBtn.disabled = false;
    wortlisteDiv.innerHTML = "<b>Erlaubte Wörter:</b><br>" + Object.keys(wordIndex).join(", ");
  };

  document.getElementById("btn-vorhersage").onclick = handlePrediction;
  document.getElementById("btn-weiter").onclick = handleWeiter;
  document.getElementById("btn-auto").onclick = handleAuto;
  document.getElementById("btn-stop").onclick = handleStop;
  document.getElementById("btn-reset").onclick = handleReset;
}

main();