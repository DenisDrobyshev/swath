"use strict";

const els = {
  status: document.getElementById("status"),
  model: document.getElementById("model"),
  modelNote: document.getElementById("model-note"),
  alpha: document.getElementById("alpha"),
  alphaOut: document.getElementById("alpha-out"),
  tile: document.getElementById("tile"),
  tta: document.getElementById("tta"),
  run: document.getElementById("run"),
  hint: document.getElementById("hint"),
  file: document.getElementById("file"),
  dropzone: document.getElementById("dropzone"),
  stage: document.getElementById("stage"),
  canvas: document.getElementById("canvas"),
  spinner: document.getElementById("spinner"),
  results: document.getElementById("results"),
  legend: document.getElementById("legend"),
  geo: document.getElementById("geo"),
  downloads: document.getElementById("downloads"),
  timing: document.getElementById("timing"),
  version: document.getElementById("version"),
  tabs: Array.from(document.querySelectorAll(".tab")),
};

const state = {
  models: [],
  file: null,
  result: null,
  imageBitmap: null,
  maskBitmap: null,
  view: "overlay",
};

function setHint(text, bad) {
  els.hint.textContent = text;
  els.hint.classList.toggle("bad", Boolean(bad));
}

function formatBytes(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(0) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

function formatArea(squareMetres) {
  if (squareMetres >= 1e6) return (squareMetres / 1e6).toFixed(2) + " km²";
  if (squareMetres >= 1e4) return (squareMetres / 1e4).toFixed(2) + " ha";
  return Math.round(squareMetres) + " m²";
}

function loadImage(source) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("could not decode the returned image"));
    image.src = source;
  });
}

async function boot() {
  try {
    const health = await fetch("/api/health").then((r) => r.json());
    els.version.textContent = "swath " + health.version;
    const geo = health.geo_support ? "geo on" : "geo off";
    els.status.textContent = `${health.device} · ${health.models} model${
      health.models === 1 ? "" : "s"
    } · ${geo}`;
    if (!health.models) {
      els.status.classList.add("bad");
      setHint("No checkpoint is loaded. Start the service with --checkpoint.", true);
    }
  } catch (error) {
    els.status.textContent = "service unreachable";
    els.status.classList.add("bad");
    return;
  }

  try {
    const payload = await fetch("/api/models").then((r) => r.json());
    state.models = payload.models || [];
    els.model.innerHTML = "";
    for (const model of state.models) {
      const option = document.createElement("option");
      option.value = model.id;
      option.textContent = `${model.title} (${model.id})`;
      els.model.appendChild(option);
    }
    describeModel();
  } catch (error) {
    setHint("Could not list the models: " + error.message, true);
  }
}

function currentModel() {
  return state.models.find((model) => model.id === els.model.value) || state.models[0];
}

function describeModel() {
  const model = currentModel();
  if (!model) {
    els.modelNote.textContent = "";
    return;
  }
  const parts = [
    `${model.classes.length} classes`,
    `${(model.parameters / 1e6).toFixed(1)}M params`,
  ];
  if (model.metrics && typeof model.metrics.mean_iou === "number") {
    parts.push(`mIoU ${model.metrics.mean_iou.toFixed(3)}`);
  }
  els.modelNote.textContent = parts.join(" · ");
}

function acceptFile(file) {
  if (!file) return;
  state.file = file;
  state.result = null;
  state.imageBitmap = null;
  state.maskBitmap = null;
  els.results.hidden = true;
  els.timing.textContent = "";
  els.canvas.hidden = true;
  els.dropzone.hidden = false;
  els.run.disabled = false;
  setHint(`${file.name} · ${formatBytes(file.size)}`);
}

/** Draw the current view. The overlay is composed here, so opacity is free. */
function paint() {
  const canvas = els.canvas;
  const base = state.imageBitmap;
  if (!base) return;

  canvas.width = base.naturalWidth;
  canvas.height = base.naturalHeight;
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);

  if (state.view !== "mask") {
    context.drawImage(base, 0, 0);
  }
  if (state.view !== "image" && state.maskBitmap) {
    context.globalAlpha = state.view === "mask" ? 1 : Number(els.alpha.value) / 100;
    context.imageSmoothingEnabled = false;
    context.drawImage(state.maskBitmap, 0, 0, canvas.width, canvas.height);
    context.globalAlpha = 1;
  }
  canvas.hidden = false;
  els.dropzone.hidden = true;
}

function setView(view) {
  state.view = view;
  for (const tab of els.tabs) tab.classList.toggle("active", tab.dataset.view === view);
  paint();
}

function renderResult(result) {
  state.result = result;
  els.timing.textContent = `${result.width}×${result.height} px · ${result.seconds.toFixed(2)} s`;

  const rows = result.classes.filter((row) => row.pixels > 0).sort((a, b) => b.share - a.share);

  els.legend.innerHTML = "";
  for (const row of rows) {
    const item = document.createElement("li");
    const percent = (row.share * 100).toFixed(1) + "%";
    item.innerHTML = `
      <div class="row">
        <span class="swatch" style="background:${row.color}"></span>
        <span class="name"></span>
        <span class="value">${percent}</span>
      </div>
      <div class="track"><div class="fill" style="width:${Math.max(
        row.share * 100,
        1.5
      )}%;background:${row.color}"></div></div>`;
    item.querySelector(".name").textContent = row.class;
    if (typeof row.area_m2 === "number") {
      const area = document.createElement("div");
      area.className = "area";
      area.textContent = formatArea(row.area_m2);
      item.appendChild(area);
    }
    els.legend.appendChild(item);
  }

  if (result.georeferenced && result.geo) {
    const size = result.geo.pixel_size.map((v) => v.toFixed(3)).join(" × ");
    els.geo.innerHTML = `Georeferenced · <code></code> · pixel ${size} map units`;
    els.geo.querySelector("code").textContent = result.geo.crs;
  } else {
    els.geo.textContent = "No coordinate reference system in the upload — areas are in pixels.";
  }

  els.downloads.innerHTML = "";
  const labels = { mask_png: "Mask PNG", geotiff: "Mask GeoTIFF", geojson: "Polygons GeoJSON" };
  for (const [key, href] of Object.entries(result.downloads || {})) {
    const link = document.createElement("a");
    link.href = href;
    link.textContent = labels[key] || key;
    link.setAttribute("download", "");
    els.downloads.appendChild(link);
  }

  els.results.hidden = false;
  setView("overlay");
}

async function segment() {
  if (!state.file) return;
  els.run.disabled = true;
  els.spinner.hidden = false;
  setHint("Running the model…");

  const body = new FormData();
  body.append("file", state.file);
  body.append("model_id", els.model.value || "");
  body.append("tile", els.tile.value);
  body.append("overlap", String(Math.round(Number(els.tile.value) / 4)));
  body.append("tta", els.tta.checked ? "true" : "false");

  try {
    const response = await fetch("/api/segment", { method: "POST", body });
    if (!response.ok) {
      const detail = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(detail.detail || response.statusText);
    }
    const payload = await response.json();
    [state.imageBitmap, state.maskBitmap] = await Promise.all([
      loadImage(payload.image_png),
      loadImage(payload.mask_png),
    ]);
    renderResult(payload);
    setHint(`${state.file.name} · done`);
  } catch (error) {
    setHint(error.message, true);
  } finally {
    els.spinner.hidden = true;
    els.run.disabled = false;
  }
}

els.file.addEventListener("change", (event) => acceptFile(event.target.files[0]));
els.run.addEventListener("click", segment);
els.model.addEventListener("change", describeModel);

els.alpha.addEventListener("input", () => {
  els.alphaOut.textContent = els.alpha.value + "%";
  if (state.view === "overlay") paint();
});

for (const tab of els.tabs) {
  tab.addEventListener("click", () => setView(tab.dataset.view));
}

for (const type of ["dragenter", "dragover"]) {
  els.stage.addEventListener(type, (event) => {
    event.preventDefault();
    els.dropzone.classList.add("over");
  });
}
for (const type of ["dragleave", "drop"]) {
  els.stage.addEventListener(type, (event) => {
    event.preventDefault();
    els.dropzone.classList.remove("over");
  });
}
els.stage.addEventListener("drop", (event) => {
  const file = event.dataTransfer && event.dataTransfer.files[0];
  if (file) acceptFile(file);
});

boot();
