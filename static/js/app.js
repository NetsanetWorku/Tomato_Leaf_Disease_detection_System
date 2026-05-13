/**
 * Tomato Disease Detection — Frontend JavaScript
 * Handles drag-and-drop upload, image preview, API call, and result rendering.
 */

(function () {
  "use strict";

  // ─── DOM References ──────────────────────────────────────────────────────────
  const dropZone        = document.getElementById("dropZone");
  const dropContent     = document.getElementById("dropContent");
  const previewContainer= document.getElementById("previewContainer");
  const previewImage    = document.getElementById("previewImage");
  const fileInput       = document.getElementById("fileInput");
  const changeImageBtn  = document.getElementById("changeImageBtn");
  const detectBtn       = document.getElementById("detectBtn");
  const detectBtnText   = document.getElementById("detectBtnText");
  const detectBtnSpinner= document.getElementById("detectBtnSpinner");

  const resultSection   = document.getElementById("resultSection");
  const resultImage     = document.getElementById("resultImage");
  const resultDisease   = document.getElementById("resultDisease");
  const confidenceBadge = document.getElementById("confidenceBadge");
  const severityBadge   = document.getElementById("severityBadge");
  const resultDescription = document.getElementById("resultDescription");
  const resultTreatment = document.getElementById("resultTreatment");
  const top3Bars        = document.getElementById("top3Bars");
  const tryAnotherBtn   = document.getElementById("tryAnotherBtn");

  const errorToast      = document.getElementById("errorToast");
  const errorMessage    = document.getElementById("errorMessage");
  const toastClose      = document.getElementById("toastClose");

  let selectedFile = null;

  // ─── File Selection ───────────────────────────────────────────────────────────

  fileInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) handleFile(file);
  });

  changeImageBtn.addEventListener("click", () => {
    fileInput.value = "";
    selectedFile = null;
    showDropZone();
    hideResult();
  });

  tryAnotherBtn.addEventListener("click", () => {
    fileInput.value = "";
    selectedFile = null;
    showDropZone();
    hideResult();
    window.scrollTo({ top: 0, behavior: "smooth" });
  });

  // ─── Drag & Drop ─────────────────────────────────────────────────────────────

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("drag-over");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  });

  // ─── Handle File ─────────────────────────────────────────────────────────────

  function handleFile(file) {
    const allowed = ["image/jpeg", "image/png", "image/bmp", "image/webp"];
    if (!allowed.includes(file.type)) {
      showError("Invalid file type. Please upload a JPG, PNG, BMP, or WebP image.");
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      showError("File is too large. Maximum size is 10 MB.");
      return;
    }

    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImage.src = e.target.result;
      showPreview();
    };
    reader.readAsDataURL(file);
  }

  // ─── Detect Button ────────────────────────────────────────────────────────────

  detectBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    setLoading(true);
    hideResult();

    const formData = new FormData();
    formData.append("image", selectedFile);

    try {
      const response = await fetch("/predict", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok || data.error) {
        showError(data.error || "Prediction failed. Please try again.");
        return;
      }

      renderResult(data);
    } catch (err) {
      showError("Network error. Please check your connection and try again.");
    } finally {
      setLoading(false);
    }
  });

  // ─── Render Result ────────────────────────────────────────────────────────────

  function renderResult(data) {
    // Image
    resultImage.src = data.image_url;

    // Disease name
    resultDisease.textContent = data.display_name;

    // Confidence badge
    confidenceBadge.textContent = `${data.confidence.toFixed(1)}% confidence`;

    // Severity badge
    severityBadge.textContent = `Severity: ${data.severity}`;
    severityBadge.className = "severity-badge severity-" + data.severity.toLowerCase();

    // Description & treatment
    resultDescription.textContent = data.description;
    resultTreatment.textContent   = data.treatment;

    // Top-3 bars
    top3Bars.innerHTML = "";
    const fillClasses = ["", "secondary", "tertiary"];
    data.top3.forEach((item, i) => {
      const div = document.createElement("div");
      div.className = "top3-bar-item";
      div.innerHTML = `
        <div class="top3-bar-label">
          <span>${item.display}</span>
          <span>${item.confidence.toFixed(1)}%</span>
        </div>
        <div class="top3-bar-track">
          <div class="top3-bar-fill ${fillClasses[i]}" style="width: 0%"></div>
        </div>
      `;
      top3Bars.appendChild(div);

      // Animate bar fill
      requestAnimationFrame(() => {
        setTimeout(() => {
          div.querySelector(".top3-bar-fill").style.width = item.confidence + "%";
        }, 50 + i * 100);
      });
    });

    showResult();
    resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // ─── UI State Helpers ─────────────────────────────────────────────────────────

  function showDropZone() {
    dropContent.style.display = "block";
    previewContainer.style.display = "none";
  }

  function showPreview() {
    dropContent.style.display = "none";
    previewContainer.style.display = "flex";
  }

  function showResult() {
    resultSection.style.display = "block";
  }

  function hideResult() {
    resultSection.style.display = "none";
  }

  function setLoading(loading) {
    detectBtn.disabled = loading;
    detectBtnText.style.display   = loading ? "none"         : "inline";
    detectBtnSpinner.style.display= loading ? "inline-block" : "none";
  }

  // ─── Error Toast ──────────────────────────────────────────────────────────────

  function showError(msg) {
    errorMessage.textContent = msg;
    errorToast.style.display = "flex";
    setTimeout(hideError, 5000);
  }

  function hideError() {
    errorToast.style.display = "none";
  }

  toastClose.addEventListener("click", hideError);

})();
