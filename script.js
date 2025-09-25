/**************************************************************
 * Full client-side OCR page (Tesseract.js v4 + OpenCV.js)
 * Optimized for speed + accuracy:
 * - All variants + all PSMs still tested (accuracy unchanged)
 * - OCR runs in parallel → much faster overall
 * - Deskew + CLAHE kept for max accuracy
 **************************************************************/

// UI elements
const fileInput = document.getElementById('fileInput');
const btnExtract = document.getElementById('btnExtract');
const btnCopy = document.getElementById('btnCopy');
const btnDownload = document.getElementById('btnDownload');
const btnClear = document.getElementById('btnClear');
const btnReprocess = document.getElementById('btnReprocess');
const previewImg = document.getElementById('previewImg');
const previewBox = document.getElementById('previewBox');
const variantsRow = document.getElementById('variantsRow');
const statusEl = document.getElementById('status');
const cvStatus = document.getElementById('cvstatus');
const tStatus = document.getElementById('tstatus');
const logBox = document.getElementById('log');
const resultArea = document.getElementById('resultArea');
const confBadge = document.getElementById('confBadge');
const optDeskew = document.getElementById('optDeskew');
const optShowAll = document.getElementById('optShowAll');
const optNormalize = document.getElementById('optNormalize');

// canvases (hidden)
const srcCanvas = document.getElementById('srcCanvas');
const procCanvas = document.getElementById('procCanvas');

// state
let cvReady = false;
let tessReady = false;
let lastFile = null;

// helper log
function log(...args) {
  console.log(...args);
  let txt = args.map(a => (typeof a === 'string' ? a : JSON.stringify(a))).join(' ');
  logBox.innerText = txt + "\n" + logBox.innerText;
}

// enable/disable buttons only when both libs loaded
function updateReady() {
  btnExtract.disabled = !(cvReady && tessReady);
  btnReprocess.disabled = !(cvReady && tessReady);
}

// --- initialize OpenCV ---
if (typeof cv !== 'undefined') {
  cv['onRuntimeInitialized'] = () => {
    cvReady = true;
    cvStatus.innerText = 'ready';
    log('OpenCV loaded');
    updateReady();
  };
} else {
  (function pollCv() {
    if (typeof cv !== 'undefined') {
      cv['onRuntimeInitialized'] = () => {
        cvReady = true;
        cvStatus.innerText = 'ready';
        log('OpenCV loaded (late)');
        updateReady();
      };
    } else {
      setTimeout(pollCv, 200);
    }
  })();
}

// --- initialize Tesseract ---
function initTesseract() {
  tessReady = true;
  tStatus.innerText = 'ready';
  log('Tesseract ready (v4, no worker init needed)');
  updateReady();
}
initTesseract();

// --- file handling ---
document.querySelector('.filelabel').addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (e) => {
  lastFile = e.target.files[0] || null;
  if (lastFile) {
    const url = URL.createObjectURL(lastFile);
    previewImg.src = url;
    resultArea.value = '';
    confBadge.innerText = 'Confidence: —';
    variantsRow.style.display = 'none';
    variantsRow.innerHTML = '';
  }
});

btnCopy.addEventListener('click', () => {
  resultArea.select();
  document.execCommand('copy');
  alert('Text copied to clipboard ✅');
});

btnClear.addEventListener('click', () => {
  resultArea.value = '';
  previewImg.src = '';
  fileInput.value = '';
  lastFile = null;
  confBadge.innerText = 'Confidence: —';
  variantsRow.style.display = 'none';
  variantsRow.innerHTML = '';
});

btnDownload.addEventListener('click', () => {
  const txt = resultArea.value || '';
  const blob = new Blob([txt], { type: 'text/plain' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'extracted.txt';
  a.click();
});

btnReprocess.addEventListener('click', () => {
  if (!lastFile) return alert('Please choose an image first.');
  doExtract(lastFile);
});

btnExtract.addEventListener('click', () => {
  if (!lastFile) return alert('Please choose an image first.');
  doExtract(lastFile);
});

/***********************
 * Image preprocessing (OpenCV)
 ***********************/
function imgToCanvas(file, canvas) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = function () {
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0);
      resolve();
    };
    img.onerror = () => resolve();
    img.src = URL.createObjectURL(file);
  });
}

function computeSkewAngle(mat) {
  try {
    const edges = new cv.Mat();
    cv.Canny(mat, edges, 50, 150);
    const lines = new cv.Mat();
    cv.HoughLinesP(edges, lines, 1, Math.PI / 180, 160, mat.cols / 8, 20);
    const angles = [];
    for (let i = 0; i < lines.rows; ++i) {
      const x1 = lines.data32S[i * 4], y1 = lines.data32S[i * 4 + 1];
      const x2 = lines.data32S[i * 4 + 2], y2 = lines.data32S[i * 4 + 3];
      const angle = (Math.atan2(y2 - y1, x2 - x1) * 180) / Math.PI;
      if (Math.abs(angle) < 45) angles.push(angle);
    }
    edges.delete(); lines.delete();
    if (angles.length === 0) return 0;
    angles.sort((a, b) => a - b);
    return angles[Math.floor(angles.length / 2)];
  } catch (err) {
    console.warn('skew detection failed', err);
    return 0;
  }
}

function rotateMat(src, angle) {
  const center = new cv.Point(src.cols / 2, src.rows / 2);
  const M = cv.getRotationMatrix2D(center, angle, 1);
  const dst = new cv.Mat();
  const dsize = new cv.Size(src.cols, src.rows);
  cv.warpAffine(src, dst, M, dsize, cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar(255, 255, 255, 255));
  M.delete();
  return dst;
}

async function createProcessedVariants(file, doDeskew = true) {
  await imgToCanvas(file, srcCanvas);
  let src = cv.imread(srcCanvas);

  if (src.channels() === 1) cv.cvtColor(src, src, cv.COLOR_GRAY2RGBA);

  const targetMinWidth = 2200;
  if (src.cols < targetMinWidth) {
    const scale = targetMinWidth / src.cols;
    const dsize = new cv.Size(Math.round(src.cols * scale), Math.round(src.rows * scale));
    const resized = new cv.Mat();
    cv.resize(src, resized, dsize, 0, 0, cv.INTER_CUBIC);
    src.delete(); src = resized;
  }

  let gray = new cv.Mat();
  cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

  if (doDeskew) {
    const angle = computeSkewAngle(gray);
    if (Math.abs(angle) > 0.35) {
      const rotated = rotateMat(gray, -angle);
      gray.delete(); gray = rotated;
      log('deskewed by', (-angle).toFixed(2), 'deg');
    } else {
      log('deskew angle negligible', angle.toFixed(2));
    }
  }

  const variants = [];

  try {
    let v0 = new cv.Mat();
    cv.equalizeHist(gray, v0);
    let dst0 = new cv.Mat();
    cv.adaptiveThreshold(v0, dst0, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 21, 10);
    let kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(1, 1));
    cv.morphologyEx(dst0, dst0, cv.MORPH_OPEN, kernel);
    kernel.delete();
    variants.push(dst0);
    v0.delete();
  } catch (e) { log('variant0 failed', e); }

  try {
    let blur = new cv.Mat();
    cv.GaussianBlur(gray, blur, new cv.Size(3, 3), 0, 0, cv.BORDER_DEFAULT);
    let dst1 = new cv.Mat();
    cv.threshold(blur, dst1, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);
    variants.push(dst1);
    blur.delete();
  } catch (e) { log('variant1 failed', e); }

  try {
    let bil = new cv.Mat();
    cv.bilateralFilter(gray, bil, 9, 75, 75, cv.BORDER_DEFAULT);
    let med = new cv.Mat();
    cv.medianBlur(bil, med, 3);
    let dst2 = new cv.Mat();
    cv.adaptiveThreshold(med, dst2, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 31, 12);
    variants.push(dst2);
    bil.delete(); med.delete();
  } catch (e) { log('variant2 failed', e); }

  try {
    let claheBase = new cv.Mat();
    cv.equalizeHist(gray, claheBase);
    let claheDst = new cv.Mat();
    let claheObj = new cv.CLAHE(2.0, new cv.Size(8, 8));
    claheObj.apply(claheBase, claheDst);
    let dst3 = new cv.Mat();
    cv.threshold(claheDst, dst3, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);
    variants.push(dst3);
    claheBase.delete(); claheDst.delete(); claheObj.delete();
  } catch (e) { log('variant3 failed', e); }

  const blobs = [];
  for (let i = 0; i < variants.length; i++) {
    cv.imshow(procCanvas, variants[i]);
    if (i === 0) previewImg.src = procCanvas.toDataURL('image/png');
    const blob = await new Promise(resolve => procCanvas.toBlob(resolve, 'image/png'));
    blobs.push({ blob, mat: variants[i] });
  }

  gray.delete(); src.delete();
  return blobs;
}

function cleanText(s) {
  if (!s) return s;
  s = s.replace(/\u200B/g, '');
  s = s.replace(/[^\S\r\n]+/g, ' ');
  s = s.replace(/[ \t]+\n/g, '\n');
  s = s.replace(/\n{3,}/g, '\n\n');
  s = s.split('\n').map(l => l.trim()).join('\n');
  return s.trim();
}

// --- OCR extraction (optimized) ---
async function doExtract(file) {
  if (!cvReady) return alert('OpenCV not ready yet — please wait a moment and try again.');

  logBox.innerText = '';
  statusEl.innerText = 'Preparing images...';
  confBadge.innerText = 'Confidence: —';
  resultArea.value = '';
  previewImg.src = '';
  variantsRow.style.display = 'none';
  variantsRow.innerHTML = '';

  try {
    const variants = await createProcessedVariants(file, optDeskew.checked);
    statusEl.innerText = `Running OCR on ${variants.length} variants...`;
    log('Created', variants.length, 'variants');

    const psms = ['6', '3', '4', '11'];
    const ocrTasks = [];

    for (let i = 0; i < variants.length; i++) {
      const vb = variants[i].blob;
      for (const psm of psms) {
        ocrTasks.push(
          Tesseract.recognize(vb, 'eng', {
            logger: m => {
              if (m && m.status) {
                tStatus.innerText = m.status + ' ' + Math.round((m.progress || 0) * 100) + '%';
              }
            },
            load_fast: false, // slightly better accuracy
            tessedit_pageseg_mode: psm,
            tessedit_char_whitelist: 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,:-()/%&$ '

          }).then(res => ({
            text: res.data.text || '',
            avgConf: res.data.confidence || 0,
            variant: i + 1,
            psm,
            raw: res
          }))
        );
      }
    }

    // run all OCR calls in parallel
    const results = await Promise.all(ocrTasks);

    results.sort((a, b) => b.avgConf - a.avgConf);
    const best = results[0] || { text: '', avgConf: 0 };
    let out = best.text || '';
    if (optNormalize.checked) out = cleanText(out);
    resultArea.value = out;
    confBadge.innerText = `Confidence: ${best.avgConf ? best.avgConf.toFixed(1) : '—'}`;

    if (optShowAll.checked) {
      variantsRow.innerHTML = '';
      for (let i = 0; i < variants.length; i++) {
        cv.imshow(procCanvas, variants[i].mat);
        const dataUrl = procCanvas.toDataURL('image/png');
        const img = document.createElement('img');
        img.src = dataUrl;
        img.className = 'variant-thumb';
        img.title = `variant ${i + 1}`;
        img.onclick = () => { previewImg.src = dataUrl; };
        variantsRow.appendChild(img);
      }
      variantsRow.style.display = 'flex';
      const bestMat = variants[Math.max(0, best.variant - 1)].mat;
      cv.imshow(procCanvas, bestMat);
      previewImg.src = procCanvas.toDataURL('image/png');
    } else {
      const bestMat = variants[Math.max(0, best.variant - 1)].mat;
      cv.imshow(procCanvas, bestMat);
      previewImg.src = procCanvas.toDataURL('image/png');
    }

    statusEl.innerText = 'Done.';
    log('Best result:', `variant=${best.variant} psm=${best.psm} conf=${best.avgConf.toFixed(2)}`);

    for (let v of variants) try { v.mat.delete(); } catch (e) { }
  } catch (err) {
    console.error(err);
    alert('Error during processing: ' + (err.message || err));
    statusEl.innerText = 'Error';
    log('Error during processing:', err);
  }
}
