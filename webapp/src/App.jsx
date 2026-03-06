import { useEffect, useMemo, useRef, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

function createDataUrl(base64Body) {
  if (base64Body.startsWith("data:image")) {
    return base64Body;
  }
  return `data:image/png;base64,${base64Body}`;
}

export default function App() {
  const [file, setFile] = useState(null);
  const [sourceUrl, setSourceUrl] = useState("");
  const [resultUrl, setResultUrl] = useState("");
  const [scaleFactor, setScaleFactor] = useState("2.0");
  const [modelName, setModelName] = useState("real_esrgan_x2");
  const [slider, setSlider] = useState(50);
  const [latencyMs, setLatencyMs] = useState(null);
  const [status, setStatus] = useState("idle");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [progressPct, setProgressPct] = useState(0);
  const [frameAspect, setFrameAspect] = useState("4 / 3");
  const [backendMode, setBackendMode] = useState("unknown");
  const [backendReady, setBackendReady] = useState(false);
  const [backendModel, setBackendModel] = useState("-");
  const comparePanelRef = useRef(null);
  const fileInputRef = useRef(null);
  const isAnimeModel = modelName === "anime_gan_hayao";

  const hasBoth = useMemo(() => Boolean(sourceUrl && resultUrl), [sourceUrl, resultUrl]);
  const isFinalizing = isLoading && progressPct >= 95;

  useEffect(() => {
    if (!isLoading) return undefined;
    const timer = setInterval(() => {
      setProgressPct((prev) => {
        if (prev >= 98) return prev;
        if (prev < 70) return Math.min(98, prev + 5);
        if (prev < 90) return Math.min(98, prev + 2);
        return Math.min(98, prev + 1);
      });
    }, 180);
    return () => clearInterval(timer);
  }, [isLoading]);

  useEffect(() => {
    if (hasBoth && status === "success") {
      comparePanelRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, [hasBoth, status]);

  useEffect(() => {
    fetchBackendStatus(modelName);
  }, [modelName]);

  useEffect(
    () => () => {
      if (sourceUrl) URL.revokeObjectURL(sourceUrl);
    },
    [sourceUrl]
  );

  async function handleFileChange(event) {
    const selected = event.target.files?.[0];
    if (!selected) return;
    setFile(selected);
    const nextUrl = URL.createObjectURL(selected);
    setSourceUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return nextUrl;
    });
    try {
      const ratio = await readImageAspectRatio(nextUrl);
      setFrameAspect(ratio);
    } catch (_err) {
      setFrameAspect("4 / 3");
    }
    setResultUrl("");
    setLatencyMs(null);
    setStatus("ready");
    setError("");
    setSlider(50);
    setProgressPct(0);
  }

  async function handleUpscale(event) {
    event.preventDefault();
    if (!file) {
      setError("Select an image first.");
      return;
    }

    setIsLoading(true);
    setError("");
    setStatus("processing");
    setProgressPct(0);

    try {
      const formData = new FormData();
      formData.append("image", file);
      let endpoint = "/upscale";
      if (modelName === "anime_gan_hayao") {
        endpoint = "/anime/hayao";
      } else {
        formData.append("model_name", modelName);
        formData.append("scale_factor", scaleFactor);
      }

      const response = await fetch(`${API_BASE}${endpoint}`, {
        method: "POST",
        body: formData
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Upscaling failed.");
      }

      setResultUrl(createDataUrl(data.image_data));
      setLatencyMs(data.inference_time_ms);
      setStatus(data.status || "success");
      setProgressPct(100);
    } catch (err) {
      setStatus("failure");
      setError(err.message || "Unknown error");
      setProgressPct(0);
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="page">
      <div className="glow glow-a" />
      <div className="glow glow-b" />
      <main className="layout">
        <header>
          <p className="kicker">RAInder / Super Resolution</p>
          <h1>Image Upscaling Console</h1>
          <p className="subtle">
            Upload a low-resolution image and compare before/after output with live latency stats.
          </p>
        </header>

        <section className="panel">
          <form onSubmit={handleUpscale} className="controls">
            <label>
              Source image
              <div className="file-picker">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/png,image/jpeg"
                  onChange={handleFileChange}
                  className="file-input-hidden"
                />
                <button
                  type="button"
                  className="file-button"
                  onClick={() => fileInputRef.current?.click()}
                >
                  Choose image
                </button>
                <span className="file-name">{file ? file.name : "No file selected"}</span>
              </div>
            </label>

            {isAnimeModel ? (
              <label>
                Operation
                <select value="anime_style_transfer" disabled>
                  <option value="anime_style_transfer">Style transfer (anime)</option>
                </select>
              </label>
            ) : (
              <label>
                Scale factor
                <select
                  value={scaleFactor}
                  onChange={(e) => {
                    const nextScale = e.target.value;
                    setScaleFactor(nextScale);
                    setModelName(nextScale === "2.0" ? "real_esrgan_x2" : "real_esrgan_x4");
                    setProgressPct(0);
                    setStatus("ready");
                  }}
                >
                  <option value="2.0">2x</option>
                  <option value="4.0">4x</option>
                </select>
              </label>
            )}

            <label>
              Model
              <select value={modelName} onChange={(e) => setModelName(e.target.value)}>
                <option value="real_esrgan_x2">Real-ESRGAN x2</option>
                <option value="real_esrgan_x4">Real-ESRGAN x4</option>
                <option value="anime_gan_hayao">AnimeGANv2 Hayao</option>
              </select>
            </label>

            <button disabled={isLoading} type="submit">
              {isLoading
                ? isAnimeModel
                  ? "Styling..."
                  : "Upscaling..."
                : isAnimeModel
                  ? "Run Style Transfer"
                  : "Run Upscale"}
            </button>
          </form>

          <div className="metrics">
            <div>
              <span>Progress</span>
              <div className="progress-shell" aria-label="Upscale progress">
                <div className="progress-liquid" style={{ width: `${progressPct}%` }}>
                  <div className="progress-wave" />
                </div>
              </div>
              <strong>{progressPct}%</strong>
            </div>
            <div>
              <span>Inference</span>
              <strong>{latencyMs === null ? "-" : `${latencyMs.toFixed(2)} ms`}</strong>
            </div>
            <div>
              <span>Backend</span>
              <div className="backend-row">
                <span className={`mode-badge ${backendReady ? "ready" : "not-ready"}`}>
                  {backendReady ? "ready" : "not ready"}
                </span>
                <strong>{backendMode}</strong>
              </div>
              <small className="backend-detail">{backendModel}</small>
            </div>
          </div>
          <p className="state-note">
            State: {status}
            {isFinalizing ? " (finalizing...)" : ""}
          </p>

          {error ? <p className="error">{error}</p> : null}
        </section>

        <section className="panel" ref={comparePanelRef}>
          {hasBoth ? (
            <>
              <div className="compare">
                <img src={resultUrl} alt="Upscaled output" className="compare-layer layer-output" />
                <img
                  src={sourceUrl}
                  alt="Original source"
                  className="compare-layer layer-input"
                  style={{ clipPath: `inset(0 0 0 ${slider}%)` }}
                />
                <div className="divider" style={{ left: `${slider}%` }} />
              </div>
              <label className="slider">
                Reveal upscaled output
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={slider}
                  onChange={(e) => setSlider(Number(e.target.value))}
                />
              </label>
              <p className="compare-note">Top: original input. Bottom: upscaled output.</p>
            </>
          ) : (
            <div className="empty" style={{ aspectRatio: frameAspect }}>
              Upload and run an image to see the comparison preview.
            </div>
          )}
        </section>
      </main>
    </div>
  );

  async function fetchBackendStatus(targetModel) {
    try {
      const url = new URL(`${API_BASE}/ready`);
      if (targetModel) {
        url.searchParams.set("model_name", targetModel);
      }
      const response = await fetch(url.toString());
      if (!response.ok) {
        setBackendReady(false);
        setBackendMode("unknown");
        setBackendModel(targetModel || "-");
        return;
      }
      const data = await response.json();
      setBackendReady(Boolean(data.model_ready && data.server_ready && data.server_live));
      setBackendMode(data.mode || "unknown");
      setBackendModel(data.model_name || targetModel || "-");
    } catch (_err) {
      setBackendReady(false);
      setBackendMode("offline");
      setBackendModel(targetModel || "-");
    }
  }
}

function readImageAspectRatio(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      if (!img.naturalWidth || !img.naturalHeight) {
        reject(new Error("Invalid image dimensions"));
        return;
      }
      resolve(`${img.naturalWidth} / ${img.naturalHeight}`);
    };
    img.onerror = () => reject(new Error("Failed to read image dimensions"));
    img.src = url;
  });
}
