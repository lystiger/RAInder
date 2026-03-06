import { useMemo, useState } from "react";

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
  const [modelName, setModelName] = useState("super_resolution_model");
  const [slider, setSlider] = useState(50);
  const [latencyMs, setLatencyMs] = useState(null);
  const [status, setStatus] = useState("idle");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  const hasBoth = useMemo(() => Boolean(sourceUrl && resultUrl), [sourceUrl, resultUrl]);

  function handleFileChange(event) {
    const selected = event.target.files?.[0];
    if (!selected) return;
    setFile(selected);
    setSourceUrl(URL.createObjectURL(selected));
    setResultUrl("");
    setLatencyMs(null);
    setStatus("ready");
    setError("");
    setSlider(50);
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

    try {
      const formData = new FormData();
      formData.append("image", file);
      formData.append("model_name", modelName);
      formData.append("scale_factor", scaleFactor);

      const response = await fetch(`${API_BASE}/upscale`, {
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
    } catch (err) {
      setStatus("failure");
      setError(err.message || "Unknown error");
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
              <input type="file" accept="image/png,image/jpeg" onChange={handleFileChange} />
            </label>

            <label>
              Scale factor
              <select value={scaleFactor} onChange={(e) => setScaleFactor(e.target.value)}>
                <option value="2.0">2x</option>
                <option value="4.0">4x</option>
              </select>
            </label>

            <label>
              Model
              <input value={modelName} onChange={(e) => setModelName(e.target.value)} />
            </label>

            <button disabled={isLoading} type="submit">
              {isLoading ? "Upscaling..." : "Run Upscale"}
            </button>
          </form>

          <div className="metrics">
            <div>
              <span>Status</span>
              <strong>{status}</strong>
            </div>
            <div>
              <span>Inference</span>
              <strong>{latencyMs === null ? "-" : `${latencyMs.toFixed(2)} ms`}</strong>
            </div>
            <div>
              <span>API</span>
              <strong>{API_BASE}</strong>
            </div>
          </div>

          {error ? <p className="error">{error}</p> : null}
        </section>

        <section className="panel">
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
            <div className="empty">Upload and run an image to see the comparison preview.</div>
          )}
        </section>
      </main>
    </div>
  );
}
