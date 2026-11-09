import { useState, useRef } from "react";

const WAVEFORM_HEIGHTS = [30, 55, 40, 70, 45, 60, 35, 65, 50, 75, 40, 55, 30, 60, 45, 70, 35, 50];

export default function App() {
  const [text, setText] = useState("");
  const [file, setFile] = useState(null);
  const [imageUrl, setImageUrl] = useState("");
  const [previewSrc, setPreviewSrc] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const f = e.target.files[0];
    if (f) {
      setFile(f);
      setImageUrl("");
      setPreviewSrc(URL.createObjectURL(f));
    }
  };

  const handleUrlChange = (e) => {
    const val = e.target.value;
    setImageUrl(val);
    if (val) {
      setFile(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
      setPreviewSrc(val);
    } else {
      setPreviewSrc(null);
    }
  };

  const clearImage = () => {
    setFile(null);
    setImageUrl("");
    setPreviewSrc(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleSubmit = async () => {
    // Biztonsági ellenőrzés: ha semmi sincs megadva
    if (!text.trim() && !file && !imageUrl.trim()) {
      setError("Kérlek adj meg legalább egy szöveget vagy tölts fel egy képet!");
      return;
    }

    setLoading(true);
    setResult(null);
    setError(null);

    const formData = new FormData();
    if (text.trim()) formData.append("text", text.trim());
    if (file) formData.append("file", file);
    if (imageUrl.trim()) formData.append("image_url", imageUrl.trim());

    try {
      const res = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });
      if (!res.ok) throw new Error(`Szerver hiba (${res.status})`);
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setResult(data);
    } catch (err) {
      setError(
        err.message.includes("Szerver hiba") || err.message !== ""
          ? err.message
          : "Nem sikerült kapcsolódni az elemző szerverhez. Ellenőrizd, hogy fut-e a backend a 8000-es porton."
      );
    }
    setLoading(false);
  };

  const isReal = result && result.real_prob >= result.fake_prob;

  return (
    <div className="min-h-screen bg-ink text-[#E7EDF3] font-body">
      <div className="max-w-3xl mx-auto px-6 py-16">
        {/* --- Header / hero --- */}
        <div className="mb-10">
          <div className="font-mono text-xs tracking-[0.2em] text-muted uppercase mb-3">
            Szöveg- és képalapú hitelesség-elemzés
          </div>
          <h1 className="font-display text-4xl md:text-5xl font-semibold tracking-tight text-white">
            Jel, vagy zaj?
          </h1>
          <p className="mt-3 text-muted max-w-lg leading-relaxed">
            Illeszd be a hír szövegét, vagy csatolj hozzá képet (akár külön-külön is). A
            rendszer BERT és Vision Transformer alapú elemzést futtat, valószínűségi
            ítéletet ad a hitelességről, és megmutatja, melyik bemenet mennyire befolyásolta.
          </p>

          {/* waveform motif */}
          <div className="flex items-end gap-[3px] h-8 mt-6 opacity-70">
            {WAVEFORM_HEIGHTS.map((h, i) => (
              <div
                key={i}
                className={`w-[3px] bg-real/60 rounded-full ${loading ? "bar-pulse" : ""}`}
                style={{
                  height: loading ? "100%" : `${h}%`,
                  animationDelay: `${i * 0.05}s`,
                }}
              />
            ))}
          </div>
        </div>

        {/* --- Input console --- */}
        <div className="grid md:grid-cols-3 gap-4">
          <div className="md:col-span-2 relative">
            <div className="font-mono text-[11px] tracking-[0.15em] text-muted uppercase mb-2">
              Bemenet // Szöveg (opcionális)
            </div>
            <div className={`relative rounded-lg overflow-hidden ${loading ? "scanline" : ""}`}>
              <textarea
                className="w-full bg-panel border border-panelBorder rounded-lg p-4 text-[15px] leading-relaxed placeholder:text-muted/60 focus:outline-none focus:ring-2 focus:ring-real/50 resize-none"
                rows={6}
                placeholder="pl. scientists discover giant stone head shaped rock in field"
                value={text}
                onChange={(e) => setText(e.target.value)}
                disabled={loading}
              />
            </div>
          </div>

          <div>
            <div className="font-mono text-[11px] tracking-[0.15em] text-muted uppercase mb-2">
              Bemenet // Kép (opcionális)
            </div>
            <div className="bg-panel border border-panelBorder rounded-lg p-4 h-[calc(100%-1.75rem)] flex flex-col gap-3">
              {previewSrc ? (
                <div className="relative">
                  <img
                    src={previewSrc}
                    alt="Előnézet"
                    className="w-full h-24 object-cover rounded-md border border-panelBorder"
                  />
                  <button
                    onClick={clearImage}
                    className="absolute -top-2 -right-2 w-6 h-6 rounded-full bg-ink border border-panelBorder text-muted hover:text-fake text-xs flex items-center justify-center"
                    aria-label="Kép eltávolítása"
                  >
                    ✕
                  </button>
                </div>
              ) : (
                <label className="flex-1 flex flex-col items-center justify-center border border-dashed border-panelBorder rounded-md text-muted text-xs cursor-pointer hover:border-real/50 hover:text-real/80 transition-colors py-4">
                  <span>Fájl kiválasztása</span>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    onChange={handleFileChange}
                    className="hidden"
                    disabled={loading}
                  />
                </label>
              )}

              <div className="flex items-center gap-2 text-[10px] font-mono text-muted uppercase">
                <div className="flex-1 h-px bg-panelBorder" />
                vagy
                <div className="flex-1 h-px bg-panelBorder" />
              </div>

              <input
                className="w-full bg-ink border border-panelBorder rounded-md px-3 py-2 text-xs placeholder:text-muted/60 focus:outline-none focus:ring-2 focus:ring-real/50"
                placeholder="kép URL"
                value={imageUrl}
                onChange={handleUrlChange}
                disabled={loading}
              />
            </div>
          </div>
        </div>

        {/* --- Submit --- */}
        <button
          onClick={handleSubmit}
          disabled={loading || (!text.trim() && !file && !imageUrl.trim())}
          className="mt-5 w-full md:w-auto px-6 py-3 rounded-lg bg-real text-ink font-display font-semibold text-sm tracking-wide hover:brightness-110 active:brightness-95 disabled:opacity-30 disabled:cursor-not-allowed transition-all"
        >
          {loading ? "Elemzés…" : "Elemzés indítása"}
        </button>

        {/* --- Error state --- */}
        {error && (
          <div className="mt-6 settle rounded-lg border border-fake/30 bg-fake/10 px-4 py-3 text-sm text-fake">
            {error}
          </div>
        )}

        {/* --- Verdict panel --- */}
        <div className="mt-8">
          <div className="font-mono text-[11px] tracking-[0.15em] text-muted uppercase mb-2">
            Eredmény
          </div>

          {!result && !loading && !error && (
            <div className="border border-dashed border-panelBorder rounded-lg px-6 py-10 text-center text-muted text-sm">
              A jel elemzésére vár…
            </div>
          )}

          {result && (
            <div className="settle bg-panel border border-panelBorder rounded-lg p-6">
              <div className="flex items-baseline justify-between flex-wrap gap-2">
                <div
                  className={`font-display text-2xl font-semibold ${
                    isReal ? "text-real" : "text-fake"
                  }`}
                >
                  {isReal ? "VALÓS HÍR" : "ÁLHÍR"}
                </div>
                <div className="font-mono text-sm text-muted">
                  {isReal ? result.real_prob : result.fake_prob}% biztos
                </div>
              </div>

              {/* confidence rail */}
              <div className="mt-5 relative h-2 rounded-full bg-gradient-to-r from-fake/40 via-panelBorder to-real/40">
                <div
                  className="absolute top-1/2 -translate-y-1/2 w-3 h-3 rounded-full bg-white shadow-[0_0_0_3px_rgba(255,255,255,0.15)] transition-all duration-700 ease-out"
                  style={{ left: `calc(${result.real_prob}% - 6px)` }}
                />
              </div>
              <div className="flex justify-between mt-2 font-mono text-[11px] text-muted">
                <span>ÁLHÍR {result.fake_prob}%</span>
                <span>VALÓS {result.real_prob}%</span>
              </div>

              {/* Generált képaláírás (BLIP) */}
              {result.caption && (
                <div className="mt-6">
                  <div className="font-mono text-[11px] tracking-[0.15em] text-muted uppercase mb-1">
                    A modell szerint a képen
                  </div>
                  <div className="text-sm italic text-[#E7EDF3]/90">„{result.caption}”</div>
                </div>
              )}

              {/* Modalitás-hozzájárulás (Shapley-értékek) */}
              {result.contributions && result.contributions.length > 0 && (
                <div className="mt-6">
                  <div className="font-mono text-[11px] tracking-[0.15em] text-muted uppercase mb-2">
                    Mi befolyásolta a döntést
                  </div>
                  <div className="flex h-3 rounded-full overflow-hidden border border-panelBorder">
                    {result.contributions.map((c) => (
                      <div
                        key={c.modality}
                        className={`${c.direction === "fake" ? "bg-fake/70" : "bg-real/70"} border-r border-ink last:border-r-0 transition-all duration-700`}
                        style={{ width: `${c.share}%` }}
                        title={`${c.modality}: ${c.share}%`}
                      />
                    ))}
                  </div>
                  <div className="mt-3 space-y-1">
                    {result.contributions.map((c) => (
                      <div key={c.modality} className="flex justify-between text-xs font-mono">
                        <span className="text-[#E7EDF3]">{c.modality}</span>
                        <span className={c.direction === "fake" ? "text-fake" : "text-real"}>
                          {c.share}% · {c.direction === "fake" ? "álhír felé" : "valós felé"} húz
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Dynamic mode info */}
              <div className="mt-4 text-xs text-muted/80 font-mono">
                {result.mode === "text_only"
                  ? "* csak szöveges elemzés"
                  : result.mode === "image_only"
                  ? "* csak képi elemzés"
                  : "* multimodális elemzés (szöveg + kép)"}
                {result.engine === "legacy" && " · régi modell (a fúziós fej még nincs betanítva)"}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}