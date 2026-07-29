import { useState } from "react";

export default function App() {
  const [text, setText] = useState("");
  const [file, setFile] = useState(null);
  const [imageUrl, setImageUrl] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    setLoading(true);
    const formData = new FormData();
    formData.append("text", text);
    if (file) formData.append("file", file);
    if (imageUrl) formData.append("image_url", imageUrl);

    const res = await fetch("http://localhost:8000/predict", {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
    setResult(data);
    setLoading(false);
  };

  return (
    <div className="max-w-xl mx-auto p-6 space-y-4">
      <h1 className="text-2xl font-bold">Álhír-detektor</h1>
      <textarea
        className="w-full border rounded p-2"
        rows={4}
        placeholder="Írd be a hír szövegét..."
        value={text}
        onChange={(e) => setText(e.target.value)}
      />
      <input
        type="file"
        accept="image/*"
        onChange={(e) => setFile(e.target.files[0])}
      />
      <input
        className="w-full border rounded p-2"
        placeholder="...vagy kép URL"
        value={imageUrl}
        onChange={(e) => setImageUrl(e.target.value)}
      />
      <button
        onClick={handleSubmit}
        disabled={loading || !text}
        className="bg-blue-600 text-white px-4 py-2 rounded disabled:opacity-50"
      >
        {loading ? "Elemzés..." : "Ellenőrzés"}
      </button>

      {result && (
        <div className="border rounded p-4 space-y-2">
          <div className="text-lg font-semibold">
            {result.fake_prob > result.real_prob ? "🚨 ÁLHÍR" : "✅ VALÓS HÍR"}
          </div>
          <div>Valós: {result.real_prob.toFixed(1)}% | Álhír: {result.fake_prob.toFixed(1)}%</div>

          <div className="mt-3">
            <div className="text-sm font-medium mb-1">Modalitás-hozzájárulás (SHAP)</div>
            <div className="flex h-6 rounded overflow-hidden">
              <div
                className="bg-indigo-500 flex items-center justify-center text-white text-xs"
                style={{ width: `${result.modality_contribution.text_pct}%` }}
              >
                Szöveg {result.modality_contribution.text_pct}%
              </div>
              <div
                className="bg-amber-500 flex items-center justify-center text-white text-xs"
                style={{ width: `${result.modality_contribution.image_pct}%` }}
              >
                Kép {result.modality_contribution.image_pct}%
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}