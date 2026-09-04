"use client";

import { ChangeEvent, useState } from "react";

interface Prediction {
  breed: string;
  confidence: number;
}

interface Result {
  prediction: string;
  confidence: number;
  top_predictions: Prediction[];
}

export default function Home() {
  const [image, setImage] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<Result | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  function handleImageChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];

    if (!file) return;

    if (!file.type.startsWith("image/")) {
      setError("Please select an image file.");
      return;
    }

    setImage(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
    setError("");
  }

  async function analyzeImage() {
    if (!image) {
      setError("Please upload an image first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    const formData = new FormData();
    formData.append("file", image);

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || data.error || "Prediction failed."
        );
      }

      setResult(data);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Something went wrong while analyzing the image."
      );
    } finally {
      setLoading(false);
    }
  }

  function reset() {
    setImage(null);
    setPreview(null);
    setResult(null);
    setError("");
  }

  return (
    <main className="min-h-screen bg-[#f6f7f2] text-gray-900">

      {/* Header */}
      <header className="border-b border-gray-200 bg-white">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-5">

          <div>
            <h1 className="text-xl font-bold tracking-tight">
              LivestockIQ
            </h1>

            <p className="text-xs text-gray-500">
              Cattle & Buffalo Breed Identification
            </p>
          </div>

          <div className="rounded-full bg-gray-100 px-4 py-2 text-sm text-gray-600">
            AI Powered
          </div>

        </div>
      </header>

      {/* Main */}
      <section className="mx-auto max-w-5xl px-6 py-16">

        <div className="text-center">

          <p className="text-sm font-semibold uppercase tracking-widest text-gray-500">
            Indian Livestock AI
          </p>

          <h2 className="mt-4 text-4xl font-bold tracking-tight sm:text-6xl">
            Identify the breed.
            <br />
            Understand your livestock.
          </h2>

          <p className="mx-auto mt-6 max-w-2xl text-lg text-gray-600">
            Upload a clear image of a cattle or buffalo and let our
            AI model identify its most likely breed.
          </p>

        </div>

        {/* Upload card */}
        <div className="mx-auto mt-12 max-w-3xl">

          <div className="rounded-3xl border border-gray-200 bg-white p-6 shadow-sm sm:p-8">

            <label className="group flex min-h-[360px] cursor-pointer items-center justify-center rounded-2xl border-2 border-dashed border-gray-300 bg-gray-50 p-6 transition hover:border-gray-500 hover:bg-gray-100">

              {preview ? (
                <div className="flex w-full flex-col items-center">

                  <img
                    src={preview}
                    alt="Uploaded livestock"
                    className="max-h-[300px] max-w-full rounded-xl object-contain"
                  />

                  <p className="mt-4 text-sm text-gray-500">
                    Click to select another image
                  </p>

                </div>
              ) : (
                <div className="text-center">

                  <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-gray-200 text-2xl">
                    +
                  </div>

                  <h3 className="mt-5 text-lg font-semibold">
                    Upload livestock image
                  </h3>

                  <p className="mt-2 text-sm text-gray-500">
                    Drag and drop or click to browse
                  </p>

                  <p className="mt-1 text-xs text-gray-400">
                    JPG, PNG or WEBP
                  </p>

                </div>
              )}

              <input
                type="file"
                accept="image/jpeg,image/png,image/webp"
                onChange={handleImageChange}
                className="hidden"
              />

            </label>

            {/* Error */}
            {error && (
              <div className="mt-5 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
                {error}
              </div>
            )}

            {/* Buttons */}
            <div className="mt-6 flex gap-3">

              <button
                onClick={analyzeImage}
                disabled={!image || loading}
                className="flex-1 rounded-xl bg-gray-900 px-6 py-4 font-semibold text-white transition hover:bg-gray-800 disabled:cursor-not-allowed disabled:opacity-40"
              >
                {loading ? "Analyzing image..." : "Identify Breed"}
              </button>

              {image && (
                <button
                  onClick={reset}
                  disabled={loading}
                  className="rounded-xl border border-gray-300 px-6 py-4 font-semibold transition hover:bg-gray-100 disabled:opacity-40"
                >
                  Reset
                </button>
              )}

            </div>

          </div>
        </div>

        {/* Results */}
        {result && (
          <div className="mx-auto mt-10 max-w-3xl">

            <div className="rounded-3xl border border-gray-200 bg-white p-8 shadow-sm">

              <div className="text-center">

                <p className="text-sm font-medium uppercase tracking-widest text-gray-500">
                  Most likely breed
                </p>

                <h3 className="mt-3 text-4xl font-bold">
                  {result.prediction}
                </h3>

                <div className="mt-4 inline-flex rounded-full bg-gray-100 px-5 py-2 text-sm font-medium">
                  {result.confidence.toFixed(2)}% confidence
                </div>

              </div>

              <div className="mt-10">

                <h4 className="text-lg font-semibold">
                  Top predictions
                </h4>

                <div className="mt-5 space-y-5">

                  {result.top_predictions.map(
                    (prediction, index) => (
                      <div key={index}>

                        <div className="mb-2 flex justify-between text-sm">

                          <span className="font-medium">
                            {prediction.breed}
                          </span>

                          <span className="text-gray-500">
                            {prediction.confidence.toFixed(2)}%
                          </span>

                        </div>

                        <div className="h-3 overflow-hidden rounded-full bg-gray-100">

                          <div
                            className="h-full rounded-full bg-gray-900 transition-all duration-700"
                            style={{
                              width: `${Math.min(
                                prediction.confidence,
                                100
                              )}%`,
                            }}
                          />

                        </div>

                      </div>
                    )
                  )}

                </div>

              </div>

            </div>

          </div>
        )}

      </section>

      {/* Footer */}
      <footer className="border-t border-gray-200 bg-white">

        <div className="mx-auto max-w-7xl px-6 py-8 text-center text-sm text-gray-500">

          LivestockIQ · AI-powered Indian cattle & buffalo breed identification

        </div>

      </footer>

    </main>
  );
} 