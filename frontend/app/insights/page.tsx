"use client";

import { useEffect, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  Bar,
  BarChart,
} from "recharts";
import AppNav from "@/components/AppNav";
import GlassPanel from "@/components/GlassPanel";

interface ROCPoint {
  fpr: number;
  tpr: number;
}
interface CalibrationPoint {
  mean_predicted: number;
  observed_frequency: number;
}
interface ConfusionMatrix {
  true_negative: number;
  false_positive: number;
  false_negative: number;
  true_positive: number;
}
interface GlobalFeatureImportance {
  feature: string;
  mean_abs_shap: number;
}
interface ModelInsights {
  roc_auc: number;
  accuracy: number;
  n_test_samples: number;
  trained_at: string;
  roc_curve: ROCPoint[];
  calibration_curve: CalibrationPoint[];
  confusion_matrix: ConfusionMatrix;
  global_feature_importance: GlobalFeatureImportance[];
}

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "https://major-project-explainable-ai-based.onrender.com";

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-xs text-white/40">{label}</p>
      <p className="mt-1 text-3xl font-semibold tabular-nums">{value}</p>
    </div>
  );
}

export default function InsightsPage() {
  const [data, setData] = useState<ModelInsights | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const res = await fetch(`${API_URL}/insights`);
        if (!res.ok) throw new Error(`Request failed (${res.status})`);
        const json = await res.json();
        if (!cancelled) setData(json);
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Something went wrong");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, []);

  const rocData = data?.roc_curve.map((p) => ({ ...p, reference: p.fpr })) ?? [];
  const calData =
    data?.calibration_curve.map((p) => ({ ...p, reference: p.mean_predicted })) ?? [];
  const importanceData =
    data?.global_feature_importance
      .slice()
      .sort((a, b) => b.mean_abs_shap - a.mean_abs_shap)
      .slice(0, 10)
      .map((f) => ({ name: f.feature.replace(/_/g, " "), importance: f.mean_abs_shap })) ?? [];

  const cm = data?.confusion_matrix;

  return (
    <>
      <AppNav />
      <main className="min-h-screen bg-[#050506] px-6 py-16 text-white md:px-12">
        <div className="mx-auto max-w-6xl">
          <header className="mb-10">
            <h1 className="text-3xl font-semibold tracking-tight md:text-4xl">Model Insights</h1>
            <p className="mt-2 max-w-2xl text-sm text-white/50">
              Properties of the model as a whole rather than any single applicant —
              how well it separates good and bad risk, whether its probabilities
              mean what they say, and which features drive it overall.
            </p>
          </header>

          {loading && <p className="text-sm text-white/40">Loading model insights…</p>}
          {error && (
            <p className="text-sm text-rose-300">
              {error} — is the backend running at {API_URL}?
            </p>
          )}

          {data && (
            <div className="space-y-6">
              <GlassPanel className="grid grid-cols-2 gap-6 p-6 sm:grid-cols-4">
                <StatCard label="ROC-AUC" value={data.roc_auc.toFixed(3)} />
                <StatCard label="Accuracy" value={`${(data.accuracy * 100).toFixed(1)}%`} />
                <StatCard label="Test set size" value={data.n_test_samples.toLocaleString()} />
                <StatCard
                  label="Trained"
                  value={new Date(data.trained_at).toLocaleDateString()}
                />
              </GlassPanel>

              <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                <GlassPanel className="p-6">
                  <h2 className="mb-1 text-sm font-medium text-white/70">ROC curve</h2>
                  <p className="mb-5 text-xs text-white/40">
                    How well the model separates defaulters from non-defaulters across
                    every possible threshold. The dashed diagonal is a coin flip; the
                    further the solid line bows toward the top-left, the better.
                  </p>
                  <ResponsiveContainer width="100%" height={260}>
                    <LineChart data={rocData} margin={{ left: 8, right: 8 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                      <XAxis
                        dataKey="fpr"
                        type="number"
                        domain={[0, 1]}
                        stroke="rgba(255,255,255,0.3)"
                        fontSize={11}
                        label={{ value: "False positive rate", position: "insideBottom", offset: -5, fill: "rgba(255,255,255,0.3)", fontSize: 11 }}
                      />
                      <YAxis
                        domain={[0, 1]}
                        stroke="rgba(255,255,255,0.3)"
                        fontSize={11}
                        label={{ value: "True positive rate", angle: -90, position: "insideLeft", fill: "rgba(255,255,255,0.3)", fontSize: 11 }}
                      />
                      <Tooltip
                        contentStyle={{ background: "#0a0a0c", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 12, fontSize: 12 }}
                        formatter={(value) => Number(value).toFixed(3)}
                      />
                      <Line type="monotone" dataKey="tpr" stroke="#38bdf8" strokeWidth={2} dot={false} name="Model" />
                      <Line type="linear" dataKey="reference" stroke="rgba(255,255,255,0.25)" strokeDasharray="4 4" dot={false} name="Random" />
                    </LineChart>
                  </ResponsiveContainer>
                </GlassPanel>

                <GlassPanel className="p-6">
                  <h2 className="mb-1 text-sm font-medium text-white/70">Calibration</h2>
                  <p className="mb-5 text-xs text-white/40">
                    When the model says "30% risk," do roughly 30% of those applicants
                    actually default? The closer the solid line hugs the dashed
                    diagonal, the more trustworthy the raw probabilities are.
                  </p>
                  <ResponsiveContainer width="100%" height={260}>
                    <LineChart data={calData} margin={{ left: 8, right: 8 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                      <XAxis
                        dataKey="mean_predicted"
                        type="number"
                        domain={[0, 1]}
                        stroke="rgba(255,255,255,0.3)"
                        fontSize={11}
                        label={{ value: "Mean predicted risk", position: "insideBottom", offset: -5, fill: "rgba(255,255,255,0.3)", fontSize: 11 }}
                      />
                      <YAxis
                        domain={[0, 1]}
                        stroke="rgba(255,255,255,0.3)"
                        fontSize={11}
                        label={{ value: "Observed default rate", angle: -90, position: "insideLeft", fill: "rgba(255,255,255,0.3)", fontSize: 11 }}
                      />
                      <Tooltip
                        contentStyle={{ background: "#0a0a0c", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 12, fontSize: 12 }}
                        formatter={(value) => Number(value).toFixed(3)}
                      />
                      <Line type="monotone" dataKey="observed_frequency" stroke="#fb7185" strokeWidth={2} dot={{ r: 3 }} name="Observed" />
                      <Line type="linear" dataKey="reference" stroke="rgba(255,255,255,0.25)" strokeDasharray="4 4" dot={false} name="Perfectly calibrated" />
                    </LineChart>
                  </ResponsiveContainer>
                </GlassPanel>
              </div>

              <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                <GlassPanel className="p-6">
                  <h2 className="mb-1 text-sm font-medium text-white/70">Global feature importance</h2>
                  <p className="mb-5 text-xs text-white/40">
                    Mean absolute SHAP value across the test set — which features move
                    predictions the most on average, not just for one applicant.
                  </p>
                  <ResponsiveContainer width="100%" height={Math.max(importanceData.length * 32, 200)}>
                    <BarChart data={importanceData} layout="vertical" margin={{ left: 24 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" horizontal={false} />
                      <XAxis type="number" stroke="rgba(255,255,255,0.3)" fontSize={11} />
                      <YAxis type="category" dataKey="name" width={150} stroke="rgba(255,255,255,0.3)" fontSize={11} />
                      <Tooltip
                        contentStyle={{ background: "#0a0a0c", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 12, fontSize: 12 }}
                        formatter={(value) => Number(value).toFixed(4)}
                      />
                      <Bar dataKey="importance" radius={[0, 6, 6, 0]} fill="#a78bfa" />
                    </BarChart>
                  </ResponsiveContainer>
                </GlassPanel>

                <GlassPanel className="p-6">
                  <h2 className="mb-1 text-sm font-medium text-white/70">Confusion matrix</h2>
                  <p className="mb-5 text-xs text-white/40">
                    At the default 50% threshold, on the held-out test set.
                  </p>
                  {cm && (
                    <div className="grid grid-cols-2 gap-3">
                      <div className="rounded-2xl border border-emerald-400/20 bg-emerald-400/5 p-4">
                        <p className="text-xs text-white/40">True negative</p>
                        <p className="mt-1 text-2xl font-semibold">{cm.true_negative}</p>
                        <p className="mt-1 text-xs text-white/30">Correctly predicted low risk</p>
                      </div>
                      <div className="rounded-2xl border border-amber-400/20 bg-amber-400/5 p-4">
                        <p className="text-xs text-white/40">False positive</p>
                        <p className="mt-1 text-2xl font-semibold">{cm.false_positive}</p>
                        <p className="mt-1 text-xs text-white/30">Flagged risky, actually repaid</p>
                      </div>
                      <div className="rounded-2xl border border-rose-400/20 bg-rose-400/5 p-4">
                        <p className="text-xs text-white/40">False negative</p>
                        <p className="mt-1 text-2xl font-semibold">{cm.false_negative}</p>
                        <p className="mt-1 text-xs text-white/30">Missed — approved but defaulted</p>
                      </div>
                      <div className="rounded-2xl border border-sky-400/20 bg-sky-400/5 p-4">
                        <p className="text-xs text-white/40">True positive</p>
                        <p className="mt-1 text-2xl font-semibold">{cm.true_positive}</p>
                        <p className="mt-1 text-xs text-white/30">Correctly predicted default</p>
                      </div>
                    </div>
                  )}
                </GlassPanel>
              </div>
            </div>
          )}
        </div>
      </main>
    </>
  );
}
