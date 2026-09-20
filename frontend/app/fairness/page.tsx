"use client";

import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import AppNav from "@/components/AppNav";
import GlassPanel from "@/components/GlassPanel";

interface FairnessGroup {
  group: string;
  n: number;
  selection_rate: number;
  actual_default_rate: number;
  mean_predicted_risk: number;
  true_positive_rate: number | null;
}
interface FairnessReport {
  groups: FairnessGroup[];
  four_fifths_ratio: number | null;
  four_fifths_pass: boolean | null;
  methodology: string;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "https://major-project-explainable-ai-based.onrender.com";

export default function FairnessPage() {
  const [data, setData] = useState<FairnessReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const res = await fetch(`${API_URL}/fairness`);
        if (!res.ok) throw new Error(`Request failed (${res.status})`);
        const json = await res.json();
        if (!cancelled) setData(json);
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : "Something went wrong");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, []);

  const selectionData =
    data?.groups.map((g) => ({
      name: g.group,
      "Selection rate": Math.round(g.selection_rate * 1000) / 10,
    })) ?? [];

  const calibrationData =
    data?.groups.map((g) => ({
      name: g.group,
      "Actual default rate": Math.round(g.actual_default_rate * 1000) / 10,
      "Mean predicted risk": Math.round(g.mean_predicted_risk * 1000) / 10,
    })) ?? [];

  const passes = data?.four_fifths_pass;

  return (
    <>
      <AppNav />
      <main className="min-h-screen bg-[#050506] px-6 py-16 text-white md:px-12">
        <div className="mx-auto max-w-6xl">
          <header className="mb-10">
            <h1 className="text-3xl font-semibold tracking-tight md:text-4xl">Fairness Audit</h1>
            <p className="mt-2 max-w-2xl text-sm text-white/50">
              Age is both a model input and a class explicitly protected against
              discrimination in credit decisions under the US Equal Credit
              Opportunity Act — which makes it a meaningful, realistic dimension
              to audit rather than a synthetic stand-in.
            </p>
          </header>

          {loading && <p className="text-sm text-white/40">Loading fairness report…</p>}
          {error && (
            <p className="text-sm text-rose-300">
              {error} — is the backend running at {API_URL}?
            </p>
          )}

          {data && (
            <div className="space-y-6">
              <GlassPanel className="p-6">
                <div className="flex flex-wrap items-center justify-between gap-4">
                  <div>
                    <p className="text-xs text-white/40">Four-fifths ratio</p>
                    <p className="mt-1 text-4xl font-semibold tabular-nums">
                      {data.four_fifths_ratio?.toFixed(3) ?? "n/a"}
                    </p>
                  </div>
                  <span
                    className={`inline-flex rounded-full px-4 py-2 text-sm font-medium ring-1 ${
                      passes
                        ? "bg-emerald-400/10 text-emerald-300 ring-emerald-400/40"
                        : "bg-rose-400/10 text-rose-300 ring-rose-400/40"
                    }`}
                  >
                    {passes ? "Passes the four-fifths rule" : "Fails the four-fifths rule"}
                  </span>
                </div>
                <p className="mt-4 text-xs leading-relaxed text-white/40">{data.methodology}</p>
              </GlassPanel>

              <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                <GlassPanel className="p-6">
                  <h2 className="mb-1 text-sm font-medium text-white/70">
                    Selection rate by age group
                  </h2>
                  <p className="mb-5 text-xs text-white/40">
                    Share of each group the model would approve or conditionally approve.
                  </p>
                  <ResponsiveContainer width="100%" height={260}>
                    <BarChart data={selectionData} margin={{ left: 0, right: 8 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" vertical={false} />
                      <XAxis dataKey="name" stroke="rgba(255,255,255,0.3)" fontSize={11} />
                      <YAxis stroke="rgba(255,255,255,0.3)" fontSize={11} unit="%" />
                      <Tooltip
                        contentStyle={{ background: "#0a0a0c", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 12, fontSize: 12 }}
                        formatter={(value) => `${value}%`}
                      />
                      <Bar dataKey="Selection rate" radius={[6, 6, 0, 0]} fill="#38bdf8" />
                    </BarChart>
                  </ResponsiveContainer>
                </GlassPanel>

                <GlassPanel className="p-6">
                  <h2 className="mb-1 text-sm font-medium text-white/70">
                    Predicted vs. actual default rate
                  </h2>
                  <p className="mb-5 text-xs text-white/40">
                    If these two bars diverge for a group, the model's risk scores
                    mean something different for that group than for others.
                  </p>
                  <ResponsiveContainer width="100%" height={260}>
                    <BarChart data={calibrationData} margin={{ left: 0, right: 8 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" vertical={false} />
                      <XAxis dataKey="name" stroke="rgba(255,255,255,0.3)" fontSize={11} />
                      <YAxis stroke="rgba(255,255,255,0.3)" fontSize={11} unit="%" />
                      <Tooltip
                        contentStyle={{ background: "#0a0a0c", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 12, fontSize: 12 }}
                        formatter={(value) => `${value}%`}
                      />
                      <Legend wrapperStyle={{ fontSize: 11, color: "rgba(255,255,255,0.5)" }} />
                      <Bar dataKey="Mean predicted risk" radius={[6, 6, 0, 0]} fill="#a78bfa" />
                      <Bar dataKey="Actual default rate" radius={[6, 6, 0, 0]} fill="#fb7185" />
                    </BarChart>
                  </ResponsiveContainer>
                </GlassPanel>
              </div>

              <GlassPanel className="overflow-x-auto p-6">
                <h2 className="mb-4 text-sm font-medium text-white/70">Per-group detail</h2>
                <table className="w-full min-w-[560px] text-left text-sm">
                  <thead>
                    <tr className="border-b border-white/10 text-xs text-white/40">
                      <th className="pb-2 pr-4 font-normal">Group</th>
                      <th className="pb-2 pr-4 font-normal">n</th>
                      <th className="pb-2 pr-4 font-normal">Selection rate</th>
                      <th className="pb-2 pr-4 font-normal">Actual default rate</th>
                      <th className="pb-2 pr-4 font-normal">True positive rate</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.groups.map((g) => (
                      <tr key={g.group} className="border-b border-white/5">
                        <td className="py-2 pr-4">{g.group}</td>
                        <td className="py-2 pr-4 text-white/60">{g.n}</td>
                        <td className="py-2 pr-4 text-white/60">
                          {(g.selection_rate * 100).toFixed(1)}%
                        </td>
                        <td className="py-2 pr-4 text-white/60">
                          {(g.actual_default_rate * 100).toFixed(1)}%
                        </td>
                        <td className="py-2 pr-4 text-white/60">
                          {g.true_positive_rate != null
                            ? `${(g.true_positive_rate * 100).toFixed(1)}%`
                            : "n/a"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </GlassPanel>
            </div>
          )}
        </div>
      </main>
    </>
  );
}
