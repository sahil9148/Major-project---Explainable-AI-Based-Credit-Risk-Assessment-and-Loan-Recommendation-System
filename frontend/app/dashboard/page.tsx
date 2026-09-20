"use client";

import { useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import AppNav from "@/components/AppNav";
import GlassPanel from "@/components/GlassPanel";

// Mirrors backend/schemas.py -- kept in sync by hand since this is a
// two-service project. For a larger system, generate this from the
// FastAPI OpenAPI schema instead of hand-maintaining it.
interface ApplicantForm {
  age: number;
  annual_income: number;
  loan_amount: number;
  credit_score: number;
  employment_length_years: number;
  debt_to_income_ratio: number;
  num_open_credit_lines: number;
  num_credit_inquiries_last_6m: number;
  credit_utilization_rate: number;
  past_defaults: number;
  home_ownership: "RENT" | "MORTGAGE" | "OWN";
  loan_purpose:
    | "debt_consolidation"
    | "home_improvement"
    | "education"
    | "business"
    | "medical"
    | "other";
}

interface FeatureContribution {
  feature: string;
  value: number | string;
  shap_value: number;
  direction: "increases_risk" | "decreases_risk";
}

interface CounterfactualSuggestion {
  feature: string;
  description: string;
  new_risk_score: number;
  new_tier: "LOW" | "MODERATE" | "HIGH" | "VERY_HIGH";
}

interface RiskAssessment {
  risk_score: number;
  risk_tier: "LOW" | "MODERATE" | "HIGH" | "VERY_HIGH";
  recommendation: "APPROVE" | "APPROVE_WITH_CONDITIONS" | "MANUAL_REVIEW" | "DENY";
  recommended_interest_rate_pct: number | null;
  recommended_loan_amount: number | null;
  base_value: number;
  top_contributions: FeatureContribution[];
  narrative: string;
  counterfactuals: CounterfactualSuggestion[];
  model_version: string;
}

const DEFAULT_FORM: ApplicantForm = {
  age: 32,
  annual_income: 68000,
  loan_amount: 15000,
  credit_score: 690,
  employment_length_years: 5,
  debt_to_income_ratio: 0.31,
  num_open_credit_lines: 4,
  num_credit_inquiries_last_6m: 1,
  credit_utilization_rate: 0.35,
  past_defaults: 0,
  home_ownership: "MORTGAGE",
  loan_purpose: "debt_consolidation",
};

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "https://major-project-explainable-ai-based.onrender.com";

const TIER_STYLES: Record<string, { text: string; ring: string; bg: string }> = {
  LOW: { text: "text-emerald-300", ring: "ring-emerald-400/40", bg: "bg-emerald-400/10" },
  MODERATE: { text: "text-sky-300", ring: "ring-sky-400/40", bg: "bg-sky-400/10" },
  HIGH: { text: "text-amber-300", ring: "ring-amber-400/40", bg: "bg-amber-400/10" },
  VERY_HIGH: { text: "text-rose-300", ring: "ring-rose-400/40", bg: "bg-rose-400/10" },
};

function NumberField({
  label,
  value,
  onChange,
  min,
  max,
  step = 1,
  suffix,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step?: number;
  suffix?: string;
}) {
  return (
    <label className="block">
      <div className="mb-1 flex items-center justify-between text-xs text-white/50">
        <span>{label}</span>
        <span className="tabular-nums text-white/70">
          {value}
          {suffix}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-sky-400"
      />
    </label>
  );
}

export default function DashboardPage() {
  const [form, setForm] = useState<ApplicantForm>(DEFAULT_FORM);
  const [result, setResult] = useState<RiskAssessment | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const update = <K extends keyof ApplicantForm>(key: K, value: ApplicantForm[K]) => {
    setForm((prev) => ({ ...prev, [key]: value }));
  };

  async function runAssessment() {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail ? JSON.stringify(body.detail) : `Request failed (${res.status})`);
      }
      const data: RiskAssessment = await res.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  const chartData =
    result?.top_contributions.map((c) => ({
      name: c.feature.replace(/_/g, " "),
      shap: c.shap_value,
    })) ?? [];

  const tierStyle = result ? TIER_STYLES[result.risk_tier] : TIER_STYLES.MODERATE;

  return (
    <>
      <AppNav />
      <main className="min-h-screen bg-[#050506] px-6 py-16 text-white md:px-12">
      <div className="mx-auto max-w-6xl">
        <header className="mb-10">
          <h1 className="text-3xl font-semibold tracking-tight md:text-4xl">RiskLens</h1>
          <p className="mt-2 max-w-xl text-sm text-white/50">
            Adjust the applicant profile and run the model to see the predicted
            default risk, the resulting loan recommendation, and exactly which
            factors the model weighed to get there.
          </p>
        </header>

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-5">
          {/* Input form */}
          <GlassPanel className="p-6 lg:col-span-2">
            <h2 className="mb-5 text-sm font-medium text-white/70">Applicant profile</h2>
            <div className="space-y-5">
              <NumberField label="Age" value={form.age} onChange={(v) => update("age", v)} min={18} max={80} />
              <NumberField label="Annual income" value={form.annual_income} onChange={(v) => update("annual_income", v)} min={15000} max={280000} step={1000} suffix=" USD" />
              <NumberField label="Requested loan amount" value={form.loan_amount} onChange={(v) => update("loan_amount", v)} min={1000} max={50000} step={500} suffix=" USD" />
              <NumberField label="Credit score" value={form.credit_score} onChange={(v) => update("credit_score", v)} min={300} max={850} />
              <NumberField label="Employment length" value={form.employment_length_years} onChange={(v) => update("employment_length_years", v)} min={0} max={40} suffix=" yrs" />
              <NumberField label="Debt-to-income ratio" value={form.debt_to_income_ratio} onChange={(v) => update("debt_to_income_ratio", v)} min={0} max={0.65} step={0.01} />
              <NumberField label="Open credit lines" value={form.num_open_credit_lines} onChange={(v) => update("num_open_credit_lines", v)} min={0} max={20} />
              <NumberField label="Inquiries, last 6 months" value={form.num_credit_inquiries_last_6m} onChange={(v) => update("num_credit_inquiries_last_6m", v)} min={0} max={10} />
              <NumberField label="Credit utilization" value={form.credit_utilization_rate} onChange={(v) => update("credit_utilization_rate", v)} min={0} max={1} step={0.01} />
              <NumberField label="Past defaults" value={form.past_defaults} onChange={(v) => update("past_defaults", v)} min={0} max={5} />

              <label className="block">
                <div className="mb-1 text-xs text-white/50">Home ownership</div>
                <select
                  value={form.home_ownership}
                  onChange={(e) => update("home_ownership", e.target.value as ApplicantForm["home_ownership"])}
                  className="w-full rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-sm"
                >
                  <option value="RENT">Rent</option>
                  <option value="MORTGAGE">Mortgage</option>
                  <option value="OWN">Own</option>
                </select>
              </label>

              <label className="block">
                <div className="mb-1 text-xs text-white/50">Loan purpose</div>
                <select
                  value={form.loan_purpose}
                  onChange={(e) => update("loan_purpose", e.target.value as ApplicantForm["loan_purpose"])}
                  className="w-full rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-sm"
                >
                  <option value="debt_consolidation">Debt consolidation</option>
                  <option value="home_improvement">Home improvement</option>
                  <option value="education">Education</option>
                  <option value="business">Business</option>
                  <option value="medical">Medical</option>
                  <option value="other">Other</option>
                </select>
              </label>

              <button
                onClick={runAssessment}
                disabled={loading}
                className="mt-2 w-full rounded-xl bg-white py-3 text-sm font-semibold text-black transition hover:bg-sky-200 disabled:opacity-50"
              >
                {loading ? "Scoring…" : "Run risk assessment"}
              </button>
              {error && (
                <p className="text-xs text-rose-300">
                  {error} — is the backend running at {API_URL}?
                </p>
              )}
            </div>
          </GlassPanel>

          {/* Results */}
          <div className="space-y-6 lg:col-span-3">
            <GlassPanel className="p-6">
              {!result ? (
                <div className="flex h-40 items-center justify-center text-sm text-white/40">
                  Run an assessment to see results here.
                </div>
              ) : (
                <div className="grid grid-cols-1 gap-6 sm:grid-cols-3">
                  <div>
                    <p className="text-xs text-white/40">Risk score</p>
                    <p className="mt-1 text-4xl font-semibold tabular-nums">
                      {(result.risk_score * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-white/40">Risk tier</p>
                    <p
                      className={`mt-1 inline-flex rounded-full px-3 py-1 text-sm font-medium ring-1 ${tierStyle.bg} ${tierStyle.text} ${tierStyle.ring}`}
                    >
                      {result.risk_tier.replace("_", " ")}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-white/40">Recommendation</p>
                    <p className="mt-1 text-lg font-semibold">
                      {result.recommendation.replace(/_/g, " ")}
                    </p>
                  </div>
                  {result.recommended_interest_rate_pct != null && (
                    <div>
                      <p className="text-xs text-white/40">Suggested rate</p>
                      <p className="mt-1 text-lg font-semibold">
                        {result.recommended_interest_rate_pct}%
                      </p>
                    </div>
                  )}
                  {result.recommended_loan_amount != null && (
                    <div>
                      <p className="text-xs text-white/40">Suggested amount</p>
                      <p className="mt-1 text-lg font-semibold">
                        ${result.recommended_loan_amount.toLocaleString()}
                      </p>
                    </div>
                  )}
                </div>
              )}
            </GlassPanel>

            {result && (
              <GlassPanel className="p-6">
                <h2 className="mb-2 text-sm font-medium text-white/70">In plain English</h2>
                <p className="text-sm leading-relaxed text-white/80">{result.narrative}</p>
              </GlassPanel>
            )}

            <GlassPanel className="p-6">
              <h2 className="mb-1 text-sm font-medium text-white/70">Why the model decided this</h2>
              <p className="mb-5 text-xs text-white/40">
                SHAP values — how much each factor pushed the prediction above
                or below the model&apos;s baseline.
              </p>
              {chartData.length === 0 ? (
                <div className="flex h-56 items-center justify-center text-sm text-white/40">
                  No explanation yet.
                </div>
              ) : (
                <ResponsiveContainer width="100%" height={Math.max(chartData.length * 36, 200)}>
                  <BarChart data={chartData} layout="vertical" margin={{ left: 24 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" horizontal={false} />
                    <XAxis type="number" stroke="rgba(255,255,255,0.3)" fontSize={11} />
                    <YAxis type="category" dataKey="name" width={160} stroke="rgba(255,255,255,0.3)" fontSize={11} />
                    <Tooltip
                      contentStyle={{
                        background: "#0a0a0c",
                        border: "1px solid rgba(255,255,255,0.1)",
                        borderRadius: 12,
                        fontSize: 12,
                      }}
                      formatter={(value) => Number(value).toFixed(4)}
                    />
                    <Bar dataKey="shap" radius={[0, 6, 6, 0]}>
                      {chartData.map((entry, index) => (
                        <Cell key={index} fill={entry.shap > 0 ? "#fb7185" : "#38bdf8"} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              )}
              <div className="mt-4 flex gap-4 text-xs text-white/40">
                <span className="flex items-center gap-1.5">
                  <span className="h-2 w-2 rounded-full bg-rose-400" /> increases risk
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="h-2 w-2 rounded-full bg-sky-400" /> decreases risk
                </span>
              </div>
            </GlassPanel>

            {result && result.counterfactuals.length > 0 && (
              <GlassPanel className="p-6">
                <h2 className="mb-1 text-sm font-medium text-white/70">What would it take?</h2>
                <p className="mb-5 text-xs text-white/40">
                  The smallest realistic changes that would move this applicant to a
                  better risk tier.
                </p>
                <div className="space-y-3">
                  {result.counterfactuals.map((cf, i) => (
                    <div
                      key={i}
                      className="rounded-2xl border border-white/10 bg-white/[0.03] p-4"
                    >
                      <p className="text-sm text-white/80">{cf.description}</p>
                      <p className="mt-2 text-xs text-white/40">
                        New predicted risk:{" "}
                        <span className="text-white/70">
                          {(cf.new_risk_score * 100).toFixed(1)}%
                        </span>{" "}
                        ({cf.new_tier.replace("_", " ")})
                      </p>
                    </div>
                  ))}
                </div>
              </GlassPanel>
            )}

            {result && result.counterfactuals.length === 0 && result.risk_tier === "LOW" && (
              <GlassPanel className="p-6">
                <p className="text-sm text-white/60">
                  This applicant is already in the lowest risk tier — there's nothing
                  meaningful to suggest changing.
                </p>
              </GlassPanel>
            )}
          </div>
        </div>
      </div>
      </main>
    </>
  );
}
