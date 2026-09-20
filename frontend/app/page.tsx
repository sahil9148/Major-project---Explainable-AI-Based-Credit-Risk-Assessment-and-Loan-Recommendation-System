import dynamic from "next/dynamic";
import Link from "next/link";

// WebGL only makes sense in the browser -- skip the server render entirely.
const CanvasScroll = dynamic(() => import("@/components/CanvasScroll"), {
  ssr: false,
  loading: () => (
    <div className="flex h-screen w-full items-center justify-center bg-black text-white/40">
      Loading experience…
    </div>
  ),
});

const STEPS = [
  {
    title: "Submit the profile",
    body: "Income, credit history, and loan details go in through a single typed API request.",
  },
  {
    title: "The model scores it",
    body: "A gradient-boosted model trained on historical repayment outcomes returns a default probability in milliseconds.",
  },
  {
    title: "See the reasoning",
    body: "SHAP breaks the score down into the exact factors that moved it, in either direction.",
  },
];

export default function Home() {
  return (
    <main className="bg-black">
      <CanvasScroll />

      <section className="mx-auto max-w-4xl px-6 py-28 text-white">
        <h2 className="text-3xl font-semibold tracking-tight md:text-4xl">
          A model that shows its work
        </h2>
        <p className="mt-4 max-w-xl text-white/50">
          Most credit models are a black box: a score, and nothing else.
          RiskLens pairs an XGBoost risk model with SHAP explainability, so
          every recommendation comes with the reasons behind it.
        </p>

        <div className="mt-16 grid gap-10 sm:grid-cols-3">
          {STEPS.map((step, i) => (
            <div key={step.title}>
              <p className="text-sm text-white/30">{`0${i + 1}`}</p>
              <h3 className="mt-3 text-lg font-medium">{step.title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-white/50">{step.body}</p>
            </div>
          ))}
        </div>

        <div className="mt-16 flex flex-wrap items-center gap-4">
          <Link
            href="/dashboard"
            className="inline-flex items-center rounded-full bg-white px-6 py-3 text-sm font-semibold text-black transition hover:bg-sky-200"
          >
            Try the risk dashboard
          </Link>
          <Link
            href="/insights"
            className="inline-flex items-center rounded-full border border-white/15 px-6 py-3 text-sm font-medium text-white/70 transition hover:border-white/30 hover:text-white"
          >
            Model insights
          </Link>
          <Link
            href="/fairness"
            className="inline-flex items-center rounded-full border border-white/15 px-6 py-3 text-sm font-medium text-white/70 transition hover:border-white/30 hover:text-white"
          >
            Fairness audit
          </Link>
        </div>
      </section>
    </main>
  );
}
