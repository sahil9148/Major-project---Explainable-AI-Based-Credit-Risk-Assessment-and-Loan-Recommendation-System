import type { Metadata } from "next";
import type { ReactNode } from "react";
import "./globals.css";

export const metadata: Metadata = {
  title: "RiskLens — Explainable Credit Risk",
  description:
    "XGBoost + SHAP powered credit risk scoring with a transparent, explainable loan recommendation dashboard.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
