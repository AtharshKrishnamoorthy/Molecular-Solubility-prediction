import type { MoleculeDetails, MoleculeResult, PredictResponse } from "./types";

const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// ── Raw endpoint calls ────────────────────────────────────

export async function predictLogS(smiles: string[]): Promise<PredictResponse> {
  const res = await fetch(`${BASE_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ smiles }),
  });
  if (!res.ok) throw new Error(`Predict failed: ${res.statusText}`);
  return res.json();
}

export async function getMoleculeDetails(smiles: string): Promise<MoleculeDetails | null> {
  const res = await fetch(`${BASE_URL}/details`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ smiles }),
  });
  if (!res.ok) return null;
  return res.json();
}

export async function healthCheck(): Promise<boolean> {
  try {
    const res = await fetch(`${BASE_URL}/health`);
    const data = await res.json();
    return data?.status === "ok";
  } catch {
    return false;
  }
}

// ── Combined helper used by the dashboard ─────────────────
// Calls /predict (batch) + /details (per molecule) in parallel

export async function analyzeSmiles(smilesList: string[]): Promise<MoleculeResult[]> {
  const [predictRes, detailsResults] = await Promise.all([
    predictLogS(smilesList),
    Promise.all(smilesList.map((smi) => getMoleculeDetails(smi))),
  ]);

  return predictRes.predictions.map((item, i) => ({
    smiles: item.smiles,
    logS: item.logS,
    error: item.error,
    details: detailsResults[i],
  }));
}
