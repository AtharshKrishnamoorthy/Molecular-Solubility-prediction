// ── Predict ──────────────────────────────────────────────

export interface PredictionItem {
  smiles: string;
  logS: number | null;
  error?: string;
}

export interface PredictResponse {
  predictions: PredictionItem[];
}

// ── Details ──────────────────────────────────────────────

export interface MoleculeDetails {
  iupac_name: string | null;
  common_names: string[];
  canonical_smiles: string;
  inchikey: string;
  formula: string;
  mol_weight: number;
  logP: number;
  h_donors: number;
  h_acceptors: number;
  rotatable_bonds: number;
  tpsa: number;
  rings: number;
  aromatic_rings: number;
  atom_count: number;
  heavy_atom_count: number;
}

// ── Combined result per molecule (used in dashboard) ─────

export interface MoleculeResult {
  smiles: string;
  logS: number | null;
  error?: string;
  details: MoleculeDetails | null;
}

// ── Auth ─────────────────────────────────────────────────

export interface SignupResponse {
  message: string;
  user_id: string | null;
}

export interface SigninResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  user_id: string;
}

// ── User ─────────────────────────────────────────────────

export interface User {
  id: string;
  email: string;
  created_at: string;
}

export interface CreateUserBody {
  email: string;
  hashed_password: string;
}

export interface UpdateEmailBody {
  new_email: string;
}

export interface UpdatePasswordBody {
  new_hashed_password: string;
}

// ── Analytics ────────────────────────────────────────────

export interface AnalyticsRecord {
  id: string;
  user_id: string;
  smiles: string;
  logS: number | null;
  iupac_name: string | null;
  common_names: string[] | null;
  canonical_smiles: string | null;
  inchikey: string | null;
  formula: string | null;
  mol_weight: number | null;
  logP: number | null;
  tpsa: number | null;
  h_donors: number | null;
  h_acceptors: number | null;
  rotatable_bonds: number | null;
  rings: number | null;
  aromatic_rings: number | null;
  atom_count: number | null;
  heavy_atom_count: number | null;
  created_at: string;
}

export interface CreateAnalyticsBody {
  user_id: string;
  user_email?: string | null;
  smiles: string;
  logS?: number | null;
  details?: Partial<MoleculeDetails> | null;
}
