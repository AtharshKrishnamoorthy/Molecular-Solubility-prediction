import joblib
import numpy as np
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

model = joblib.load("solubility_model.pkl")


def aromatic_proportion(mol):
    aromatic = sum(1 for i in range(mol.GetNumAtoms()) if mol.GetAtomWithIdx(i).GetIsAromatic())
    return aromatic / Descriptors.HeavyAtomCount(mol)


def smiles_to_features(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    features = np.array([[
        Descriptors.MolLogP(mol),
        Descriptors.MolWt(mol),
        Descriptors.NumRotatableBonds(mol),
        aromatic_proportion(mol),
    ]])
    return mol, features


def predict_logS(smiles: str):
    mol, features = smiles_to_features(smiles)
    if features is None:
        return None
    return round(float(model.predict(features)[0]), 4)


def get_molecule_name(canonical_smiles: str):
    
    try:
        compounds = pcp.get_compounds(canonical_smiles, "smiles")
        if not compounds:
            return None, []
        c = compounds[0]
        iupac_name = c.iupac_name
        synonyms = c.synonyms[:5] if c.synonyms else []  # top 5 common names
        return iupac_name, synonyms
    except Exception:
        return None, []


def get_molecule_details(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    canonical = Chem.MolToSmiles(mol)
    iupac_name, synonyms = get_molecule_name(canonical)
    return {
        "iupac_name":         iupac_name,
        "common_names":       synonyms,
        "canonical_smiles":   canonical,
        "inchikey":           Chem.MolToInchiKey(mol),
        "formula":            rdMolDescriptors.CalcMolFormula(mol),
        "mol_weight":         round(Descriptors.MolWt(mol), 4),
        "logP":               round(Descriptors.MolLogP(mol), 4),
        "h_donors":           Descriptors.NumHDonors(mol),
        "h_acceptors":        Descriptors.NumHAcceptors(mol),
        "rotatable_bonds":    Descriptors.NumRotatableBonds(mol),
        "tpsa":               round(Descriptors.TPSA(mol), 4),
        "rings":              rdMolDescriptors.CalcNumRings(mol),
        "aromatic_rings":     rdMolDescriptors.CalcNumAromaticRings(mol),
        "atom_count":         mol.GetNumAtoms(),
        "heavy_atom_count":   mol.GetNumHeavyAtoms(),
    }
