from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from main import predict_logS, get_molecule_details

app = FastAPI(title="Molecular Solubility API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    smiles: list[str]

class DetailsRequest(BaseModel):
    smiles: str



@app.get("/")
def root():
    return {"message": "Molecular Solubility Prediction API"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(body: PredictRequest):
    results = []
    for smi in body.smiles:
        logS = predict_logS(smi)
        if logS is None:
            results.append({"smiles": smi, "logS": None, "error": "Invalid SMILES"})
        else:
            results.append({"smiles": smi, "logS": logS})
    return {"predictions": results}


@app.post("/details")
def details(body: DetailsRequest):
    info = get_molecule_details(body.smiles)
    if info is None:
        raise HTTPException(status_code=400, detail="Invalid SMILES string")
    return info
