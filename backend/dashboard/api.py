import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from uuid import UUID
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from db.database import get_db
from dashboard.main import (
    get_user_by_id, get_user_by_email, create_user,
    update_user_email, update_user_password, delete_user,
    create_analytics, get_analytics_by_user,
    get_analytics_by_id, delete_analytics, delete_all_analytics_for_user,
)

app = FastAPI(title="Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
#  Pydantic schemas
# ──────────────────────────────────────────────

class CreateUserBody(BaseModel):
    email: EmailStr
    hashed_password: str

class UpdateEmailBody(BaseModel):
    new_email: EmailStr

class UpdatePasswordBody(BaseModel):
    new_hashed_password: str

class CreateAnalyticsBody(BaseModel):
    user_id: UUID
    user_email: str | None = None
    smiles: str
    logS: float | None = None
    details: dict | None = None


# ──────────────────────────────────────────────
#  Users
# ──────────────────────────────────────────────

@app.get("/users/{user_id}")
def get_user(user_id: UUID, db: Session = Depends(get_db)):
    user = get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@app.get("/users/by-email/{email}")
def get_user_email(email: str, db: Session = Depends(get_db)):
    user = get_user_by_email(db, email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@app.post("/users", status_code=201)
def add_user(body: CreateUserBody, db: Session = Depends(get_db)):
    if get_user_by_email(db, body.email):
        raise HTTPException(status_code=409, detail="Email already registered")
    return create_user(db, body.email, body.hashed_password)


@app.patch("/users/{user_id}/email")
def change_email(user_id: UUID, body: UpdateEmailBody, db: Session = Depends(get_db)):
    user = update_user_email(db, user_id, body.new_email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@app.patch("/users/{user_id}/password")
def change_password(user_id: UUID, body: UpdatePasswordBody, db: Session = Depends(get_db)):
    user = update_user_password(db, user_id, body.new_hashed_password)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@app.delete("/users/{user_id}")
def remove_user(user_id: UUID, db: Session = Depends(get_db)):
    if not delete_user(db, user_id):
        raise HTTPException(status_code=404, detail="User not found")
    return {"deleted": True}


# ──────────────────────────────────────────────
#  Analytics
# ──────────────────────────────────────────────

@app.post("/analytics", status_code=201)
def add_analytics(body: CreateAnalyticsBody, db: Session = Depends(get_db)):
    # Auto-upsert user row so analytics FK never fires,
    # covering accounts created before DB mirroring existed.
    if body.user_email and not get_user_by_id(db, body.user_id):
        try:
            create_user(db, email=body.user_email,
                        hashed_password="SUPABASE_MANAGED",
                        user_id=body.user_id)
        except IntegrityError:
            db.rollback()  # email already exists under a different id — ignore
    try:
        return create_analytics(db, body.user_id, body.smiles, body.logS, body.details)
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=422,
            detail="User record could not be resolved. Please sign out and sign back in.",
        )


@app.get("/analytics/user/{user_id}")
def get_user_analytics(user_id: UUID, db: Session = Depends(get_db)):
    return get_analytics_by_user(db, user_id)


@app.get("/analytics/{analytics_id}")
def get_one_analytics(analytics_id: UUID, db: Session = Depends(get_db)):
    row = get_analytics_by_id(db, analytics_id)
    if not row:
        raise HTTPException(status_code=404, detail="Analytics record not found")
    return row


@app.delete("/analytics/{analytics_id}")
def remove_analytics(analytics_id: UUID, db: Session = Depends(get_db)):
    if not delete_analytics(db, analytics_id):
        raise HTTPException(status_code=404, detail="Analytics record not found")
    return {"deleted": True}


@app.delete("/analytics/user/{user_id}/all")
def remove_all_user_analytics(user_id: UUID, db: Session = Depends(get_db)):
    count = delete_all_analytics_for_user(db, user_id)
    return {"deleted_count": count}
