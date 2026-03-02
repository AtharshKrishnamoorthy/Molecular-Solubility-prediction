import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from auth.main import signup_user, signin_user

app = FastAPI(title="Auth API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AuthRequest(BaseModel):
    email: EmailStr
    password: str


@app.post("/signup")
def signup(body: AuthRequest):
    try:
        result = signup_user(body.email, body.password)
        return {
            "message": "Signup successful. Check your email to confirm.",
            "user_id": result["user"].id if result["user"] else None,
        }
    except Exception as e:
        msg = str(e)
        if "rate limit" in msg.lower() or "429" in msg:
            raise HTTPException(status_code=429, detail="Too many signups. Please wait a few minutes and try again.")
        raise HTTPException(status_code=400, detail=msg)


@app.post("/signin")
def signin(body: AuthRequest):
    try:
        result = signin_user(body.email, body.password)
        session = result["session"]
        return {
            "access_token":  session.access_token,
            "refresh_token": session.refresh_token,
            "token_type":    "bearer",
            "user_id":       result["user"].id,
        }
    except Exception as e:
        msg = str(e)
        if "rate limit" in msg.lower() or "429" in msg:
            raise HTTPException(status_code=429, detail="Too many attempts. Please wait a few minutes and try again.")
        raise HTTPException(status_code=401, detail=msg)
