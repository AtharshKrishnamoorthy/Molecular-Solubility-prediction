import os
import sys
import pathlib
from uuid import UUID

# Ensure backend/ is on the path so db.*, dashboard.* imports resolve
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from supabase import create_client, Client
from dotenv import load_dotenv

from db.database import SessionLocal
from dashboard.main import create_user, get_user_by_email

load_dotenv()

_supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_ANON_KEY"),
)


def _upsert_user_in_db(email: str, supabase_id: str | None) -> None:
    """Mirror a Supabase auth user into our users table using their Supabase UUID."""
    db = SessionLocal()
    try:
        if not get_user_by_email(db, email):
            uid = UUID(str(supabase_id)) if supabase_id else None
            create_user(db, email=email, hashed_password="SUPABASE_MANAGED", user_id=uid)
    except Exception:
        db.rollback()
    finally:
        db.close()


def signup_user(email: str, password: str) -> dict:
    response = _supabase.auth.sign_up({"email": email, "password": password})

    if response.user:
        _upsert_user_in_db(email, str(response.user.id))

    return {
        "user":    response.user,
        "session": response.session,
    }


def signin_user(email: str, password: str) -> dict:
    response = _supabase.auth.sign_in_with_password({"email": email, "password": password})

    # Mirror the user into our DB on every login.
    # This auto-fixes accounts created before DB mirroring was added.
    if response.user:
        _upsert_user_in_db(email, str(response.user.id))

    return {
        "user":    response.user,
        "session": response.session,
    }
