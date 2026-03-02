import uuid
from uuid import UUID
from sqlalchemy.orm import Session

from db.models import User, Analytics


# ──────────────────────────────────────────────
#  USERS
# ──────────────────────────────────────────────

def get_user_by_id(db: Session, user_id: UUID) -> User | None:
    return db.query(User).filter(User.id == user_id).first()


def get_user_by_email(db: Session, email: str) -> User | None:
    return db.query(User).filter(User.email == email).first()


def create_user(db: Session, email: str, hashed_password: str, user_id: UUID | None = None) -> User:
    """
    Create a user row.  Pass user_id to use the Supabase auth UUID so that
    analytics FK references resolve correctly.  Omit it to auto-generate a UUID.
    """
    user = User(id=user_id if user_id is not None else uuid.uuid4(),
                email=email, password=hashed_password)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def update_user_email(db: Session, user_id: UUID, new_email: str) -> User | None:
    user = get_user_by_id(db, user_id)
    if not user:
        return None
    user.email = new_email
    db.commit()
    db.refresh(user)
    return user


def update_user_password(db: Session, user_id: UUID, new_hashed_password: str) -> User | None:
    user = get_user_by_id(db, user_id)
    if not user:
        return None
    user.password = new_hashed_password
    db.commit()
    db.refresh(user)
    return user


def delete_user(db: Session, user_id: UUID) -> bool:
    user = get_user_by_id(db, user_id)
    if not user:
        return False
    db.delete(user)
    db.commit()
    return True


# ──────────────────────────────────────────────
#  ANALYTICS
# ──────────────────────────────────────────────

def create_analytics(db: Session, user_id: UUID, smiles: str, logS: float | None, details: dict | None) -> Analytics:
    """
    Pass the dict returned by get_molecule_details() as `details`.
    All fields are optional — if details is None, only smiles + logS are stored.
    """
    row = Analytics(
        user_id          = user_id,
        smiles           = smiles,
        logS             = logS,
        iupac_name       = details.get("iupac_name")       if details else None,
        common_names     = details.get("common_names")      if details else None,
        canonical_smiles = details.get("canonical_smiles")  if details else None,
        inchikey         = details.get("inchikey")          if details else None,
        formula          = details.get("formula")           if details else None,
        mol_weight       = details.get("mol_weight")        if details else None,
        logP             = details.get("logP")              if details else None,
        tpsa             = details.get("tpsa")              if details else None,
        h_donors         = details.get("h_donors")          if details else None,
        h_acceptors      = details.get("h_acceptors")       if details else None,
        rotatable_bonds  = details.get("rotatable_bonds")   if details else None,
        rings            = details.get("rings")             if details else None,
        aromatic_rings   = details.get("aromatic_rings")    if details else None,
        atom_count       = details.get("atom_count")        if details else None,
        heavy_atom_count = details.get("heavy_atom_count")  if details else None,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def get_analytics_by_user(db: Session, user_id: UUID) -> list[Analytics]:
    """Return full history for a user, newest first."""
    return (
        db.query(Analytics)
        .filter(Analytics.user_id == user_id)
        .order_by(Analytics.created_at.desc())
        .all()
    )


def get_analytics_by_id(db: Session, analytics_id: UUID) -> Analytics | None:
    return db.query(Analytics).filter(Analytics.id == analytics_id).first()


def delete_analytics(db: Session, analytics_id: UUID) -> bool:
    row = get_analytics_by_id(db, analytics_id)
    if not row:
        return False
    db.delete(row)
    db.commit()
    return True


def delete_all_analytics_for_user(db: Session, user_id: UUID) -> int:
    """Deletes all analytics rows for a user. Returns count of deleted rows."""
    rows = db.query(Analytics).filter(Analytics.user_id == user_id)
    count = rows.count()
    rows.delete()
    db.commit()
    return count
