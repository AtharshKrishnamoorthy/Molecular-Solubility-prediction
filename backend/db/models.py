import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Float, Integer, DateTime, ForeignKey, Text, ARRAY
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

try:
    from .database import Base   # when used as a package
except ImportError:
    from database import Base    # when run directly alongside database.py


class User(Base):
    __tablename__ = "users"

    id         = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email      = Column(String, unique=True, nullable=False, index=True)
    password   = Column(String, nullable=False)          # store hashed passwords only
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    analytics  = relationship("Analytics", back_populates="user", cascade="all, delete-orphan")


class Analytics(Base):
    __tablename__ = "analytics"

    id                = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id           = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)

    # Input
    smiles            = Column(Text, nullable=False)

    # Prediction
    logS              = Column(Float, nullable=True)

    # Identity
    iupac_name        = Column(Text, nullable=True)
    common_names      = Column(ARRAY(Text), nullable=True)   # list of synonyms
    canonical_smiles  = Column(Text, nullable=True)
    inchikey          = Column(String, nullable=True)
    formula           = Column(String, nullable=True)

    # Physicochemical descriptors
    mol_weight        = Column(Float, nullable=True)
    logP              = Column(Float, nullable=True)
    tpsa              = Column(Float, nullable=True)
    h_donors          = Column(Integer, nullable=True)
    h_acceptors       = Column(Integer, nullable=True)
    rotatable_bonds   = Column(Integer, nullable=True)
    rings             = Column(Integer, nullable=True)
    aromatic_rings    = Column(Integer, nullable=True)
    atom_count        = Column(Integer, nullable=True)
    heavy_atom_count  = Column(Integer, nullable=True)

    created_at        = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    user              = relationship("User", back_populates="analytics")
