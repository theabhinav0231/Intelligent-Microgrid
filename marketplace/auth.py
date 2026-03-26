"""
auth.py
=======
API Key Authentication for the P2P Energy Marketplace.
OOP Design: Abstract Base Auth Provider + Concrete APIKey implementation.
Integrates with FastAPI security dependencies.
"""

import hashlib
import secrets
from abc import ABC, abstractmethod
from typing import Optional, Tuple
from fastapi import Security, HTTPException, status, Depends
from fastapi.security.api_key import APIKeyHeader

from .repositories import NodeRepository
from .database import get_db
from sqlalchemy.orm import Session

# Look for X-API-Key in HTTP headers
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

class BaseAuthProvider(ABC):
    """Contract for any authentication mechanism (API Key, JWT, etc.)."""
    @abstractmethod
    def authenticate(self, api_key: str) -> Optional[str]:
        """Returns node_id if key is valid, else None."""
        pass

class APIKeyAuthService(BaseAuthProvider):
    """
    Concrete API Key provider.
    Validates hashed API keys against the database (NodeRepository).
    """

    def __init__(self, node_repo: NodeRepository):
        self._node_repo = node_repo

    def authenticate(self, api_key: str) -> Optional[str]:
        """Validates key by hashing and checking Node store."""
        if not api_key:
            return None
            
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        node = self._node_repo.get_by_api_key_hash(key_hash)
        
        if node and node.is_active:
            return node.id
        return None

    @staticmethod
    def generate_api_key() -> Tuple[str, str]:
        """Returns (plaintext_key, hashed_key). Plaintext is shown ONCE on registration."""
        plaintext = secrets.token_hex(32)
        hashed    = hashlib.sha256(plaintext.encode()).hexdigest()
        return plaintext, hashed


# ── FastAPI Security Dependency ──

def get_auth_service(db: Session = Depends(get_db)) -> APIKeyAuthService:
    """Injects auth service with DB access."""
    return APIKeyAuthService(NodeRepository(db))

def authenticate_node(
    api_key: str = Security(API_KEY_HEADER),
    auth_service: APIKeyAuthService = Depends(get_auth_service)
) -> str:
    """
    FastAPI dependency used to protect write-endpoints.
    Returns the authenticated node_id if success.
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-API-Key header missing"
        )
        
    node_id = auth_service.authenticate(api_key)
    if not node_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or inactive API Key"
        )
        
    return node_id
