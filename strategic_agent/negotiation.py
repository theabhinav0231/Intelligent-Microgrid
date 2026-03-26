"""
strategic_agent/negotiation.py
=============================
Client for the central P2P Marketplace API.
"""
import logging
import requests
from typing import Dict, Any, List, Optional
from edge import config

logger = logging.getLogger("StrategicAgent.Market")

class MarketplaceClient:
    """
    Handles interactions with the FastAPI marketplace.
    Now with API Key Auth.
    """
    def __init__(self, 
                 base_url: str = config.MARKETPLACE_URL,
                 api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key  = api_key

    def _get_headers(self) -> Dict[str, str]:
        """Provides headers for authenticated requests."""
        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    def get_market_snapshot(self) -> Dict[str, Any]:
        """Fetches the current order book and best prices."""
        try:
            resp = requests.get(f"{self.base_url}/orders", timeout=5)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Failed to fetch market snapshot: {e}")
            return {}

    def get_market_stats(self) -> Dict[str, Any]:
        """Fetches aggregate market statistics."""
        try:
            resp = requests.get(f"{self.base_url}/stats", timeout=5)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Failed to fetch market stats: {e}")
            return {}

    def get_wallet_balance(self, node_id: str) -> Dict[str, Any]:
        """Checks the financial balance of the node."""
        try:
            resp = requests.get(f"{self.base_url}/wallet/{node_id}", timeout=5)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Failed to fetch wallet info: {e}")
            return {"balance_inr": 0.0}

    def get_node_trades(self, node_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Fetches recent trade history for a specific node."""
        try:
            resp = requests.get(f"{self.base_url}/trades", params={"limit": limit}, timeout=5)
            resp.raise_for_status()
            # Result set filtering locally in prototype for node_id
            return [t for t in resp.json() if t["buyer_node_id"] == node_id or t["seller_node_id"] == node_id]
        except Exception as e:
            logger.error(f"Failed to fetch node trades: {e}")
            return []

    def place_order(self, 
                    node_id: str, 
                    order_type: str, 
                    quantity_kwh: float, 
                    price_per_kwh: float) -> Dict[str, Any]:
        """Submits a limit order to the marketplace with API Key."""
        payload = {
            "node_id": node_id,
            "order_type": order_type.lower(),
            "quantity_kwh": round(quantity_kwh, 4),
            "price_per_kwh": round(price_per_kwh, 2)
        }
        try:
            resp = requests.post(
                f"{self.base_url}/orders", 
                json=payload, 
                headers=self._get_headers(),
                timeout=5
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return {"error": str(e), "matched": False}

    def discover_best_peer(self, order_type: str) -> Optional[str]:
        """
        Helper to find the best counterparty in the current order book.
        Returns the node_id of the best match.
        """
        snapshot = self.get_market_snapshot()
        # If we want to BUY, we look at PENDING SELL orders
        if order_type.upper() == "BUY":
            sells = snapshot.get("pending_sell_orders", [])
            if sells:
                return sells[0]["node_id"]
        # If we want to SELL, we look at PENDING BUY orders
        elif order_type.upper() == "SELL":
            buys = snapshot.get("pending_buy_orders", [])
            if buys:
                return buys[0]["node_id"]
        return None
