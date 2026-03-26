"""
repositories.py
===============
All database access logic encapsulated in repository classes.
OOP Design: Abstract Base Class + Concrete Repository Pattern.
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import and_, func, distinct
from datetime import datetime, timezone

from .models import Order, Trade, Node, Wallet, Settlement, OHLCVCandle, OrderStatus, OrderType

T = TypeVar("T")

# ── Abstract Base Class ──
class BaseRepository(ABC, Generic[T]):
    """Contract for all repositories."""
    @abstractmethod
    def get_by_id(self, id: any) -> Optional[T]:
        pass

    @abstractmethod
    def save(self, entity: T) -> T:
        pass

# ── Concrete Repositories ──

class OrderRepository(BaseRepository[Order]):
    """Handles all DB operations for Orders."""
    def __init__(self, db: Session):
        self._db = db

    def get_by_id(self, id: int) -> Optional[Order]:
        return self._db.query(Order).filter(Order.id == id).first()

    def save(self, entity: Order) -> Order:
        self._db.add(entity)
        self._db.flush()
        return entity

    def get_pending_counterparties(self, order_type: OrderType, exclude_node_id: str) -> List[Order]:
        """Finds sorted potential counterparties for matching."""
        opposite = OrderType.SELL if order_type == OrderType.BUY else OrderType.BUY
        
        query = self._db.query(Order).filter(
            and_(
                Order.order_type == opposite,
                Order.status.in_([OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED]),
                Order.remaining_kwh > 0.0001,
                Order.node_id != exclude_node_id,
            )
        )
        
        if opposite == OrderType.SELL:
            # For a buyer, finding cheapest sell orders first
            query = query.order_by(Order.price_per_kwh.asc(), Order.created_at.asc())
        else:
            # For a seller, finding highest buy orders first
            query = query.order_by(Order.price_per_kwh.desc(), Order.created_at.asc())
            
        return query.all()

    def cancel(self, order_id: int) -> Optional[Order]:
        order = self.get_by_id(order_id)
        if order and order.status in [OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED]:
            order.status = OrderStatus.CANCELLED
            self._db.flush()
            return order
        return None


class TradeRepository(BaseRepository[Trade]):
    """Handles all DB operations for Trades."""
    def __init__(self, db: Session):
        self._db = db

    def get_by_id(self, id: int) -> Optional[Trade]:
        return self._db.query(Trade).filter(Trade.id == id).first()

    def save(self, entity: Trade) -> Trade:
        self._db.add(entity)
        self._db.flush()
        return entity

    def get_recent(self, n: int = 10, city: Optional[str] = None) -> List[Trade]:
        query = self._db.query(Trade)
        if city:
            query = query.filter(Trade.city == city)
        return query.order_by(Trade.executed_at.desc()).limit(n).all()

    def get_by_node(self, node_id: str) -> List[Trade]:
        return (
            self._db.query(Trade)
            .filter((Trade.buyer_node_id == node_id) | (Trade.seller_node_id == node_id))
            .order_by(Trade.executed_at.desc())
            .all()
        )


class NodeRepository(BaseRepository[Node]):
    """Handles Node registration and metadata."""
    def __init__(self, db: Session):
        self._db = db

    def get_by_id(self, id: str) -> Optional[Node]:
        return self._db.query(Node).filter(Node.id == id).first()

    def get_all(self, city: Optional[str] = None) -> List[Node]:
        query = self._db.query(Node)
        if city:
            query = query.filter(Node.city == city)
        return query.all()

    def save(self, entity: Node) -> Node:
        self._db.add(entity)
        self._db.flush()
        return entity

    def get_by_api_key_hash(self, key_hash: str) -> Optional[Node]:
        return self._db.query(Node).filter(Node.api_key_hash == key_hash).first()


class WalletRepository(BaseRepository[Wallet]):
    """Handles wallet balances."""
    def __init__(self, db: Session):
        self._db = db

    def get_by_id(self, node_id: str) -> Optional[Wallet]:
        return self._db.query(Wallet).filter(Wallet.node_id == node_id).first()

    def get_or_create(self, node_id: str) -> Wallet:
        wallet = self.get_by_id(node_id)
        if not wallet:
            wallet = Wallet(node_id=node_id, balance_inr=0.0)
            self._db.add(wallet)
            self._db.flush()
        return wallet

    def save(self, entity: Wallet) -> Wallet:
        self._db.add(entity)
        self._db.flush()
        return entity


class MarketAnalyticsRepository:
    """Handles complex queries for market statistics."""
    def __init__(self, db: Session):
        self._db = db

    def get_stats(self, city: Optional[str] = None) -> dict:
        trade_query = self._db.query(Trade)
        order_query = self._db.query(Order).filter(Order.status.in_([OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED]))
        
        if city:
            trade_query = trade_query.filter(Trade.city == city)
            order_query = order_query.filter(Order.city == city)
            
        total_trades = trade_query.count()
        total_volume = trade_query.with_entities(func.sum(Trade.quantity_kwh)).scalar() or 0.0
        total_value  = trade_query.with_entities(func.sum(Trade.total_cost)).scalar() or 0.0
        avg_price    = trade_query.with_entities(func.avg(Trade.price_per_kwh)).scalar()
        
        pending_count = order_query.count()
        active_nodes  = order_query.with_entities(func.count(distinct(Order.node_id))).scalar() or 0
        
        return {
            "total_trades": total_trades,
            "total_volume_kwh": round(total_volume, 4),
            "total_value_inr": round(total_value, 2),
            "average_price_per_kwh": round(avg_price, 2) if avg_price else None,
            "total_pending_orders": pending_count,
            "active_nodes": active_nodes,
        }
