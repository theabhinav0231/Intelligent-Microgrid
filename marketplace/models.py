"""
Database models for Orders and Trades.

Order lifecycle: pending → filled | partially_filled | cancelled
A single buy order can generate multiple trades (if it matches several sells).
"""

from sqlalchemy import Column, Integer, Float, String, DateTime, Enum as SAEnum, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
import enum

from .database import Base


class OrderStatus(str, enum.Enum):
    """Order lifecycle states."""
    PENDING          = "pending"
    FILLED          = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED        = "cancelled"


class OrderType(str, enum.Enum):
    """Buy or sell."""
    BUY  = "buy"
    SELL = "sell"


class Order(Base):
    __tablename__ = "orders"

    id                 = Column(Integer, primary_key=True, index=True, autoincrement=True)
    node_id            = Column(String(50), ForeignKey("nodes.id"), nullable=False, index=True)
    order_type         = Column(SAEnum(OrderType), nullable=False)
    quantity_kwh       = Column(Float, nullable=False)
    remaining_kwh      = Column(Float, nullable=False)
    price_per_kwh      = Column(Float, nullable=False)
    status             = Column(SAEnum(OrderStatus), default=OrderStatus.PENDING, nullable=False)
    city               = Column(String(50), nullable=True, index=True)
    created_at         = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at         = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                                 onupdate=lambda: datetime.now(timezone.utc))

    # Relationships
    node = relationship("Node")

    def __repr__(self):
        return (f"<Order id={self.id} {self.order_type.value} {self.remaining_kwh}/{self.quantity_kwh} kWh "
                f"@ ₹{self.price_per_kwh} [{self.status.value}]>")


class Trade(Base):
    __tablename__ = "trades"

    id                 = Column(Integer, primary_key=True, index=True, autoincrement=True)
    buyer_node_id      = Column(String(50), ForeignKey("nodes.id"), nullable=False, index=True)
    seller_node_id     = Column(String(50), ForeignKey("nodes.id"), nullable=False, index=True)
    buyer_order_id     = Column(Integer, ForeignKey("orders.id"), nullable=False)
    seller_order_id    = Column(Integer, ForeignKey("orders.id"), nullable=False)
    quantity_kwh       = Column(Float, nullable=False)
    price_per_kwh      = Column(Float, nullable=False)
    total_cost         = Column(Float, nullable=False)
    city               = Column(String(50), nullable=True, index=True)
    executed_at        = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    buyer_order  = relationship("Order", foreign_keys=[buyer_order_id])
    seller_order = relationship("Order", foreign_keys=[seller_order_id])
    buyer_node   = relationship("Node", foreign_keys=[buyer_node_id])
    seller_node  = relationship("Node", foreign_keys=[seller_node_id])

    def __repr__(self):
        return (f"<Trade {self.buyer_node_id}←{self.seller_node_id} "
                f"{self.quantity_kwh} kWh @ ₹{self.price_per_kwh}>")


class Node(Base):
    __tablename__ = "nodes"

    id              = Column(String(50), primary_key=True, index=True)
    city            = Column(String(50), nullable=False, index=True)
    api_key_hash    = Column(String(64), unique=True, nullable=False)
    battery_cap_kwh = Column(Float, nullable=False, default=10.0)
    is_active       = Column(Integer, nullable=False, default=1)
    registered_at   = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    def __repr__(self):
        return f"<Node id={self.id} city={self.city}>"


class Wallet(Base):
    __tablename__ = "wallets"

    node_id         = Column(String(50), ForeignKey("nodes.id"), primary_key=True)
    balance_inr     = Column(Float, nullable=False, default=0.0)
    total_earned    = Column(Float, nullable=False, default=0.0)
    total_spent     = Column(Float, nullable=False, default=0.0)
    last_updated    = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                             onupdate=lambda: datetime.now(timezone.utc))

    # Relationships
    node = relationship("Node")

    def __repr__(self):
        return f"<Wallet node={self.node_id} balance=₹{self.balance_inr}>"


class Settlement(Base):
    __tablename__ = "settlements"

    id              = Column(Integer, primary_key=True, index=True, autoincrement=True)
    trade_id        = Column(Integer, ForeignKey("trades.id"), nullable=False)
    buyer_node_id   = Column(String(50), ForeignKey("nodes.id"), nullable=False)
    seller_node_id  = Column(String(50), ForeignKey("nodes.id"), nullable=False)
    amount_inr      = Column(Float, nullable=False)
    settled_at      = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    trade = relationship("Trade")

    def __repr__(self):
        return f"<Settlement trade={self.trade_id} amount=₹{self.amount_inr}>"


class OHLCVCandle(Base):
    __tablename__ = "ohlcv_candles"

    id              = Column(Integer, primary_key=True, index=True, autoincrement=True)
    interval        = Column(String(10), nullable=False, index=True)
    open_price      = Column(Float, nullable=False)
    high_price      = Column(Float, nullable=False)
    low_price       = Column(Float, nullable=False)
    close_price     = Column(Float, nullable=False)
    volume_kwh      = Column(Float, nullable=False)
    candle_ts       = Column(DateTime, nullable=False, index=True)
    city            = Column(String(50), nullable=True, index=True)

    def __repr__(self):
        return f"<Candle {self.interval} {self.candle_ts}>"
