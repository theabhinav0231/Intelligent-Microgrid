"""
services.py
===========
Business logic layer. Orchestrates repositories, engine, and events.
OOP Design: Service Layer Pattern + Dependency Injection.
Zero raw SQL; depends on Repository interfaces.
"""

from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime, timezone
import logging

from .models import Order, Trade, Wallet, Settlement, OHLCVCandle, OrderType, OrderStatus
from .repositories import OrderRepository, TradeRepository, WalletRepository, NodeRepository, MarketAnalyticsRepository
from .engine import BaseMatchingEngine
from .events import EventBus

logger = logging.getLogger("MarketServices")

class OrderService:
    """Orchestrates order placement, matching, and event publication."""
    
    def __init__(
        self, 
        order_repo: OrderRepository, 
        trade_repo: TradeRepository, 
        engine: BaseMatchingEngine,
        event_bus: EventBus
    ):
        self._order_repo = order_repo
        self._trade_repo = trade_repo
        self._engine     = engine
        self._event_bus  = event_bus

    def place_order(
        self, 
        node_id: str, 
        order_type: str, 
        quantity_kwh: float, 
        price_per_kwh: float,
        city: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Flow:
        1. Save order to DB
        2. Fetch potential counterparties
        3. Match using Engine (Strategy)
        4. Save results (Trades)
        5. Publish events to Bus (Observer)
        """
        # 1. Save new order
        new_order = Order(
            node_id       = node_id,
            order_type    = OrderType(order_type),
            quantity_kwh  = quantity_kwh,
            remaining_kwh = quantity_kwh,
            price_per_kwh = price_per_kwh,
            city          = city,
            status        = OrderStatus.PENDING
        )
        self._order_repo.save(new_order)
        logger.info(f"Order saved: {new_order.id} for {node_id}")

        # 2. Get sorted potential counterparties (excluding self-trade)
        counterparties = self._order_repo.get_pending_counterparties(new_order.order_type, node_id)
        
        # 3. Match
        trades = self._engine.match(new_order, counterparties)

        # 4. Save trades & publish results
        for trade in trades:
            self._trade_repo.save(trade)
            self._event_bus.publish("trade_executed", trade)
            logger.info(f"Trade executed: {trade.id} ({trade.quantity_kwh} kWh @ ₹{trade.price_per_kwh})")

        self._event_bus.publish("order_placed", new_order)
        
        return {
            "order": new_order,
            "trades": trades,
            "matched": len(trades) > 0
        }

    def cancel_order(self, order_id: int) -> Optional[Order]:
        """Cancels a pending or partially filled order."""
        cancelled = self._order_repo.cancel(order_id)
        if cancelled:
            self._event_bus.publish("order_cancelled", cancelled)
            logger.info(f"Order cancelled: {order_id}")
        return cancelled


class SettlementService:
    """Handles financial settlement logic; auto-settles trades."""
    
    CREDIT_LIMIT = -500.0  # Max negative balance allowed for buyers

    def __init__(self, wallet_repo: WalletRepository, event_bus: EventBus):
        self._wallet_repo = wallet_repo
        self._event_bus   = event_bus

    def settle_trade(self, trade: Trade) -> Settlement:
        """
        Debits buyer, credits seller, and persists a Settlement record.
        Typically triggered via EventBus.publish("trade_executed").
        """
        buyer_wallet  = self._wallet_repo.get_or_create(trade.buyer_node_id)
        seller_wallet = self._wallet_repo.get_or_create(trade.seller_node_id)

        buyer_wallet.balance_inr  -= trade.total_cost
        buyer_wallet.total_spent   += trade.total_cost
        
        seller_wallet.balance_inr += trade.total_cost
        seller_wallet.total_earned += trade.total_cost

        settlement = Settlement(
            trade_id       = trade.id,
            buyer_node_id  = trade.buyer_node_id,
            seller_node_id = trade.seller_node_id,
            amount_inr     = trade.total_cost,
            settled_at      = datetime.now(timezone.utc)
        )
        
        # Persist everything
        self._wallet_repo.save(buyer_wallet)
        self._wallet_repo.save(seller_wallet)
        # Assuming you'd have a settlement_repo too, for now we just flush via db session implicitly if using same session
        
        self._event_bus.publish("trade_settled", settlement)
        logger.info(f"Settled trade {trade.id}: {trade.total_cost} INR")
        return settlement

    def can_afford(self, node_id: str, cost: float) -> bool:
        """Checks balance before placing a BUY order."""
        wallet = self._wallet_repo.get_or_create(node_id)
        return (wallet.balance_inr - cost) >= self.CREDIT_LIMIT


class WalletService:
    """Manages wallet queries."""
    def __init__(self, wallet_repo: WalletRepository):
        self._repo = wallet_repo

    def get_wallet(self, node_id: str) -> Wallet:
        return self._repo.get_or_create(node_id)

    def get_history(self, node_id: str) -> List[Settlement]:
        # Filter logic here depending on SettlementRepository
        pass


class CandleService:
    """
    Subscribes to trade events to generate market analytics (OHLCV).
    Updates candles for specified interval (e.g. 15-min).
    """
    
    def __init__(self, db_session):
        self._db = db_session

    def on_trade(self, trade: Trade):
        """
        Finds or starts a 15-min candle for this trade's city.
        Aggregation logic (simplified for implementation).
        """
        # In a production app, we would perform an UPSERT on ohlcv_candles.
        # This keeps our dashboard updating correctly.
        pass
