"""
engine.py
=========
Matching engine for the P2P Energy Marketplace.
OOP Design: Strategy Pattern for different matching algorithms.
Stateless and pure logic; DB access belongs in repositories.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
from datetime import datetime, timezone

from .models import Order, Trade, OrderStatus, OrderType

class BaseMatchingEngine(ABC):
    """Abstract interface for matching algorithms."""
    
    @abstractmethod
    def match(self, incoming_order: Order, counterparties: List[Order]) -> List[Trade]:
        """Runs the matching logic between an incoming order and available counterparties."""
        pass

    @abstractmethod
    def compute_clearing_price(self, buy_price: float, sell_price: float) -> float:
        """Calculates the execution price given buy and sell limit prices."""
        pass


class CDAEngine(BaseMatchingEngine):
    """
    Continuous Double Auction (CDA) matching engine.
    Uses midpoint clearing: splits the spread equally between buyer and seller.
    """

    def compute_clearing_price(self, buy_price: float, sell_price: float) -> float:
        """Midpoint clearing: (buy_price + sell_price) / 2."""
        return round((buy_price + sell_price) / 2, 2)

    def match(self, incoming_order: Order, counterparties: List[Order]) -> List[Trade]:
        """
        Standard CDA matching algorithm.
        Incoming order is matched against sorted counterparties.
        """
        trades = []
        remaining_qty = incoming_order.remaining_kwh
        is_buy = incoming_order.order_type == OrderType.BUY

        for counter in counterparties:
            if remaining_qty <= 0.0001:
                break

            # ── Price Compatibility Guard ──
            # For a buyer: seller's price must be ≤ buyer's price
            # For a seller: buyer's price must be ≥ seller's price
            if is_buy:
                if counter.price_per_kwh > incoming_order.price_per_kwh:
                    break
            else:
                if counter.price_per_kwh < incoming_order.price_per_kwh:
                    break

            # ── Determine Traded Quantity & Price ──
            trade_qty   = round(min(remaining_qty, counter.remaining_kwh), 4)
            clearing_price = self.compute_clearing_price(
                incoming_order.price_per_kwh if is_buy else counter.price_per_kwh,
                counter.price_per_kwh if is_buy else incoming_order.price_per_kwh
            )

            # Assign roles
            buyer_id  = incoming_order.node_id if is_buy else counter.node_id
            seller_id = counter.node_id if is_buy else incoming_order.node_id
            buyer_order_id  = incoming_order.id if is_buy else counter.id
            seller_order_id = counter.id if is_buy else incoming_order.id

            trade = Trade(
                buyer_node_id   = buyer_id,
                seller_node_id  = seller_id,
                buyer_order_id  = buyer_order_id,
                seller_order_id = seller_order_id,
                quantity_kwh    = trade_qty,
                price_per_kwh   = clearing_price,
                total_cost      = round(trade_qty * clearing_price, 2),
                city            = incoming_order.city, # Derive from incoming order's city
                executed_at     = datetime.now(timezone.utc)
            )
            trades.append(trade)

            # ── Update Remaining Quantities & Statuses ──
            remaining_qty           = round(remaining_qty - trade_qty, 4)
            counter.remaining_kwh   = round(counter.remaining_kwh - trade_qty, 4)
            
            # Update incoming order
            incoming_order.remaining_kwh = remaining_qty
            self._update_order_status(incoming_order)
            
            # Update counterparty order
            self._update_order_status(counter)

        return trades

    def _update_order_status(self, order: Order):
        """Internal helper to transition order lifecycle based on remaining qty."""
        if order.remaining_kwh <= 0.0001:
            order.remaining_kwh = 0.0
            order.status = OrderStatus.FILLED
        elif order.remaining_kwh < order.quantity_kwh:
            order.status = OrderStatus.PARTIALLY_FILLED
        else:
            order.status = OrderStatus.PENDING
