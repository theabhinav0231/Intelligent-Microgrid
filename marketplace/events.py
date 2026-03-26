"""
events.py
=========
Observer pattern implementation to decouple event producers from consumers.
Enables real-time market feeds via MQTT and SSE.
"""

import json
from collections import defaultdict
from typing import Callable, Any, List
from datetime import datetime
import asyncio
import logging

from .models import Order, Trade

# Simple logger for events
logger = logging.getLogger("MarketEvents")

class EventBus:
    """
    Observer Pattern: Decouples market events from side-effects.
    One point of publication, multiple subscribers.
    """

    def __init__(self):
        self._subscribers: dict[str, list[Callable]] = defaultdict(list)

    def subscribe(self, event_name: str, handler: Callable[[Any], None]):
        """Registers a handler for a specific event type."""
        self._subscribers[event_name].append(handler)
        logger.debug(f"Subscribed handler {handler.__name__} to event '{event_name}'")

    def publish(self, event_name: str, data: Any):
        """Notifies all registered subscribers for an event."""
        logger.info(f"Published event '{event_name}': {type(data).__name__}")
        for handler in self._subscribers[event_name]:
            try:
                # Handle both sync and async handlers
                if asyncio.iscoroutinefunction(handler):
                    asyncio.create_task(handler(data))
                else:
                    handler(data)
            except Exception as e:
                logger.error(f"Handler {handler} failed for event '{event_name}': {e}")


class MQTTNotifier:
    """
    Subscribes to market events and publishes them to the MQTT broker.
    Integrates the marketplace with the orchestrator ecosystem.
    """

    def __init__(self, broker_host: str, broker_port: int, client_id: str = "Marketplace_EventBus"):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.client_id   = client_id
        self._client     = None

    def connect(self):
        """Initializes the MQTT client (lazily)."""
        import paho.mqtt.client as mqtt
        self._client = mqtt.Client(client_id=self.client_id)
        self._client.connect(self.broker_host, self.broker_port)
        self._client.loop_start()

    def on_trade_executed(self, trade: Trade):
        """Publishes trade details to MQTT."""
        if not self._client: return
        payload = json.dumps({
            "event":          "trade_executed",
            "buyer_node_id":  trade.buyer_node_id,
            "seller_node_id": trade.seller_node_id,
            "quantity_kwh":   trade.quantity_kwh,
            "price_per_kwh":  trade.price_per_kwh,
            "total_cost":     trade.total_cost,
            "executed_at":    trade.executed_at.isoformat(),
            "city":           trade.city
        })
        self._client.publish("marketplace/events/trade", payload, qos=1)
        # Also publish to node-specific settlement topics
        self._client.publish(f"microgrid/{trade.buyer_node_id}/settle", payload, qos=2)
        self._client.publish(f"microgrid/{trade.seller_node_id}/settle", payload, qos=2)

    def on_order_placed(self, order: Order):
        """Publishes new order activity to MQTT."""
        if not self._client: return
        payload = json.dumps({
            "event":         "order_placed",
            "node_id":       order.node_id,
            "order_type":    order.order_type.value,
            "quantity_kwh": order.quantity_kwh,
            "price_per_kwh": order.price_per_kwh,
            "city":          order.city
        })
        self._client.publish("marketplace/events/order", payload, qos=0)


class SSENotifier:
    """
    Buffers events for Server-Sent Events (SSE).
    Used by the dashboard to show real-time market ticker.
    """

    def __init__(self):
        self._queues: List[asyncio.Queue] = []

    def subscribe(self) -> asyncio.Queue:
        """Creates a new queue for a single SSE connection."""
        queue = asyncio.Queue()
        self._queues.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue):
        """Removes a queue when connection is closed."""
        self._queues.remove(queue)

    def on_market_event(self, data: Any):
        """Pushes data to all active SSE queues."""
        # Convert to serializable dict
        if hasattr(data, '__dict__'):
            # Basic serialization logic for models
            event_data = {k: v.isoformat() if isinstance(v, datetime) else v 
                        for k, v in data.__dict__.items() if not k.startswith('_')}
        else:
            event_data = data
            
        for queue in self._queues:
            queue.put_nowait(event_data)
