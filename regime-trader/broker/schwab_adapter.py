"""Adapts SchwabClient's Schwab-native response shapes onto
broker.base.BrokerClient's interface - i.e. makes Schwab look exactly like
AlpacaClient from the outside, field name for field name, so
broker/order_executor.py, broker/position_tracker.py, and main.py can hold
either broker without knowing which one they've got.

IMPORTANT CAVEAT: the field mappings below (which nested Schwab JSON field
maps to which normalized field) are my best understanding of Schwab's
public Trader API response schema, NOT independently verified against a
live response - this sandbox has no network access to Schwab's servers.
Sanity-check get_account()/list_positions() against a real paper-account
response before trusting this for real position sizing or P&L (print the
raw response - `self.client.get_account()` - the first few times you run
this for real and compare against what shows up in your actual Schwab
account).
"""

from __future__ import annotations

from datetime import datetime

from broker.schwab_client import SchwabClient
from schwab.orders.options import OptionSymbol

# Schwab's Order.Status values that map to each of this project's
# five-value order-status vocabulary (see broker/order_executor.py's
# OrderStatus) - everything not listed here (WORKING, QUEUED, ACCEPTED,
# NEW, the various AWAITING_*/PENDING_* states) is "pending".
_FILLED_STATUSES = {"FILLED"}
_CANCELED_STATUSES = {"CANCELED", "EXPIRED", "REPLACED"}
_REJECTED_STATUSES = {"REJECTED"}


class SchwabAdapter:
    """Wraps a SchwabClient to satisfy broker.base.BrokerClient.

    Args:
        client: A constructed SchwabClient (already authenticated).
    """

    def __init__(self, client: SchwabClient) -> None:
        self.client = client

    # ------------------------------------------------------------------
    # Already broker-agnostic in shape - straight passthroughs
    # ------------------------------------------------------------------

    def is_market_open(self) -> bool:
        return self.client.is_market_open()

    def get_bars(self, symbol: str, timeframe: str, start: str, end: str):
        return self.client.get_bars(symbol, timeframe, start, end)

    def get_latest_quote(self, symbol: str) -> dict:
        return self.client.get_latest_quote(symbol)

    def get_option_chain(
        self,
        underlying_symbol: str,
        contract_type: str,
        expiration_gte,
        expiration_lte,
        strike_gte: float | None = None,
        strike_lte: float | None = None,
    ) -> list[dict]:
        return self.client.get_option_chain(underlying_symbol, contract_type, expiration_gte, expiration_lte, strike_gte, strike_lte)

    def get_option_contract_details(self, occ_symbol: str) -> dict:
        """Alpaca-shaped contract metadata, parsed directly from the OCC/OSI
        symbol (no extra API call needed - the symbol itself fully encodes
        underlying/expiration/type/strike)."""
        parsed = OptionSymbol.parse_symbol(occ_symbol)
        return {
            "symbol": occ_symbol,
            "underlying_symbol": parsed.underlying_symbol,
            "type": "call" if parsed.contract_type == "C" else "put",
            "expiration_date": parsed.expiration_date.isoformat(),
            "strike_price": float(parsed.strike_price),
        }

    # ------------------------------------------------------------------
    # Account / positions - normalized to Alpaca's flat field names
    # ------------------------------------------------------------------

    def get_account(self) -> dict:
        """Normalizes Schwab's nested securitiesAccount.currentBalances (and
        .initialBalances, for "as of last close") into Alpaca's flat shape:
        {equity, cash, buying_power, last_equity}.

        Mapping used (per Schwab's public Trader API schema):
        - equity <- currentBalances.liquidationValue
        - cash <- currentBalances.cashBalance
        - buying_power <- currentBalances.buyingPower
        - last_equity <- initialBalances.liquidationValue (Schwab's
          "initial" balances are the start-of-day snapshot - i.e.
          effectively yesterday's close, the same concept Alpaca's
          last_equity captures for get_daily_pnl).
        """
        raw = self.client.get_account()
        account = raw.get("securitiesAccount", {})
        current = account.get("currentBalances", {})
        initial = account.get("initialBalances", {})
        return {
            "equity": float(current.get("liquidationValue", 0.0)),
            "cash": float(current.get("cashBalance", current.get("cashAvailableForTrading", 0.0)) or 0.0),
            "buying_power": float(current.get("buyingPower", 0.0)),
            "last_equity": float(initial.get("liquidationValue", current.get("liquidationValue", 0.0))),
            "status": "ACTIVE",
        }

    def list_positions(self) -> list[dict]:
        return [self._normalize_position(raw) for raw in self.client.list_positions()]

    def get_position(self, symbol: str) -> dict | None:
        raw = self.client.get_position(symbol)
        return self._normalize_position(raw) if raw is not None else None

    @staticmethod
    def _normalize_position(raw: dict) -> dict:
        """Alpaca-shaped position from Schwab's securitiesAccount.positions[]
        entry. `unrealized_pl`/`current_price` are derived (market_value -
        cost_basis, market_value / qty) rather than read from a single
        Schwab field, since Schwab's per-field P&L semantics
        (currentDayProfitLoss is *day* P&L, not since-open) don't line up
        with what Alpaca's unrealized_pl means - this computes the
        since-open figure directly instead of trusting a same-named field
        that likely means something different.
        """
        instrument = raw.get("instrument", {})
        long_qty = float(raw.get("longQuantity", 0.0) or 0.0)
        short_qty = float(raw.get("shortQuantity", 0.0) or 0.0)
        qty = long_qty - short_qty
        avg_price = float(raw.get("averageLongPrice") or raw.get("averageShortPrice") or raw.get("averagePrice") or 0.0)
        market_value = float(raw.get("marketValue", 0.0) or 0.0)
        current_price = (market_value / qty) if qty else avg_price
        unrealized_pl = market_value - (avg_price * qty)
        asset_type = str(instrument.get("assetType", "EQUITY")).upper()
        return {
            "symbol": instrument.get("symbol", ""),
            "qty": qty,
            "avg_entry_price": avg_price,
            "current_price": current_price,
            "unrealized_pl": unrealized_pl,
            "market_value": market_value,
            "asset_class": "us_option" if asset_type == "OPTION" else "us_equity",
        }

    # ------------------------------------------------------------------
    # Orders - normalized to Alpaca's flat field names + status vocabulary
    # ------------------------------------------------------------------

    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        asset_class: str = "equity",
        order_type: str = "market",
        time_in_force: str = "day",
        limit_price: float | None = None,
        stop_price: float | None = None,
        client_order_id: str | None = None,
    ) -> dict:
        """Places the order, then immediately fetches its full state (Schwab's
        place_order response only ever contains an order ID, never order
        details - unlike Alpaca, which returns the full order inline)."""
        result = self.client.submit_order(
            symbol, qty, side, asset_class=asset_class, order_type=order_type, time_in_force=time_in_force, limit_price=limit_price
        )
        order_id = result.get("order_id")
        if order_id is None:
            # Schwab accepted the request but didn't hand back a parseable
            # order ID (unusual) - report what we know rather than crash.
            return {"id": "", "symbol": symbol, "qty": qty, "side": side, "status": "pending"}
        try:
            return self.get_order(str(order_id))
        except Exception:
            # Order was accepted (we have an ID) - the follow-up lookup
            # failing shouldn't lose that ID, just leave status unresolved.
            return {"id": str(order_id), "symbol": symbol, "qty": qty, "side": side, "status": "pending"}

    def cancel_order(self, order_id: str) -> None:
        self.client.cancel_order(order_id)

    def replace_order(
        self,
        order_id: str,
        qty: float | None = None,
        limit_price: float | None = None,
        stop_price: float | None = None,
        time_in_force: str | None = None,
    ) -> dict:
        """Not implemented: Schwab's replace endpoint requires submitting a
        complete new order spec (not a partial field update like Alpaca's
        replace_order), which needs the original order's full details
        reconstructed first. Cancel and resubmit instead for now."""
        raise NotImplementedError("Schwab order modification not yet wired - cancel_order + submit_order instead")

    def get_order(self, order_id: str) -> dict:
        return self._normalize_order(self.client.get_order(order_id))

    def list_orders(self, status: str = "open", after: str | datetime | None = None) -> list[dict]:
        return [self._normalize_order(raw) for raw in self.client.list_orders(status, after)]

    @staticmethod
    def _normalize_order(raw: dict) -> dict:
        legs = raw.get("orderLegCollection") or [{}]
        leg = legs[0]
        instrument = leg.get("instrument", {})
        instruction = str(leg.get("instruction", "")).lower()
        side = "buy" if "buy" in instruction else "sell"

        raw_status = str(raw.get("status", "")).upper()
        requested_qty = float(raw.get("quantity", 0.0) or 0.0)
        filled_qty = float(raw.get("filledQuantity", 0.0) or 0.0)
        if raw_status in _FILLED_STATUSES:
            status = "filled"
        elif raw_status in _CANCELED_STATUSES:
            status = "canceled"
        elif raw_status in _REJECTED_STATUSES:
            status = "rejected"
        elif 0 < filled_qty < requested_qty:
            status = "partially_filled"
        else:
            status = "pending"

        return {
            "id": str(raw.get("orderId", "")),
            "symbol": instrument.get("symbol", ""),
            "qty": requested_qty,
            "side": side,
            "status": status,
        }
