import asyncio
import ccxt.async_support as ccxt
import json
import logging
import os
import pandas as pd
import sys
import functools

from datetime import datetime
from dotenv import load_dotenv

from utils_features import calculate_rsi

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Force SelectorEventLoop on Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Settings
MARKET_TYPE = "linear"  # Perpetual futures

TOKEN_PAIRS = {
    "BROCCOLIUSDT": {
        "precision_amount": 0,
        "min_amount": 300,
    },
    "TRUMPUSDT": {
        "precision_amount": 0,
        "min_amount": 1,
    },
    "MELANIAUSDT": {
        "precision_amount": 0,
        "min_amount": 15,
    },
    "FARTCOINUSDT": {
        "precision_amount": 0,
        "min_amount": 15,
    },
    "MUBARAKUSDT": {
        "precision_amount": 0,
        "min_amount": 200,
    },
    "BTCUSDT": {
        "precision_amount": 3,
        "min_amount": 0.001,
    }
}

# Global lock for position operations across all BybitTrader instances
position_lock = asyncio.Lock()

def retry(max_retries=3, delay=1, backoff=2, retry_on=(Exception,)):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            retries = 0
            current_delay = delay
            while retries < max_retries:
                try:
                    return await func(*args, **kwargs)
                except retry_on as e:
                    retries += 1
                    if retries >= max_retries:
                        logging.error(f"Failed {func.__name__} after {max_retries} retries: {e}")
                        raise
                    logging.warning(f"Retry {retries}/{max_retries} for {func.__name__}: {e}")
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
                except Exception as e:
                    logging.error(f"Unexpected error in {func.__name__}: {e}")
                    raise
        return wrapper
    return decorator

class BybitLinearManager:
    def __init__(self, bybit_manager):
        self.bybit_manager = bybit_manager

    async def short_status(self, symbol):
        position = await self.bybit_manager.linear_status(symbol)
        return position if position["contracts"] != 0 else None

    async def short_open(self, symbol, quantity):
        try:
            order_id = await self.bybit_manager.linear_sell(symbol, quantity)
            await self._wait_or_cancel(symbol, order_id)
            return await self.short_status(symbol)
        except Exception as e:
            await self.bybit_manager.order_cancel_all(symbol)
            raise

    async def short_close(self, symbol, quantity):
        try:
            order_id = await self.bybit_manager.linear_buy(symbol, quantity)
            await self._wait_or_cancel(symbol, order_id)
            return await self.short_status(symbol)
        except Exception as e:
            await self.bybit_manager.order_cancel_all(symbol)
            raise

    async def _wait_or_cancel(self, symbol, order_id, timeout=60, poll_interval=1):
        """Wait until the order is executed (closed) or canceled."""
        start_time = asyncio.get_event_loop().time()
        while True:
            status = await self.bybit_manager.order_status(symbol, order_id)
            logger.debug(f"[BybitLinearManager] Order {order_id} status {status}")

            if status == "closed":
                logger.info(f"[BybitLinearManager] Order {order_id} executed successfully")
                return True
            elif status == "canceled":
                logger.info(f"[BybitLinearManager] Order {order_id} was canceled")
                return False
            elif status in ["rejected", "expired"]:
                logger.error(f"[BybitLinearManager] Order {order_id} failed with status {status}")
                return False

            # Check timeout
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= timeout:
                logger.warning(f"[BybitLinearManager] Timeout waiting for order {order_id} to complete after {timeout} seconds")
                await self.bybit_manager.order_cancel(symbol, order_id)
                return False

            # Wait before next poll
            await asyncio.sleep(poll_interval)

class BybitManager:
    def __init__(self, exchange):
        self.exchange = exchange

    @retry(max_retries=3, delay=1, backoff=2, retry_on=(ccxt.NetworkError, ccxt.RateLimitExceeded))
    async def linear_sell(self, symbol, quantity):
        """Create linear sell order with the specified quantity."""
        try:
            # Place a market sell order to open short
            order = await self.exchange.create_order(
                symbol=symbol,
                type="market",
                side="sell",
                amount=quantity,
                params={
                    "category": MARKET_TYPE,
                    "positionIdx": 0  # 0 for one-way mode
                }
            )
            order_id = order["id"]
            logger.info(f"[BybitManager] End to create linear sell order {order_id} for {quantity} {symbol}")
            return order_id
        except Exception as e:
            logger.error(f"[BybitManager] Failed to create linear sell order for {quantity} {symbol}: {e}")
            raise

    @retry(max_retries=3, delay=1, backoff=2, retry_on=(ccxt.NetworkError, ccxt.RateLimitExceeded))
    async def linear_buy(self, symbol, quantity):
        """Create linear buy order with the specified quantity."""
        try:
            # Place a market buy order to close short
            order = await self.exchange.create_order(
                symbol=symbol,
                type="market",
                side="buy",
                amount=quantity,
                params={
                    "category": MARKET_TYPE,
                    "positionIdx": 0,
                    "reduceOnly": True  # Ensures the order reduces the position
                }
            )
            order_id = order["id"]
            logger.info(f"[BybitManager] End to create linear buy order {order_id} for {quantity} {symbol}")
            return order_id
        except Exception as e:
            logger.error(f"[BybitManager] Failed to create linear buy order for {quantity} {symbol}: {e}")
            raise

    @retry(max_retries=3, delay=1, backoff=2, retry_on=(ccxt.NetworkError, ccxt.RateLimitExceeded))
    async def linear_status(self, symbol):
        """Retrieve information about the current position for the symbol."""
        try:
            position = await self.exchange.fetch_position(
                symbol,
                params={"category": MARKET_TYPE}
            )
            logger.info(f"[BybitManager] End to get linear status for {symbol}: {position}")
            return position
        except Exception as e:
            logger.error(f"[BybitManager] Failed to fetch linear status for {symbol}: {e}")
            raise

    @retry(max_retries=3, delay=1, backoff=2, retry_on=(ccxt.NetworkError, ccxt.RateLimitExceeded))
    async def order_status(self, symbol, order_id):
        """Get order status."""
        try:
            # Fetch cancelled or closed orders
            orders = await self.exchange.fetch_canceled_and_closed_orders(
                symbol=symbol,
                params={"orderId": order_id}
            )
            return orders[0]["status"] if len(orders) > 0 else ""
        except Exception as e:
            logger.error(f"[BybitManager] Failed to get order status for {order_id}: {e}")
            raise

    @retry(max_retries=3, delay=1, backoff=2, retry_on=(ccxt.NetworkError, ccxt.RateLimitExceeded))
    async def order_cancel(self, symbol, order_id):
        """Cancel order."""
        try:
            # Place a market sell order to open short
            await self.exchange.cancel_order(
                order_id,
                symbol=symbol
            )
            logger.info(f"[BybitManager] Order {order_id} was cancelled")
            return True
        except Exception as e:
            if isinstance(e, ccxt.OrderNotFound):
                logger.warning(f"[BybitManager] Order {order_id} not found")
                return True

            logger.error(f"[BybitManager] Failed to cancel order {order_id}: {e}")
            raise

    @retry(max_retries=3, delay=1, backoff=2, retry_on=(ccxt.NetworkError, ccxt.RateLimitExceeded))
    async def order_cancel_all(self, symbol):
        """Cancel all orders."""
        try:
            # Cancel all orders for specific symbol
            await self.exchange.cancel_all_orders(
                symbol=symbol
            )
            logger.info(f"[BybitManager] All orders for {symbol} was cancelled")
            return True
        except Exception as e:
            logger.error(f"[BybitManager] Failed to cancel all orders for {symbol}: {e}")
            raise

    @retry(max_retries=3, delay=1, backoff=2, retry_on=(ccxt.NetworkError, ccxt.RateLimitExceeded))
    async def balance(self):
        """Fetch the available USDT balance for the Unified Account."""
        try:
            # Fetch the account balance
            balance = await self.exchange.fetch_balance(params={
                "category": MARKET_TYPE  # "linear" for USDT perpetuals
            })
            
            # Extract USDT balance
            usdt_balance = balance.get("USDT", {}).get("free", 0.0)
            logger.info(f"[BybitManager] Available USDT balance: {usdt_balance}")
            return usdt_balance
        except Exception as e:
            logger.error(f"[BybitManager] Failed to get balance: {e}")
            raise

    @retry(max_retries=3, delay=1, backoff=2, retry_on=(ccxt.NetworkError, ccxt.RateLimitExceeded))
    async def kline_load(self, symbol, interval, start, end):
        """Load KLines."""
        try:
            # Split the time range into hourly intervals
            hourly_intervals = []
            current_start = start
            while current_start < end:
                current_end = min(current_start + 3600 * 1000, end)  # 3600 * 1000 ms = 1 hour
                hourly_intervals.append((current_start, current_end))
                current_start = current_end

            # Fetch OHLCV data for each interval and concatenate results
            all_ohlcv = []
            for interval_start, interval_end in hourly_intervals:
                ohlcv = await self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=f"{interval}m",
                    since=interval_start,
                    params={"endTime": interval_end}
                )
                all_ohlcv.extend(ohlcv)

            logger.info(f"[BybitManager] End to load kline for {symbol} from {start} to {end}: {len(all_ohlcv)} records.")

            return all_ohlcv
        except Exception as e:
            logger.error(f"[BybitManager] Failed to load KLines for {symbol}: {e}")
            raise

class BybitTrader:
    def __init__(self, config, bybit_manager, bybit_linear_manager, balance_min_rest, position_lock=position_lock):
        self.config = config
        self.token_pair = TOKEN_PAIRS[config["symbol"]]
        self.balance_min_rest = balance_min_rest

        self.df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        self.df_features = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "close_diff", "rsi"])

        self.bybit_manager = bybit_manager
        self.bybit_linear_manager = bybit_linear_manager

        self.position = None
        self.position_created_timestamp = None
        self.position_updated_timestamp = None
        self.position_lock = position_lock

    async def init(self):
        await self.bybit_manager.order_cancel_all(self.config["symbol"])

        self.position = await self.bybit_linear_manager.short_status(self.config["symbol"])
        self.position_created_timestamp = self.position["lastUpdateTimestamp"] if self.position else None
        self.position_updated_timestamp = self.position_created_timestamp

        now = int(datetime.now().timestamp() * 1000)
        self.df = await self._load_data(now - 2 * self.config["max_duration"], now)
        self.df_features = self._calc_features(self.df)

    async def run(self):
        """Poll Bybit API for the last ended period's kline data, checking if already processed."""
        while True:
            # Sleep for 1 second
            await asyncio.sleep(1)

            try:
                last_complete_period_timestamp = self._last_complete_period_timestamp()
                last_timestamp = self.df["timestamp"].iloc[-1] if not self.df.empty else 0

                # Check for new closed period
                if last_complete_period_timestamp <= last_timestamp:
                    logger.debug(f"[BybitTrader] [{self.config['symbol']}] Last period {datetime.fromtimestamp(last_complete_period_timestamp / 1000)} already processed")
                    continue

                # Load and concatenate new data with existing dataframe, overwriting by timestamp
                now = int(datetime.now().timestamp() * 1000)
                df = await self._load_data(last_timestamp + 1, now)
                self.df = pd.concat([self.df, df]).drop_duplicates(subset=["timestamp"], keep="last").sort_values(by="timestamp").reset_index(drop=True)

                # Filter self.df by timestamp
                self.df = self.df[self.df["timestamp"] >= now - 2 * self.config["max_duration"]]

                # Calc features
                self.df_features = self._calc_features(self.df)

                await self._update_position()

                logger.info(f"[BybitTrader] [{self.config['symbol']}] End to process data for: {datetime.fromtimestamp(last_complete_period_timestamp / 1000)}")
            except Exception as e:
                logger.exception(f"[BybitTrader] [{self.config['symbol']}] Failed to process data: {e}")

    def _last_complete_period_timestamp(self):
        seconds_since_epoch = datetime.now().timestamp()

        interval_seconds = int(self.config["interval"]) * 60  # Interval in seconds (e.g., 300 for 5m)
        last_period_end = int(seconds_since_epoch // interval_seconds) * interval_seconds  # End of last period in seconds
        last_period_start = last_period_end - interval_seconds  # Start of last period in seconds
        last_period_start_ms = int(last_period_start * 1000)  # Convert to milliseconds

        return last_period_start_ms

    async def _load_data(self, start, end):
        try:
            klines = await self.bybit_manager.kline_load(self.config["symbol"], self.config["interval"], start, end)
            last_complete_period_timestamp = self._last_complete_period_timestamp()

            df_new = pd.DataFrame([
                {
                    "timestamp": int(kline[0]),
                    "open": float(kline[1]),
                    "high": float(kline[2]),
                    "low": float(kline[3]),
                    "close": float(kline[4]),
                    "volume": float(kline[5]),
                }
                for kline in klines if int(kline[0]) <= last_complete_period_timestamp
            ])

            if df_new.empty:
                logger.warning(f"[BybitTrader] [{self.config['symbol']}] No new klines fetched from {start} to {end}")
            elif df_new["timestamp"].min() < start:
                logger.warning(f"[BybitTrader] [{self.config['symbol']}] Fetched klines earlier than requested start time")
            return df_new
        except Exception as e:
            logger.error(f"[BybitTrader] [{self.config['symbol']}] Failed to load data: {e}")
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    def _calc_features(self, df):
        df_features = df.copy()

        df_features["close_diff"] = df_features["close"].pct_change(periods=self.config["period"])
        df_features["rsi"] = calculate_rsi(df_features, period=self.config["period"], column="close")
        df_features["zscore"] = (df_features["close"] - df_features["close"].rolling(self.config["period"]).mean()) / df_features["close"].rolling(self.config["period"]).std()

        return df_features

    async def _update_position(self):
        if self.df_features.empty:
            logger.warning(f"[BybitTrader] [{self.config['symbol']}] No features data available")
            return

        current_row = self.df_features.iloc[-1]
        price = current_row["close"]
        zscore = current_row["zscore"]
        rsi = current_row["rsi"]

        if not self.position:
            should_open = zscore > self.config["zscore_open_threshold"] and rsi > self.config["rsi_open_threshold"]
            if should_open:
                async with self.position_lock:
                    await self._open_position(price)
        else:
            # Refresh position status
            self.position = await self.bybit_linear_manager.short_status(self.config["symbol"])
            if not self.position:
                logger.warning(f"[BybitTrader] [{self.config['symbol']}] Position closed externally")
                self.position_created_timestamp = None
                self.position_updated_timestamp = None
                return

            current_timestamp = int(datetime.now().timestamp() * 1000)
            gross_profit = self.position.get("unrealizedPnl", 0)
            realized_pnl = float(self.position["info"].get("curRealisedPnl", 0))
            initial_margin = self.position.get("initialMargin", 0)
            current_qty = self.position.get("contracts", 0)

            profit = (gross_profit + realized_pnl) / initial_margin if initial_margin != 0 else 0
            duration = current_timestamp - self.position_created_timestamp if self.position_created_timestamp else 0

            should_close = (rsi < self.config["rsi_close_threshold"] and profit > self.config["profit_min"]) \
                or duration > self.config["max_duration"]
            if should_close:
                async with self.position_lock:
                    await self._close_position(current_qty)
                return

            # Find the closest record in self.df_features by self.position_updated_timestamp
            interval_ms = self.config["interval"] * 60 * 1000  # Convert interval to milliseconds
            closest_record = self.df_features[
                (self.df_features["timestamp"] > self.position_updated_timestamp - interval_ms) &
                (self.df_features["timestamp"] < self.position_updated_timestamp + interval_ms)
            ]
            closest_record = closest_record.iloc[-1] if not closest_record.empty else None
            price_diff_since_update = price - closest_record["close"] if closest_record is not None else 0

            should_increase = price_diff_since_update > self.config["close_diff_increase_threshold"] and rsi > self.config["rsi_increase_threshold"]
            if should_increase:
                async with self.position_lock:
                    await self._increase_position(price)

    async def _open_position(self, price):
        balance = await self.bybit_manager.balance()
        balance_token_amount = (balance - self.balance_min_rest) / price
        token_amount = min(self.config["amount_open"], balance_token_amount)

        qty = float(f"{token_amount:.{self.token_pair['precision_amount']}f}")
        if qty >= self.token_pair["min_amount"]:
            self.position = await self.bybit_linear_manager.short_open(self.config["symbol"], qty)
            self.position_created_timestamp = self.position["lastUpdateTimestamp"] if self.position else None
            self.position_updated_timestamp = self.position_created_timestamp

    async def _increase_position(self, price):
        balance = await self.bybit_manager.balance()
        balance_token_amount = (balance - self.balance_min_rest) / price
        token_amount = min(self.config["amount_increase"], balance_token_amount)

        qty = float(f"{token_amount:.{self.token_pair['precision_amount']}f}")
        if qty >= self.token_pair["min_amount"]:
            self.position = await self.bybit_linear_manager.short_open(self.config["symbol"], qty)
            self.self.position_updated_timestamp = self.position["lastUpdateTimestamp"] if self.position else None
            return

    async def _close_position(self, qty):
        self.position = await self.bybit_linear_manager.short_close(self.config["symbol"], qty)
        self.position_created_timestamp = None
        self.position_updated_timestamp = None

async def main():
    api_key = os.getenv("BYBIT_API_KEY")
    api_secret = os.getenv("BYBIT_API_SECRET")
    demo_trading = os.getenv("BYBIT_DEMO", "false").lower() == "true"
    balance_min_rest = float(os.getenv("BALANCE_MIN_REST"))
    config_file_path = os.getenv("CONFIG_FILE_PATH", ".config.json")

    # Load configuration from a JSON file
    try:
        with open(config_file_path, "r") as config_file:
            config = json.load(config_file)
            logger.info(f"Configuration loaded from {config_file_path}")
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {config_file_path}")
        return
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON configuration file: {e}")
        return

    exchange = ccxt.bybit({
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,  # Recommended to avoid hitting rate limits
    })
    if demo_trading:
        exchange.enable_demo_trading(demo_trading)

    # Initialize bybit managers
    bybit_manager = BybitManager(exchange)
    bybit_linear_manager = BybitLinearManager(bybit_manager)
    bybit_traders = [BybitTrader(cfg, bybit_manager, bybit_linear_manager, balance_min_rest) for cfg in config]

    # Example usage of position manager with waiting
    # try:
    #     balance = await bybit_manager.balance()
    #     logger.info(f"Balance: {balance}")

    #     position = await bybit_linear_manager.short_status()
    #     logger.info(f"Position info: {position}")

    #     position = await bybit_linear_manager.short_open(0.001)
    #     logger.info(f"Position info: {position}")

    #     position = await bybit_linear_manager.short_close(0.001)
    #     logger.info(f"Position info: {position}")
    # except Exception as e:
    #     logger.error(f"Error in position management: {e}")

    try:
        # Initialize traders
        for trader in bybit_traders:
            await trader.init()

        # Start polling for each trader concurrently
        await asyncio.gather(*(trader.run() for trader in bybit_traders))
    finally:
        await exchange.close()  # Ensure the exchange connection is closed

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Script terminated by user")
    except Exception as e:
        logger.error(f"Script terminated due to error: {e}")
