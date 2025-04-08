import asyncio
import ccxt.async_support as ccxt
import json
import logging
import os
import pandas as pd
import sys
import websockets

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

# Настройки
WS_URL = "wss://stream.bybit.com/v5/public/spot"

MARKET_TYPE = "linear"  # Perpetual futures

TOKEN_PAIRS = {
    "BTCUSDT": {
        "decimals": 8,
        "precision_amount": 3,
        "min_amount": 0.001,
    }
}

# Global lock for position operations across all BybitTrader instances
position_lock = asyncio.Lock()

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
        finally:
            await self.exchange.close()

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
        finally:
            await self.exchange.close()

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
        finally:
            await self.exchange.close()

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
        finally:
            await self.exchange.close()

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
        finally:
            await self.exchange.close()

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
        finally:
            await self.exchange.close()

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
        finally:
            await self.exchange.close()

    async def kline_load(self, symbol, interval, period):
        """Load KLines."""
        try:
            return await self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=f"{interval}m",
                limit=period
            )
        except Exception as e:
            logger.error(f"[BybitManager] Failed to load KLines for {symbol}: {e}")
            raise
        finally:
            await self.exchange.close()

class BybitListener:
    def __init__(self, traders):
        """Initialize with a list of BybitTrader instances."""
        self.traders = traders

        # Map subscription topics to their respective traders
        self.subscription_map = {
            f"kline.{trader.get_config()['interval']}.{trader.get_config()['symbol']}": trader
            for trader in traders
        }

        # Map subscription topics to their message queues
        self.message_queues = {topic: asyncio.Queue() for topic in self.subscription_map.keys()}
        self.worker_tasks = {}  # Track worker tasks per topic

    async def _process_worker(self, topic, trader):
        """Worker coroutine to process messages sequentially for a given topic."""
        queue = self.message_queues[topic]
        while True:
            try:
                message = await queue.get()
                logger.debug(f"[BybitListener] Processing message for {topic}")
                await trader.process_message(message)
                queue.task_done()
                logger.debug(f"[BybitListener] Finished processing message for {topic}")
            except Exception as e:
                logger.error(f"[BybitListener] Error processing message for {topic}: {e}")

    async def listen(self):
        """Listen to WebSocket and route messages to queues for sequential processing."""
        subscriptions = [
            {"op": "subscribe", "args": [topic]}
            for topic in self.subscription_map.keys()
        ]

        logger.info(f"[BybitListener] Starting listen for subscriptions: {list(self.subscription_map.keys())}")

        # Start worker tasks for each topic
        for topic, trader in self.subscription_map.items():
            self.worker_tasks[topic] = asyncio.create_task(self._process_worker(topic, trader))
            logger.info(f"[BybitListener] Started worker for {topic}")

        max_retries = 10
        base_delay = 1  # Initial delay in seconds
        attempt = 0

        while attempt < max_retries:
            try:
                async with websockets.connect(WS_URL, ping_interval=20, ping_timeout=10) as websocket:
                    attempt = 0  # Reset attempt counter on successful connection
                    logger.info("[BybitListener] Connected to WebSocket")

                    # Send all subscriptions
                    for subscription in subscriptions:
                        await websocket.send(json.dumps(subscription))
                        logger.info(f"[BybitListener] Subscribed to {subscription['args'][0]}")

                    while True:
                        message = await websocket.recv()
                        self._enqueue_message(message)

            except (websockets.exceptions.ConnectionClosed, asyncio.TimeoutError) as e:
                attempt += 1
                delay = base_delay * (2 ** (attempt - 1))  # Exponential backoff
                logger.warning(f"[BybitListener] WebSocket connection failed (attempt {attempt}/{max_retries}): {e}")
                logger.info(f"[BybitListener] Reconnecting in {delay:.2f} seconds...")
                await asyncio.sleep(delay)
            except Exception as e:
                logger.error(f"[BybitListener] Unexpected error in WebSocket connection: {e}")
                attempt += 1
                delay = base_delay * (2 ** (attempt - 1))
                logger.info(f"[BybitListener] Reconnecting in {delay:.2f} seconds...")
                await asyncio.sleep(delay)

        logger.error(f"[BybitListener] Max retries ({max_retries}) reached. Giving up on WebSocket connection.")

        # Cancel all worker tasks on shutdown
        for task in self.worker_tasks.values():
            task.cancel()

        raise Exception("WebSocket connection failed after max retries")

    def _enqueue_message(self, message):
        """Add the WebSocket message to the appropriate topic's queue."""
        try:
            data = json.loads(message)
            if "topic" not in data:
                logger.debug(f"[BybitListener] Ignoring message without topic: {data}")
                return

            topic = data["topic"]
            if topic not in self.subscription_map:
                logger.warning(f"[BybitListener] No trader found for topic: {topic}")
                return

            queue = self.message_queues[topic]
            queue.put_nowait(message)
            logger.debug(f"[BybitListener] Enqueued message for {topic}, queue size: {queue.qsize()}")
        except Exception as e:
            logger.error(f"[BybitListener] Failed to enqueue message: {e}")

class BybitTrader:
    def __init__(self, config, bybit_manager, bybit_linear_manager, balance_min_rest, position_lock=position_lock):
        self.config = config
        self.token_pair = TOKEN_PAIRS[config["symbol"]]
        self.balance_min_rest = balance_min_rest

        self.df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
        self.df_features = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "turnover", "close_diff", "rsi"])

        self.bybit_manager = bybit_manager
        self.bybit_linear_manager = bybit_linear_manager

        self.current_kline = None  # Store the latest kline data temporarily
        self.last_timestamp = None  # Track the timestamp of the last processed kline

        self.position = None
        self.position_timestamp = None
        self.position_lock = position_lock

    async def init(self):
        await self.bybit_manager.order_cancel_all(self.config["symbol"])

        self.position = await self.bybit_linear_manager.short_status(self.config["symbol"])
        self.position_timestamp = self.position["timestamp"] if self.position else None

        klines = await self.bybit_manager.kline_load(self.config["symbol"], self.config["interval"], 2 * self.config["period"])
        for kline in klines:
            self.df = pd.concat([self.df, pd.DataFrame({
                "timestamp": [kline[0]],
                "open": [float(kline[1])],
                "high": [float(kline[2])],
                "low": [float(kline[3])],
                "close": [float(kline[4])],
                "volume": [float(kline[5])],
                "turnover": [0.0]
            })], ignore_index=True)

        if not self.df.empty:
            self.last_timestamp = self.df["timestamp"].iloc[-1]

        self._calc_features()

    def get_config(self):
        return self.config

    async def process_message(self, message):
        try:
            data = json.loads(message)
            if not ("data" in data and data["type"] == "snapshot"):
                return

            kline = data["data"][0]
            timestamp = int(kline["start"])

            # Create a temporary DataFrame for the incoming kline
            new_kline = pd.DataFrame({
                "timestamp": [timestamp],
                "open": [float(kline["open"])],
                "high": [float(kline["high"])],
                "low": [float(kline["low"])],
                "close": [float(kline["close"])],
                "volume": [float(kline["volume"])],
                "turnover": [float(kline["turnover"])]
            })

            # If this is the first kline after historical data
            if self.current_kline is None:
                self.current_kline = new_kline
                logger.info(f"[BybitTrader] Received first kline for current interval: {datetime.fromtimestamp(timestamp/1000)}")
                return

            # Check if the timestamp has changed (new interval started)
            if timestamp != self.current_kline["timestamp"].iloc[0]:
                # The previous interval has ended; append the last stored kline as finalized
                logger.info(f"[BybitTrader] Interval ended, adding kline: {datetime.fromtimestamp(self.current_kline['timestamp'].iloc[0]/1000)}")
                self.df = pd.concat([self.df, self.current_kline], ignore_index=True)

                # Trim DataFrame to keep only 2 RSI_PERIOD entries
                if len(self.df) > 2 * self.config["period"]:
                    self.df = self.df.iloc[-2 * self.config["period"]:]

                # Update last_timestamp
                self.last_timestamp = self.current_kline["timestamp"].iloc[0]

                # Store the new kline as the current one
                self.current_kline = new_kline

                # Calculate indicators for the finalized kline
                self._calc_features()

                # Open, close, increase position
                await self._update_position()

                logger.info(f"[BybitTrader] Started new interval: {datetime.fromtimestamp(timestamp / 1000)}")
            else:
                # Same interval; update current_kline with the latest data
                self.current_kline = new_kline
                logger.debug(f"[BybitTrader] Updated current interval data: {datetime.fromtimestamp(timestamp / 1000)}")
        except Exception as e:
            logger.error(f"[BybitTrader] Failed to process: {e}")

    def _calc_features(self):
        self.df_features = self.df.copy()

        self.df_features["close_diff"] = self.df_features["close"].pct_change(periods=self.config["period"])  # 1 hour = 12 periods
        self.df_features["rsi"] = calculate_rsi(self.df_features, period=self.config["period"], column="close")  # 1 hour = 12 periods

    async def _update_position(self):
        current_row = self.df_features.iloc[-1]
        price = current_row["close"]
        price_diff = current_row["close_diff"]
        rsi = current_row["rsi"]

        if not self.position:
            shouldOpen = price_diff > self.config["close_diff_open_threshold"] and rsi > self.config["rsi_open_threshold"]
            if shouldOpen:
                async with self.position_lock:
                    await self._open_position(price)
                return
        else:
            self.position = await self.bybit_linear_manager.short_status(self.config["symbol"])

            current_timestamp = int(datetime.now().timestamp() * 1000)

            gross_profit = self.position["unrealizedPnl"]
            realized_pnl = float(self.position["info"]["curRealisedPnl"])
            initial_margin = self.position["initialMargin"]
            current_qty = self.position["contracts"]

            profit = (gross_profit + realized_pnl) / initial_margin
            duration = current_timestamp - self.position_timestamp

            shouldClose = (rsi < self.config["rsi_close_threshold"] and profit > self.config["profit_min"]) \
                or duration > self.config["max_duration"]
            if shouldClose:
                async with self.position_lock:
                    await self._close_position(current_qty)
                return

            shouldIncrease = price_diff > self.config["close_diff_increase_threshold"] and rsi > self.config["rsi_increase_threshold"]
            if shouldIncrease:
                async with self.position_lock:
                    await self._increase_position(price)
                return

    async def _open_position(self, price):
        balance = await self.bybit_manager.balance()
        balance_token_amount = (balance - self.balance_min_rest) / price
        token_amount = min(self.config["amount_open"], balance_token_amount)

        qty = float(f"{token_amount:.{self.token_pair['precision_amount']}f}")
        if qty >= self.token_pair["min_amount"]:
            self.position = await self.bybit_linear_manager.short_open(self.config["symbol"], qty)
            self.position_timestamp = self.position["timestamp"] if self.position else None

    async def _increase_position(self, price):
        balance = await self.bybit_manager.balance()
        balance_token_amount = (balance - self.balance_min_rest) / price
        token_amount = min(self.config["amount_increase"], balance_token_amount)

        qty = float(f"{token_amount:.{self.token_pair['precision_amount']}f}")
        if qty >= self.token_pair["min_amount"]:
            self.position = await self.bybit_linear_manager.short_open(self.config["symbol"], qty)
            return

    async def _close_position(self, qty):
        self.position = await self.bybit_linear_manager.short_close(self.config["symbol"], qty)
        self.position_timestamp = None

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
    bybit_listener = BybitListener(bybit_traders)

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

    # Iterate over the configuration and create traders

    # Init traders
    await asyncio.gather(*(trader.init() for trader in bybit_traders))

    # Start listen
    await bybit_listener.listen()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Script terminated by user")
    except Exception as e:
        logger.error(f"Script terminated due to error: {e}")
