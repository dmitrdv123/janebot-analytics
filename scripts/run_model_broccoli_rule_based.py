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
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Force SelectorEventLoop on Windows
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Настройки
SYMBOL = 'BTCUSDT'      # Торговый символ (USDT-margined perpetual futures)
INTERVAL = '1'          # 1-минутный интервал
RSI_PERIOD = 60         # Период для RSI (60 свечей = 1 час при 1-минутном интервале)
MARKET_TYPE = 'linear'  # Perpetual futures
LEVERAGE = 1            # Default leverage

class BybitLinearManager:
    def __init__(self, bybitManager):
        self.bybitManager = bybitManager

    async def short_open(self, quantity):
        try:
            order_id = await self.bybitManager.linear_sell(quantity)
            await self._wait_or_cancel(order_id)
            return await self.bybitManager.linear_status()
        except Exception as e:
            await self.bybitManager.order_cancel_all()
            raise

    async def short_close(self, quantity):
        try:
            order_id = await self.bybitManager.linear_buy(quantity)
            await self._wait_or_cancel(order_id)
            return await self.bybitManager.linear_status()
        except Exception as e:
            await self.bybitManager.order_cancel_all()
            raise

    async def short_status(self):
        return await self.bybitManager.linear_status()

    async def _wait_or_cancel(self, order_id, timeout=60, poll_interval=1):
        """Wait until the order is executed (closed) or canceled."""
        start_time = asyncio.get_event_loop().time()
        while True:
            status = await self.bybitManager.order_status(order_id)
            logger.debug(f"[BybitLinearManager] Order {order_id} status {status}")

            if status == 'closed':
                logger.info(f"[BybitLinearManager] Order {order_id} executed successfully")
                return True
            elif status == 'canceled':
                logger.info(f"[BybitLinearManager] Order {order_id} was canceled")
                return False
            elif status in ['rejected', 'expired']:
                logger.error(f"[BybitLinearManager] Order {order_id} failed with status {status}")
                return False

            # Check timeout
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= timeout:
                logger.warning(f"[BybitLinearManager] Timeout waiting for order {order_id} to complete after {timeout} seconds")
                await self.bybitManager.order_cancel(order_id)
                return False

            # Wait before next poll
            await asyncio.sleep(poll_interval)

class BybitManager:
    def __init__(self, exchange):
        self.exchange = exchange

        self.symbol = SYMBOL
        self.market_type = MARKET_TYPE

    async def linear_sell(self, quantity):
        """Create linear sell order with the specified quantity."""
        try:
            # Place a market sell order to open short
            order = await self.exchange.create_order(
                symbol=self.symbol,
                type='market',
                side='sell',
                amount=quantity,
                params={
                    'category': self.market_type,
                    'positionIdx': 0  # 0 for one-way mode
                }
            )
            order_id = order['id']
            logger.info(f"[BybitManager] End to create linear sell order {order_id} for {quantity} {self.symbol}")
            return order_id
        except Exception as e:
            logger.error(f"[BybitManager] Failed to create linear sell order for {quantity} {self.symbol}: {e}")
            raise
        finally:
            await self.exchange.close()

    async def linear_buy(self, quantity):
        """Create linear buy order with the specified quantity."""
        try:
            # Place a market buy order to close short
            order = await self.exchange.create_order(
                symbol=self.symbol,
                type='market',
                side='buy',
                amount=quantity,
                params={
                    'category': self.market_type,
                    'positionIdx': 0,
                    'reduceOnly': True  # Ensures the order reduces the position
                }
            )
            order_id = order['id']
            logger.info(f"[BybitManager] End to create linear buy order {order_id} for {quantity} {self.symbol}")
            return order_id
        except Exception as e:
            logger.error(f"[BybitManager] Failed to create linear buy order for {quantity} {self.symbol}: {e}")
            raise
        finally:
            await self.exchange.close()

    async def linear_status(self):
        """Retrieve information about the current position for the symbol."""
        try:
            position = await self.exchange.fetch_position(
                self.symbol,
                params={'category': self.market_type}
            )
            logger.info(f"[BybitManager] End to get linear status for {self.symbol}: {position}")
            return position
        except Exception as e:
            logger.error(f"[BybitManager] Failed to fetch linear status for {self.symbol}: {e}")
            raise
        finally:
            await self.exchange.close()

    async def order_status(self, order_id):
        """Get order status."""
        try:
            # Fetch cancelled or closed orders
            orders = await self.exchange.fetch_canceled_and_closed_orders(
                symbol=self.symbol,
                params={'orderId': order_id}
            )
            return orders[0]['status'] if len(orders) > 0 else ''
        except Exception as e:
            logger.error(f"[BybitManager] Failed to get order status for {order_id}: {e}")
            raise
        finally:
            await self.exchange.close()

    async def order_cancel(self, order_id):
        """Cancel order."""
        try:
            # Place a market sell order to open short
            await self.exchange.cancel_order(
                order_id,
                symbol=self.symbol
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

    async def order_cancel_all(self):
        """Cancel all orders."""
        try:
            # Cancel all orders for specific symbol
            await self.exchange.cancel_all_orders(
                symbol=self.symbol
            )
            logger.info(f"[BybitManager] All orders for {self.symbol} was cancelled")
            return True
        except Exception as e:
            logger.error(f"[BybitManager] Failed to cancel all orders for {self.symbol}: {e}")
            raise
        finally:
            await self.exchange.close()

    async def kline_load(self, interval, period):
        """Load KLines."""
        try:
            return await self.exchange.fetch_ohlcv(
                symbol=self.symbol,
                timeframe=interval + 'm',
                limit=period
            )
        except Exception as e:
            logger.error(f"[BybitManager] Failed to load KLines for {self.symbol}: {e}")
            raise
        finally:
            await self.exchange.close()

class BybitTrader:
    def __init__(self, bybit_manager, bybit_linear_manager):
        self.df = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        self.bybit_manager = bybit_manager
        self.bybit_linear_manager = bybit_linear_manager
        self.ws_url = "wss://stream.bybit.com/v5/public/spot"
        self.current_kline = None  # Store the latest kline data temporarily
        self.last_timestamp = None  # Track the timestamp of the last processed candlestick

    async def start(self):
        await self._init()
        self._calc_features()

        logger.info("[BybitTrader] Starting real-time updates")
        await self._websocket_connect()

    async def _init(self):
        klines = await self.bybit_manager.kline_load(INTERVAL, 2 * RSI_PERIOD)
        for kline in klines:
            self.df = pd.concat([self.df, pd.DataFrame({
                'timestamp': [kline[0]],
                'open': [float(kline[1])],
                'high': [float(kline[2])],
                'low': [float(kline[3])],
                'close': [float(kline[4])],
                'volume': [float(kline[5])],
                'turnover': [0.0]
            })], ignore_index=True)

        if not self.df.empty:
            self.last_timestamp = self.df['timestamp'].iloc[-1]

    def _calc_features(self):
        # Разница цены
        price_diff = self.df['close'].iloc[-1] - self.df['close'].iloc[-RSI_PERIOD]

        # Расчет RSI
        rsi = calculate_rsi(self.df, period=RSI_PERIOD, column='close').iloc[-1]

        # Вывод результатов
        logger.info(f"Price difference: {price_diff:.2f}, RSI: {rsi:.2f}")

    def _process_message(self, message):
        try:
            data = json.loads(message)
            if not ('data' in data and data['type'] == 'snapshot'):
                return

            kline = data['data'][0]
            timestamp = int(kline['start'])

            # Create a temporary DataFrame for the incoming kline
            new_kline = pd.DataFrame({
                'timestamp': [timestamp],
                'open': [float(kline['open'])],
                'high': [float(kline['high'])],
                'low': [float(kline['low'])],
                'close': [float(kline['close'])],
                'volume': [float(kline['volume'])],
                'turnover': [float(kline['turnover'])]
            })

            # If this is the first kline after historical data
            if self.current_kline is None:
                self.current_kline = new_kline
                logger.info(f"[BybitTrader] Received first kline for current interval: {datetime.fromtimestamp(timestamp/1000)}")
                return

            # Check if the timestamp has changed (new interval started)
            if timestamp != self.current_kline['timestamp'].iloc[0]:
                # The previous interval has ended; append the last stored kline as finalized
                logger.info(f"[BybitTrader] Interval ended, adding kline: {datetime.fromtimestamp(self.current_kline['timestamp'].iloc[0]/1000)}")
                self.df = pd.concat([self.df, self.current_kline], ignore_index=True)

                # Trim DataFrame to keep only RSI_PERIOD entries
                if len(self.df) > RSI_PERIOD:
                    self.df = self.df.iloc[-RSI_PERIOD:]

                # Update last_timestamp
                self.last_timestamp = self.current_kline['timestamp'].iloc[0]

                # Calculate indicators for the finalized candlestick
                self._calc_features()

                # Store the new kline as the current one
                self.current_kline = new_kline
                logger.info(f"[BybitTrader] Started new interval: {datetime.fromtimestamp(timestamp / 1000)}")
            else:
                # Same interval; update current_kline with the latest data
                self.current_kline = new_kline
                logger.debug(f"[BybitTrader] Updated current interval data: {datetime.fromtimestamp(timestamp / 1000)}")
        except Exception as e:
            logger.error(f"[BybitTrader] Failed to process: {e}")

    async def _websocket_connect(self):
        subscription = {
            "op": "subscribe",
            "args": [f"kline.{INTERVAL}.{SYMBOL}"]
        }
        
        max_retries = 10
        base_delay = 1  # Initial delay in seconds
        attempt = 0

        while attempt < max_retries:
            try:
                async with websockets.connect(self.ws_url, ping_interval=20, ping_timeout=10) as websocket:
                    attempt = 0  # Reset attempt counter on successful connection
                    logger.info("[BybitTrader] Connected to WebSocket")
                    await websocket.send(json.dumps(subscription))
                    logger.info(f"[BybitTrader] Subscribed to kline.{INTERVAL}.{SYMBOL}")

                    while True:
                        message = await websocket.recv()
                        self._process_message(message)

            except (websockets.exceptions.ConnectionClosed, asyncio.TimeoutError) as e:
                attempt += 1
                delay = base_delay * (2 ** (attempt - 1))  # Exponential backoff
                logger.warning(f"[BybitTrader] WebSocket connection failed (attempt {attempt}/{max_retries}): {e}")
                logger.info(f"[BybitTrader] Reconnecting in {delay:.2f} seconds...")
                await asyncio.sleep(delay)
            except Exception as e:
                logger.error(f"[BybitTrader] Unexpected error in WebSocket connection: {e}")
                attempt += 1
                delay = base_delay * (2 ** (attempt - 1))
                logger.info(f"[BybitTrader]Reconnecting in {delay:.2f} seconds...")
                await asyncio.sleep(delay)

        logger.error(f"[BybitTrader] Max retries ({max_retries}) reached. Giving up on WebSocket connection.")
        raise Exception("WebSocket connection failed after max retries")

async def main():
    api_key = os.getenv('BYBIT_API_KEY')
    api_secret = os.getenv('BYBIT_API_SECRET')
    demo_trading = os.getenv('BYBIT_DEMO', 'false').lower() == 'true'
    if not api_key or not api_secret:
        logger.error("API key and secret must be set in environment variables")
        return

    exchange = ccxt.bybit({
        'apiKey': api_key,
        'secret': api_secret,
        'enableRateLimit': True,  # Recommended to avoid hitting rate limits
    })
    if demo_trading:
        exchange.enable_demo_trading(demo_trading)

    # Initialize bybit managers
    bybit_manager = BybitManager(exchange)
    bybit_linear_manager = BybitLinearManager(bybit_manager)

    # Example usage of position manager with waiting
    # try:
    #     position = await bybit_linear_manager.short_status()
    #     logger.info(f"Position info: {position}")

    #     position = await bybit_linear_manager.short_open(0.001)
    #     logger.info(f"Position info: {position}")

    #     position = await bybit_linear_manager.short_close(0.001)
    #     logger.info(f"Position info: {position}")
    # except Exception as e:
    #     logger.error(f"Error in position management: {e}")

    # Start kline trader
    trader = BybitTrader(bybit_manager, bybit_linear_manager)
    await trader.start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Script terminated by user")
    except Exception as e:
        logger.error(f"Script terminated due to error: {e}")