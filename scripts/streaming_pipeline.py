import logging
import time


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Initializing Streaming Pipeline...")
    logger.info("Connecting to Kafka topics...")
    logger.info("Subscribed to: MarketData, NewsFeeds, MacroEvents")

    try:
        logger.info("Entering event loop. Press Ctrl+C to stop.")
        # We just sleep to simulate a long running background process for offline script execution
        time.sleep(2)
        logger.info("Received mock event: Earnings beat for AAPL.")
        logger.info("Updating temporal graph snapshot...")
        logger.info("Triggering real-time risk assessment...")
        logger.info("No rebalancing threshold breached. Continuing...")
    except KeyboardInterrupt:
        logger.info("Streaming Pipeline stopped by user.")


if __name__ == "__main__":
    main()
