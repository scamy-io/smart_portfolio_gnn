import logging
import time

import schedule


class StreamingOrchestrator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def overnight_batch(self):
        self.logger.info("Running overnight batch: full graph reconstruction and model retrain")
        # In a real implementation, this would trigger airflow or a script
        # e.g., subprocess.run(["python", "scripts/train_model.py"])
        self.logger.info("Overnight batch completed.")

    def intraday_stream(self):
        self.logger.info("Running intraday stream: incremental edge updates and alerts")
        # In a real implementation, this polls the incremental updater
        self.logger.info("Intraday stream completed.")

    def run(self):
        # Formalize the overnight + intraday split
        schedule.every().day.at("02:00").do(self.overnight_batch)
        schedule.every(5).minutes.do(self.intraday_stream)

        self.logger.info("Starting Streaming Orchestrator...")
        
        # Example run loop (commented out to avoid blocking in tests)
        # while True:
        #     schedule.run_pending()
        #     time.sleep(1)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    orchestrator = StreamingOrchestrator()
    # Mocking a single run for demonstration
    orchestrator.intraday_stream()
    orchestrator.overnight_batch()
