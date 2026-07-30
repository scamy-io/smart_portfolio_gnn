import glob
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List


class MarketDataConsumer:
    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topics: List[str] = None,
        fallback_dir: Path = None,
    ):
        self.bootstrap_servers = bootstrap_servers
        self.topics = topics or ["prices", "news", "macro"]
        self.fallback_dir = fallback_dir or Path("data/streaming/incoming")
        self.use_kafka = False
        self.consumer = None

    def start(self):
        try:
            from kafka import KafkaConsumer

            self.consumer = KafkaConsumer(
                *self.topics,
                bootstrap_servers=self.bootstrap_servers,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                auto_offset_reset="latest",
                enable_auto_commit=True,
            )
            self.use_kafka = True
            print("Kafka connected successfully.")
        except ImportError:
            print("Warning: kafka-python not installed. Falling back to file polling.")
            self._init_fallback()
        except Exception as e:
            print(
                f"Warning: Kafka connection failed ({e}). Falling back to file polling."
            )
            self._init_fallback()

    def _init_fallback(self):
        self.use_kafka = False
        self.fallback_dir.mkdir(parents=True, exist_ok=True)
        (self.fallback_dir / "processed").mkdir(parents=True, exist_ok=True)

    def poll(self, timeout_ms: int = 1000) -> List[Dict]:
        messages = []
        if self.use_kafka:
            # Poll kafka
            raw_msgs = self.consumer.poll(timeout_ms=timeout_ms)
            for topic_partition, msgs in raw_msgs.items():
                for msg in msgs:
                    messages.append(msg.value)
        else:
            # Poll files
            files = glob.glob(str(self.fallback_dir / "*.json"))
            for file_path in files:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        messages.append(data)

                    # Move to processed
                    filename = os.path.basename(file_path)
                    shutil.move(
                        file_path, str(self.fallback_dir / "processed" / filename)
                    )
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

        return messages

    def stop(self):
        if self.use_kafka and self.consumer:
            self.consumer.close()
        print("Consumer stopped.")
