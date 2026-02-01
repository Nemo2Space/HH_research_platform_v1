"""
Test Kafka Connection

Run this script to verify Kafka/Redpanda connection works.

Usage:
    python scripts/test_kafka_connection.py
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv()

from src.kafka.kafka_connection import test_kafka_connection, KafkaConfig


def main():
    print("=" * 50)
    print("Alpha Platform - Kafka Connection Test")
    print("=" * 50)
    print()

    config = KafkaConfig.from_env()
    print(f"Bootstrap Servers: {config.bootstrap_servers}")
    print(f"Group ID:          {config.group_id}")
    print()

    try:
        result = test_kafka_connection(config)
        print()
        print("=" * 50)
        if result:
            print("SUCCESS - Kafka connection working!")
        else:
            print("PARTIAL - Connection made but issues detected")
        print("=" * 50)
    except Exception as e:
        print()
        print("=" * 50)
        print(f"FAILED - {e}")
        print("=" * 50)
        sys.exit(1)


if __name__ == "__main__":
    main()