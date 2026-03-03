#!/usr/bin/env python3
"""
Seed MongoDB with transport data from CSV files.
Run this once to populate MongoDB, then use repository for access.
"""

import os
import sys
import pandas as pd
from datetime import datetime
from pathlib import Path


def seed_mongodb(
    csv_path: str = "data",
    mongo_uri: str = "mongodb://localhost:27017",
    db_name: str = "transport_service",
):
    """
    Load CSV data into MongoDB.

    Args:
        csv_path: Path to data directory with CSV files
        mongo_uri: MongoDB connection string
        db_name: Database name
    """
    try:
        from pymongo import MongoClient

        # Resolve CSV path
        if not os.path.exists(csv_path):
            csv_path = os.path.join(os.path.dirname(__file__), "..", "data")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Data directory not found: {csv_path}")

        print(f"📂 Reading CSVs from {csv_path}")

        # Load CSVs
        nodes_df = pd.read_csv(os.path.join(csv_path, "nodes.csv"))
        services_df = pd.read_csv(os.path.join(csv_path, "services.csv"))
        edges_df = pd.read_csv(os.path.join(csv_path, "edges.csv"))

        print(
            f"📊 Loaded: {len(nodes_df)} nodes, {len(services_df)} services, {len(edges_df)} edges"
        )

        # Connect to MongoDB
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        db = client[db_name]

        print(f"✅ Connected to MongoDB: {mongo_uri}/{db_name}")

        # Clear existing collections
        db.nodes.delete_many({})
        db.services.delete_many({})
        db.edges.delete_many({})
        print("🗑️  Cleared existing collections")

        # Insert nodes
        nodes_records = nodes_df.to_dict("records")
        result = db.nodes.insert_many(nodes_records)
        print(f"✅ Inserted {len(result.inserted_ids)} nodes")

        # Insert services
        services_records = services_df.to_dict("records")
        result = db.services.insert_many(services_records)
        print(f"✅ Inserted {len(result.inserted_ids)} services")

        # Insert edges
        edges_records = edges_df.to_dict("records")
        result = db.edges.insert_many(edges_records)
        print(f"✅ Inserted {len(result.inserted_ids)} edges")

        # Create indices for faster queries
        db.nodes.create_index("location_id", unique=True)
        db.services.create_index("service_id", unique=True)
        db.edges.create_index([("origin_id", 1), ("destination_id", 1)])
        print("✅ Created database indices")

        # Store metadata (version)
        version = datetime.utcnow().isoformat()
        db.metadata.update_one(
            {"type": "data_version"},
            {"$set": {"type": "data_version", "version": version, "timestamp": datetime.utcnow()}},
            upsert=True,
        )
        print(f"✅ Stored data version: {version}")

        client.close()

        print("\n🎉 MongoDB seeding complete!")
        print(
            "\n📌 Next steps:"
        )
        print("   1. Set env vars in your container:")
        print("      - MONGODB_URI=mongodb://localhost:27017")
        print("      - MONGODB_DB_NAME=transport_service")
        print("   2. Set DATA_SOURCE=mongodb in routes.py or env")
        print("   3. Restart the service")

        return True

    except Exception as e:
        print(f"❌ Error seeding MongoDB: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    db_name = os.getenv("MONGODB_DB_NAME", "transport_service")

    print("=" * 60)
    print("🌱 MongoDB Data Seeding Script")
    print("=" * 60)
    print(f"MongoDB URI: {mongo_uri}")
    print(f"Database: {db_name}")
    print()

    success = seed_mongodb(mongo_uri=mongo_uri, db_name=db_name)
    sys.exit(0 if success else 1)
