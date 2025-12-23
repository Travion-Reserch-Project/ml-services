"""
Data Repository abstraction for transport service.
Supports both CSV (dev) and MongoDB (production).
"""

import os
import pandas as pd
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
from pathlib import Path


class DataRepository(ABC):
    """Abstract base class for data access."""

    @abstractmethod
    def get_nodes(self) -> pd.DataFrame:
        """Fetch all location nodes."""
        pass

    @abstractmethod
    def get_services(self) -> pd.DataFrame:
        """Fetch all transport services."""
        pass

    @abstractmethod
    def get_edges(self) -> pd.DataFrame:
        """Fetch all routes (edges between nodes and services)."""
        pass

    @abstractmethod
    def get_edges_between(self, origin_id: int, dest_id: int) -> pd.DataFrame:
        """Fetch edges (routes) between two locations."""
        pass

    @abstractmethod
    def get_data_version(self) -> str:
        """Get current data snapshot version for tracking."""
        pass


class CSVRepository(DataRepository):
    """CSV-based repository (for development)."""

    def __init__(self, data_path: str):
        """Initialize CSV repository."""
        self.data_path = self._resolve_path(data_path)

        # Load once at init
        self._nodes_df = pd.read_csv(os.path.join(self.data_path, "nodes.csv"))
        self._services_df = pd.read_csv(os.path.join(self.data_path, "services.csv"))
        self._edges_df = pd.read_csv(os.path.join(self.data_path, "edges.csv"))

        print(f"✅ CSV Repository loaded from {self.data_path}")

    def _resolve_path(self, data_path: str) -> str:
        """Resolve data directory from multiple candidates."""
        candidates = []

        env_path = os.getenv("DATA_PATH")
        if env_path:
            candidates.append(env_path)

        candidates.append(data_path)
        candidates.append(os.path.abspath(data_path))

        module_dir = os.path.dirname(os.path.abspath(__file__))
        candidates.append(os.path.normpath(os.path.join(module_dir, "..", "data")))

        candidates.append("/app/data")

        for p in candidates:
            try_nodes = os.path.join(p, "nodes.csv")
            if os.path.exists(try_nodes):
                return p

        raise FileNotFoundError(f"Could not locate data files in: {candidates}")

    def get_nodes(self) -> pd.DataFrame:
        return self._nodes_df.copy()

    def get_services(self) -> pd.DataFrame:
        return self._services_df.copy()

    def get_edges(self) -> pd.DataFrame:
        return self._edges_df.copy()

    def get_edges_between(self, origin_id: int, dest_id: int) -> pd.DataFrame:
        result = self._edges_df[
            (self._edges_df["origin_id"] == origin_id)
            & (self._edges_df["destination_id"] == dest_id)
        ]
        return result.copy()

    def get_data_version(self) -> str:
        """Return CSV version (timestamp of files)."""
        nodes_path = os.path.join(self.data_path, "nodes.csv")
        mtime = os.path.getmtime(nodes_path)
        from datetime import datetime

        return datetime.fromtimestamp(mtime).isoformat()


class MongoDBRepository(DataRepository):
    """MongoDB-based repository (for production)."""

    def __init__(
        self,
        mongo_uri: str = "mongodb://localhost:27017",
        db_name: str = "transport_service",
    ):
        """
        Initialize MongoDB repository.

        Args:
            mongo_uri: MongoDB connection string
            db_name: Database name
        """
        from pymongo import MongoClient

        self.mongo_uri = mongo_uri
        self.db_name = db_name

        try:
            self.client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
            # Test connection
            self.client.admin.command("ping")
            self.db = self.client[db_name]

            print(f"✅ MongoDB Repository connected to {mongo_uri}/{db_name}")

            # Cache data on init
            self._cache_data()

        except Exception as e:
            raise RuntimeError(f"Failed to connect to MongoDB: {e}")

    def _cache_data(self):
        """Load data into memory for fast access."""
        self._nodes_df = pd.DataFrame(list(self.db.nodes.find({}, {"_id": 0})))
        self._services_df = pd.DataFrame(list(self.db.services.find({}, {"_id": 0})))
        self._edges_df = pd.DataFrame(list(self.db.edges.find({}, {"_id": 0})))

        if self._nodes_df.empty or self._services_df.empty or self._edges_df.empty:
            raise ValueError(
                "MongoDB collections are empty. Run seed_mongo.py to populate data."
            )

    def get_nodes(self) -> pd.DataFrame:
        return self._nodes_df.copy()

    def get_services(self) -> pd.DataFrame:
        return self._services_df.copy()

    def get_edges(self) -> pd.DataFrame:
        return self._edges_df.copy()

    def get_edges_between(self, origin_id: int, dest_id: int) -> pd.DataFrame:
        result = self._edges_df[
            (self._edges_df["origin_id"] == origin_id)
            & (self._edges_df["destination_id"] == dest_id)
        ]
        return result.copy()

    def get_data_version(self) -> str:
        """Get data version from metadata collection."""
        metadata = self.db.metadata.find_one({"type": "data_version"})
        if metadata:
            return metadata.get("version", "unknown")
        return "unknown"

    def close(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()


def create_repository(use_mongo: bool = False) -> DataRepository:
    """
    Factory function to create the appropriate repository.

    Args:
        use_mongo: If True, use MongoDB; otherwise use CSV

    Returns:
        DataRepository instance
    """
    if use_mongo:
        mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        db_name = os.getenv("MONGODB_DB_NAME", "transport_service")
        try:
            return MongoDBRepository(mongo_uri, db_name)
        except Exception as e:
            print(f"⚠️  MongoDB init failed ({e}), falling back to CSV")
            return CSVRepository("data")
    else:
        return CSVRepository("data")
