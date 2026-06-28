"""
db.py - MongoDB Database Layer for Prediction History
======================================================
Author: Data Science Team
Project: Customer Churn Prediction
Description: Handles all MongoDB operations for persisting prediction
             history. Gracefully falls back to session-state-only mode
             if MongoDB is not configured or unreachable.

Usage:
    from db import (
        save_prediction, get_prediction_history, get_stats,
        clear_history, is_mongo_available
    )
"""

import datetime
from typing import Any, Dict, List, Optional

import streamlit as st

try:
    from pymongo.mongo_client import MongoClient
    from pymongo.server_api import ServerApi
    from pymongo.errors import (
        ConnectionFailure,
        ServerSelectionTimeoutError,
        OperationFailure,
    )
    _PYMONGO_INSTALLED = True
except ImportError:
    _PYMONGO_INSTALLED = False


@st.cache_resource(show_spinner=False)
def _get_mongo_connection() -> Dict[str, Any]:
    """
    Create and cache a MongoDB connection using Streamlit secrets.

    Returns a dict with:
        - 'client': MongoClient or None
        - 'connected': bool
        - 'error': str or None
    """
    result = {"client": None, "connected": False, "error": None}

    if not _PYMONGO_INSTALLED:
        result["error"] = "pymongo is not installed"
        return result

    try:
        mongo_cfg = st.secrets["mongo"]
        uri = mongo_cfg["uri"]
    except (KeyError, FileNotFoundError):
        result["error"] = "MongoDB secrets not configured"
        return result

    try:
        client = MongoClient(
            uri,
            server_api=ServerApi("1"),
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
            socketTimeoutMS=10000,
        )
        # Verify connection
        client.admin.command("ping")
        result["client"] = client
        result["connected"] = True
        return result
    except (ConnectionFailure, ServerSelectionTimeoutError, Exception) as e:
        result["error"] = f"MongoDB connection failed: {e}"
        return result


def _get_collection():
    """Get the predictions collection, or None if unavailable."""
    conn = _get_mongo_connection()
    if not conn["connected"]:
        return None

    try:
        mongo_cfg = st.secrets["mongo"]
        db_name = mongo_cfg.get("db_name", "churnshield")
        collection_name = mongo_cfg.get("collection_name", "predictions")
        return conn["client"][db_name][collection_name]
    except (KeyError, FileNotFoundError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def is_mongo_available() -> bool:
    """Check if MongoDB is connected and available."""
    conn = _get_mongo_connection()
    return conn["connected"]


def get_connection_status() -> Dict[str, Any]:
    """
    Get detailed MongoDB connection status.

    Returns:
        Dict with 'connected' (bool), 'message' (str), and 'mode' (str).
    """
    conn = _get_mongo_connection()
    if conn["connected"]:
        return {
            "connected": True,
            "message": "Connected to MongoDB Atlas",
            "mode": "persistent",
        }
    else:
        return {
            "connected": False,
            "message": conn["error"] or "MongoDB not available",
            "mode": "session-only",
        }


def save_prediction(record: Dict[str, Any]) -> bool:
    """
    Save a prediction record to MongoDB.

    Args:
        record: Dictionary with prediction data (customer features,
                churn probability, risk score, etc.)

    Returns:
        True if saved successfully, False otherwise.
    """
    collection = _get_collection()
    if collection is None:
        return False

    try:
        # Add a proper datetime for sorting/querying
        doc = record.copy()
        doc["created_at"] = datetime.datetime.utcnow()

        collection.insert_one(doc)
        return True
    except Exception:
        return False


def get_prediction_history(limit: int = 100) -> List[Dict[str, Any]]:
    """
    Fetch recent prediction history from MongoDB.

    Args:
        limit: Maximum number of records to return.

    Returns:
        List of prediction records (newest first), or empty list.
    """
    collection = _get_collection()
    if collection is None:
        return []

    try:
        cursor = (
            collection.find({}, {"_id": 0})
            .sort("created_at", -1)
            .limit(limit)
        )
        # Reverse so oldest is first (matches the session_state order)
        records = list(cursor)
        records.reverse()
        return records
    except Exception:
        return []


def get_stats() -> Dict[str, int]:
    """
    Get aggregate prediction statistics from MongoDB.

    Returns:
        Dict with 'total_predictions' and 'churns_predicted'.
    """
    collection = _get_collection()
    if collection is None:
        return {"total_predictions": 0, "churns_predicted": 0}

    try:
        total = collection.count_documents({})
        churns = collection.count_documents({"prediction": "Churn"})
        return {
            "total_predictions": total,
            "churns_predicted": churns,
        }
    except Exception:
        return {"total_predictions": 0, "churns_predicted": 0}


def clear_history() -> bool:
    """
    Delete all prediction records from MongoDB.

    Returns:
        True if cleared successfully, False otherwise.
    """
    collection = _get_collection()
    if collection is None:
        return False

    try:
        collection.delete_many({})
        return True
    except Exception:
        return False
