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
import bcrypt
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
        import certifi
        client = MongoClient(
            uri,
            server_api=ServerApi("1"),
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
            socketTimeoutMS=10000,
            tlsCAFile=certifi.where(),
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


def _get_users_collection():
    """Get the users collection, or None if unavailable."""
    conn = _get_mongo_connection()
    if not conn["connected"]:
        return None

    try:
        mongo_cfg = st.secrets["mongo"]
        db_name = mongo_cfg.get("db_name", "churnshield")
        # Ensure we have a distinct collection for users
        collection_name = mongo_cfg.get("users_collection_name", "users")
        return conn["client"][db_name][collection_name]
    except (KeyError, FileNotFoundError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API: AUTHENTICATION
# ─────────────────────────────────────────────────────────────────────────────

def create_user(username: str, password: str) -> Dict[str, Any]:
    """
    Create a new user with a hashed password.
    Returns dict with 'success' and 'message'.
    """
    collection = _get_users_collection()
    if collection is None:
        return {"success": False, "message": "Database not available"}

    username = username.strip().lower()
    
    # Check if exists
    if collection.find_one({"username": username}):
        return {"success": False, "message": "Username already exists"}

    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    
    try:
        collection.insert_one({
            "username": username,
            "password": hashed_password,
            "created_at": datetime.datetime.utcnow()
        })
        return {"success": True, "message": "User registered successfully"}
    except Exception as e:
        return {"success": False, "message": f"Registration failed: {str(e)}"}


def authenticate_user(username: str, password: str) -> Dict[str, Any]:
    """
    Verify user credentials.
    Returns dict with 'success', 'message', and 'username' if successful.
    """
    collection = _get_users_collection()
    if collection is None:
        return {"success": False, "message": "Database not available"}

    username = username.strip().lower()
    user = collection.find_one({"username": username})
    
    if not user:
        return {"success": False, "message": "Invalid username or password"}

    if bcrypt.checkpw(password.encode('utf-8'), user["password"]):
        return {"success": True, "message": "Login successful", "username": username}
    else:
        return {"success": False, "message": "Invalid username or password"}


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API: PREDICTIONS

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


def save_prediction(record: Dict[str, Any], username: Optional[str] = None) -> bool:
    """
    Save a prediction record to MongoDB.

    Args:
        record: Dictionary with prediction data (customer features,
                churn probability, risk score, etc.)
        username: The user making the prediction

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
        if username:
            doc["username"] = username

        collection.insert_one(doc)
        return True
    except Exception:
        return False


def get_prediction_history(limit: int = 100, username: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Fetch recent prediction history from MongoDB.

    Args:
        limit: Maximum number of records to return.
        username: Filter by username

    Returns:
        List of prediction records (newest first), or empty list.
    """
    collection = _get_collection()
    if collection is None:
        return []

    try:
        query = {}
        if username:
            query["username"] = username
            
        cursor = (
            collection.find(query, {"_id": 0})
            .sort("created_at", -1)
            .limit(limit)
        )
        # Reverse so oldest is first (matches the session_state order)
        records = list(cursor)
        records.reverse()
        return records
    except Exception:
        return []


def get_stats(username: Optional[str] = None) -> Dict[str, int]:
    """
    Get aggregate prediction statistics from MongoDB.

    Returns:
        Dict with 'total_predictions' and 'churns_predicted'.
    """
    collection = _get_collection()
    if collection is None:
        return {"total_predictions": 0, "churns_predicted": 0}

    try:
        query = {}
        if username:
            query["username"] = username
            
        total = collection.count_documents(query)
        
        churn_query = {"prediction": "Churn"}
        if username:
            churn_query["username"] = username
            
        churns = collection.count_documents(churn_query)
        return {
            "total_predictions": total,
            "churns_predicted": churns,
        }
    except Exception:
        return {"total_predictions": 0, "churns_predicted": 0}


def clear_history(username: Optional[str] = None) -> bool:
    """
    Delete prediction records from MongoDB.

    Returns:
        True if cleared successfully, False otherwise.
    """
    collection = _get_collection()
    if collection is None:
        return False

    try:
        query = {}
        if username:
            query["username"] = username
            
        collection.delete_many(query)
        return True
    except Exception:
        return False
