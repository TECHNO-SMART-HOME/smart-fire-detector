# fire_alert_system.py
from pymongo import MongoClient
from datetime import datetime, timedelta

# --- CONFIGURATION ---
DB_PASSWORD = "0dHFNOAv4oAHtJuP"
DB_URI = f"mongodb+srv://tagupajameschristopher_db_user:{DB_PASSWORD}@smartdetection.nm3xrfn.mongodb.net/?appName=SmartDetection"
DB_NAME = "SmartDetectionDB"
COLLECTION_NAME = "sensor_readings"

# --- STATE MANAGEMENT ---
_client = None
_collection = None
_last_alert_time = datetime.min

def _connect():
    """Internal function to establish connection only when needed."""
    global _client, _collection
    if _collection is None:
        try:
            _client = MongoClient(DB_URI)
            db = _client[DB_NAME]
            _collection = db[COLLECTION_NAME]
            print("✅ [DB System] Connected to Cloud Database")
        except Exception as e:
            print(f"❌ [DB System] Connection Failed: {e}")

def send_alert(confidence_score):
    """
    Called by the camera loop. 
    Checks if 5 seconds have passed since the last alert. 
    If yes, sends data.
    """
    global _last_alert_time, _collection

    # 1. Check Cooldown
    now = datetime.now()
    if (now - _last_alert_time) < timedelta(seconds=5):
        return

    # 2. Connect if not connected
    if _collection is None:
        _connect()

    # 3. Prepare Payload
    data_packet = {
        "source": "Camera 01",
        "status": "CRITICAL",
        "value": int(confidence_score * 100),
        "timestamp": now
    }

    # 4. Send
    try:
        # FIXED: Explicit check against None
        if _collection is not None:
            _collection.insert_one(data_packet)
            print(f"🚀 [DB System] ALERT SENT! Fire Confidence: {int(confidence_score * 100)}%")
            _last_alert_time = now
    except Exception as e:
        print(f"⚠️ [DB System] Upload Error: {e}")