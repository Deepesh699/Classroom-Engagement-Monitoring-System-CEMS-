from database import save_live_results

live_results = [
    {
        "track_id": 1,
        "score": 78,
        "status": "Engaged",
        "orientation": "Forward"
    }
]

session_id = 1

saved = save_live_results(
    session_id,
    live_results
)

print("Records saved:", saved)