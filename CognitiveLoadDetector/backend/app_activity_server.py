import time
import psutil
from flask import Flask, jsonify
from flask_cors import CORS

# Windows-specific imports
import win32gui
import win32process

"""
app_activity_server.py
----------------------
Small local helper that detects the currently active (foreground) window
on Windows and exposes it via:

    GET http://127.0.0.1:5001/current_app
       -> {"app_name": "...", "process_name": "..."}
"""

app = Flask(__name__)
CORS(app)

last_app_name = "Unknown"
last_process_name = "Unknown"


def get_foreground_app():
    """Return (app_name, process_name) of the active window on Windows."""
    try:
        hwnd = win32gui.GetForegroundWindow()
        if not hwnd:
            return "Unknown", "Unknown"

        # Get process ID for the window
        tid, pid = win32process.GetWindowThreadProcessId(hwnd)
        process = psutil.Process(pid)
        process_name = process.name()

        # Window title as app_name
        window_title = win32gui.GetWindowText(hwnd)
        app_name = window_title if window_title else process_name

        # Optional: shorten long titles
        if len(app_name) > 40:
            app_name = app_name[:37] + "..."

        return app_name, process_name
    except Exception as e:
        print("Error in get_foreground_app:", e)
        return "Unknown", "Unknown"


@app.route("/current_app", methods=["GET"])
def current_app_route():
    global last_app_name, last_process_name
    app_name, process_name = get_foreground_app()
    last_app_name, last_process_name = app_name, process_name
    return jsonify({
        "app_name": app_name,
        "process_name": process_name
    })


if __name__ == "__main__":
    print("[AppActivity] Starting on http://127.0.0.1:5001")
    print("Press Ctrl+C to stop.")
    app.run(host="127.0.0.1", port=5001, debug=False)
