#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

if __name__ == "__main__":
    app_path = Path(__file__).parent.parent / "dashboard" / "app.py"

    print("=" * 60)
    print("STARTING STREAMLIT DASHBOARD")
    print("=" * 60)

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(app_path),
                "--server.port",
                "8501",
                "--server.headless",
                "true",
            ],
            check=True,
        )
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
