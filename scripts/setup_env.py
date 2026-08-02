"""Verify environment setup before running pipeline."""
import os
from dotenv import load_dotenv

load_dotenv()

def check_env():
    key = os.environ.get("FRED_API_KEY")
    offline = os.environ.get("SMART_PORTFOLIO_OFFLINE_MODE", "0")
    
    print("=" * 50)
    print("Environment Check")
    print("=" * 50)
    
    if offline == "1":
        print("⚠️  SMART_PORTFOLIO_OFFLINE_MODE=1")
        print("    Macro data will be zeros. NOT for production runs.")
        return True
        
    if key:
        print(f"✅ FRED_API_KEY loaded: {key[:4]}...{key[-4:]}")
        return True
    else:
        print("❌ FRED_API_KEY not found.")
        print("   1. Create .env file in project root")
        print("   2. Add: FRED_API_KEY=your_key_here")
        print("   3. Get free key: https://fred.stlouisfed.org/docs/api/api_key.html")
        return False

if __name__ == "__main__":
    ok = check_env()
    exit(0 if ok else 1)
