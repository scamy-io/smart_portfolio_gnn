#!/usr/bin/env python3
"""Week 8 Final Integration Verification."""

from pathlib import Path


def main():
    print("=" * 70)
    print("WEEK 8 FINAL INTEGRATION VERIFICATION")
    print("=" * 70)

    required_files = [
        "README.md",
        "LICENSE",
        "CONTRIBUTING.md",
        "Dockerfile",
        "Makefile",
        "requirements.txt",
        "setup.py",
        "docs/architecture.md",
        "scripts/run_full_pipeline.py",
        "scripts/download_data.py",
        "scripts/build_graphs.py",
        "scripts/train_model.py",
        "scripts/run_backtest.py",
        "scripts/run_risk_engine.py",
        "dashboard/app.py",
        "src/data_ingestion/yfinance_downloader.py",
        "src/graph_builder/htgat.py",
        "src/models/htgat.py",
        "src/risk_engine/concentration_metrics.py",
        "src/rebalancing/cost_aware_optimizer.py",
        "src/evaluation/backtester.py",
        "src/streaming/incremental_updater.py",
        "tests/test_model.py",
        ".github/workflows/ci.yml",
    ]

    print("\n--- Checking Required Files ---")
    missing = []
    for f in required_files:
        p = Path(f)
        if p.exists():
            print(f"  ✓ {f}")
        else:
            print(f"  ✗ MISSING: {f}")
            missing.append(f)

    # ── Check README quality ──
    print("\n--- Checking README ---")
    readme = Path("README.md").read_text()
    assert "# Smart Portfolio" in readme or "Portfolio Rebalancing" in readme
    assert "Quick Start" in readme or "quick start" in readme
    assert "pip install" in readme or "make install" in readme
    print("  ✓ README has title and quick start")

    # ── Check full pipeline script ──
    print("\n--- Checking Full Pipeline ---")
    pipeline = Path("scripts/run_full_pipeline.py")
    assert pipeline.exists()
    text = pipeline.read_text()
    assert "download" in text
    assert "train" in text
    assert "backtest" in text
    print("  ✓ Full pipeline script exists")

    print("\n" + "=" * 70)
    if missing:
        print(f"⚠ {len(missing)} files missing — review above")
    else:
        print("🎉 ALL FILES PRESENT — PROJECT COMPLETE")
    print("=" * 70)

    print("\n📦 RELEASE CHECKLIST:")
    print("  [ ] Push to GitHub")
    print("  [ ] Enable GitHub Actions")
    print(
        "  [ ] Add repo description + tags: gnn, portfolio-optimization, pytorch-geometric"
    )


if __name__ == "__main__":
    main()
