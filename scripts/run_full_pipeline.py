#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


def run_step(step_name, cmd):
    print("\n" + "=" * 60)
    print(f"🚀 RUNNING: {step_name}")
    print("=" * 60)
    try:
        subprocess.run(cmd, check=True)
        print(f"✓ {step_name} completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"❌ {step_name} failed with exit code {e.returncode}.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="End-to-End Pipeline Runner")
    parser.add_argument(
        "--skip-download", action="store_true", help="Skip data downloading"
    )
    parser.add_argument("--skip-train", action="store_true", help="Skip model training")
    parser.add_argument(
        "--skip-dashboard", action="store_true", help="Skip launching dashboard"
    )
    parser.add_argument(
        "--skip-gdelt", action="store_true", help="Skip GDELT processing"
    )
    args = parser.parse_args()

    # 1. Check prerequisites
    if not Path("configs/config.yaml").exists():
        print("⚠ configs/config.yaml not found. Proceeding with defaults if possible.")

    # 2. Run Week 1: download_data
    if not args.skip_download:
        dl_cmd = [sys.executable, "scripts/download_data.py"]
        if args.skip_gdelt:
            dl_cmd.append("--skip-gdelt")
        run_step("Week 1: Data Ingestion", dl_cmd)
    else:
        print("\n⏭ Skipping Data Ingestion (--skip-download)")

    # 3. Run Week 2: build_graphs
    run_step("Week 2: Graph Construction", [sys.executable, "scripts/build_graphs.py"])

    # 4. Run Week 3: train_model
    if not args.skip_train:
        run_step("Week 3: Model Training", [sys.executable, "scripts/train_model.py"])
    else:
        print("\n⏭ Skipping Model Training (--skip-train)")

    # 5. Run Week 4: run_risk_engine
    run_step(
        "Week 4: Risk Engine Analysis", [sys.executable, "scripts/run_risk_engine.py"]
    )

    # 6. Run Week 5: run_backtest
    run_step(
        "Week 5: Walk-Forward Backtest", [sys.executable, "scripts/run_backtest.py"]
    )

    # 7. Generate all plots (assuming backtest generates plots or a separate script exists)
    # run_step("Generate Plots", [sys.executable, "scripts/generate_plots.py"])

    print("\n" + "=" * 60)
    print("🎉 FULL PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print("Outputs:")
    print("  - Models: models/best_htgat.pt")
    print("  - Backtest Reports: reports/backtest_metrics.json")
    print("  - Risk Reports: reports/risk_report.json")

    # 8. Launch dashboard
    if not args.skip_dashboard:
        print("\nLaunching Dashboard...")
        try:
            subprocess.run([sys.executable, "scripts/run_dashboard.py"])
        except KeyboardInterrupt:
            print("\nPipeline execution finished.")
    else:
        print("\n⏭ Skipping Dashboard Launch (--skip-dashboard)")


if __name__ == "__main__":
    main()
