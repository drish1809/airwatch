"""
run.py — Project Entry Point
=============================
Unified CLI to seed data, collect, train, and launch the app.

Usage
-----
python run.py --demo          # generate synthetic data + train model
python run.py --collect       # start live data collection (API key needed)
python run.py --train         # re-train ML model on current data
python run.py --app           # launch Streamlit dashboard
python run.py --all           # collect + train + launch app
"""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path

# ── Ensure project root on sys.path ─────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.utils import load_config, setup_logging, ensure_dirs


def run_demo(config: dict) -> None:
    """Generate synthetic data and train the ML model."""
    print("\n🚀 Generating 30-day synthetic dataset…")
    from src.demo_data import generate
    generate(
        db_path  = config["data"]["db_path"],
        csv_path = f"{config['data']['raw_path']}/air_quality_data.csv",
    )
    run_train(config)


def run_collect(config: dict) -> None:
    """Start the live data collection scheduler (runs until interrupted)."""
    from src.collector import AirQualityCollector
    collector  = AirQualityCollector(config)
    scheduler  = collector.start_scheduler()

    print("\n📡 Collector running — press Ctrl+C to stop.\n")

    def _shutdown(sig, frame):
        print("\nShutting down scheduler…")
        scheduler.shutdown(wait=False)
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    while True:
        time.sleep(10)


def run_train(config: dict) -> None:
    """Process collected data and train the ML pipeline."""
    from src.processor import DataProcessor
    from src.trainer   import ModelTrainer

    print("\n🔧 Processing data…")
    processor = DataProcessor()
    db_path   = config["data"]["db_path"]

    try:
        df = processor.run_pipeline(db_path)
    except Exception as exc:
        csv_path = f"{config['data']['raw_path']}/air_quality_data.csv"
        print(f"  DB load failed ({exc}), falling back to CSV…")
        df = processor.load_from_csv(csv_path)
        df = processor.clean(df)
        df = processor.engineer_features(df)

    X, y = processor.prepare_ml_features(df)

    if len(X) < 50:
        print(f"⚠  Only {len(X)} training samples — gather more data for better accuracy.")

    print(f"\n🤖 Training models on {len(X)} samples…")
    trainer = ModelTrainer(config)
    results = trainer.train_and_select(X, y, feature_columns=processor.feature_columns)
    trainer.print_leaderboard(results)
    print(f"\n✅ Best model: {trainer.best_model_name}  "
          f"RMSE={trainer.best_metrics['rmse']}  R²={trainer.best_metrics['r2']}")


def run_app() -> None:
    """Launch the Streamlit dashboard."""
    print("\n🌐 Launching Streamlit dashboard…")
    app_path = str(ROOT / "app" / "main.py")
    subprocess.run(
        ["streamlit", "run", app_path,
         "--server.headless", "true",
         "--theme.base", "dark"],
        check=True,
    )


def run_all(config: dict) -> None:
    """Collect data, train model, and launch app — for deployment use."""
    from src.collector import AirQualityCollector
    from src.processor import DataProcessor
    from src.trainer   import ModelTrainer
    import threading

    # Start background collector
    collector = AirQualityCollector(config)
    scheduler = collector.start_scheduler()

    # Train immediately after first collection
    print("\n🔧 Running initial model training…")
    try:
        run_train(config)
    except Exception as exc:
        print(f"Training failed (probably not enough data yet): {exc}")

    # Launch app in a thread, keep collector alive
    app_thread = threading.Thread(target=run_app, daemon=True)
    app_thread.start()

    print("\n✅ System running. Press Ctrl+C to stop.\n")

    def _shutdown(sig, frame):
        scheduler.shutdown(wait=False)
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    while True:
        time.sleep(60)


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Air Quality Monitoring & Prediction System",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--demo",    action="store_true", help="Generate synthetic data + train model")
    parser.add_argument("--collect", action="store_true", help="Start live data collection")
    parser.add_argument("--train",   action="store_true", help="Re-train ML model")
    parser.add_argument("--app",     action="store_true", help="Launch Streamlit dashboard")
    parser.add_argument("--all",     action="store_true", help="Collect + train + launch")
    parser.add_argument("--config",  default="config/config.yaml", help="Config file path")
    args = parser.parse_args()

    config = load_config(args.config)
    log_cfg = config.get("logging", {})
    setup_logging(
        log_level    = log_cfg.get("level", "INFO"),
        log_file     = log_cfg.get("log_file", "logs/app.log"),
        max_bytes    = log_cfg.get("max_bytes", 10_485_760),
        backup_count = log_cfg.get("backup_count", 5),
    )

    ensure_dirs([
        config["data"]["raw_path"],
        config["data"]["processed_path"],
        config["models"]["save_path"],
        "logs",
    ])

    if args.demo:
        run_demo(config)
        print("\n🎉 Demo ready! Launch the app with:  python run.py --app")
    elif args.collect:
        run_collect(config)
    elif args.train:
        run_train(config)
    elif args.app:
        run_app()
    elif args.all:
        run_all(config)
    else:
        parser.print_help()
        print("\n💡 Quick start:  python run.py --demo && python run.py --app")


if __name__ == "__main__":
    main()
