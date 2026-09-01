"""Small command-line entrypoint for checks and the deterministic demo."""

from __future__ import annotations

import argparse

from defensive_avionics.integration.orchestrator import DemoOrchestrator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline academic AI simulator")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("check", help="print the configured safe runtime mode")
    subparsers.add_parser("demo", help="run a deterministic synthetic-state demo")
    subparsers.add_parser("ui", help="open the optional Pygame starter window")
    subparsers.add_parser("dashboard", help="launch the Streamlit engineering HUD dashboard")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "check":
        print("Mode: offline academic simulation")
        print("Hardware control: disabled")
        print("Operational outputs: disabled")
        return 0

    if args.command == "demo":
        orchestrator = DemoOrchestrator()
        for frame in orchestrator.synthetic_sequence():
            print(
                f"frame={frame.frame_id:02d} status={frame.status:<8} "
                f"signal={frame.signal_label:<8} urgency={frame.vision_urgency}"
            )
        return 0

    if args.command == "ui":
        from defensive_avionics.ui.pygame_app import run

        return run()

    if args.command == "dashboard":
        import subprocess
        import sys

        return subprocess.call([sys.executable, "-m", "streamlit", "run", "app/dashboard.py"])

    build_parser().print_help()
    return 0
