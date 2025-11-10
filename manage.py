import argparse
import uvicorn
import sys
import os

# --- THE FIX IS HERE ---
# Get the absolute path to the directory containing manage.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Add this directory to Python's search path
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
# -----------------------


def run_server(args):
    """Starts the FastAPI development server."""
    print(f"🚀 Starting NeuroRank API on {args.host}:{args.port}...")
    # We use the string format for uvicorn so it can reload properly
    uvicorn.run(
        "ranker_service.main:app", host=args.host, port=args.port, reload=args.reload
    )


def run_teacher(args):
    """Runs the teacher training pipeline."""
    # Import here to avoid issues if dependencies aren't installed just for running help
    from training_pipeline import train_teacher

    print("👨‍🏫 Starting Teacher (BERT) training...")
    train_teacher.main()


def run_student(args):
    """Runs the student distillation pipeline."""
    from training_pipeline import distill_student

    print("🎓 Starting Student (MiniLM) distillation...")
    distill_student.main()


def main():
    parser = argparse.ArgumentParser(description="NeuroRank Management Interface")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- Server Command ---
    server_parser = subparsers.add_parser("runserver", help="Start the API server")
    server_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    server_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    server_parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for dev"
    )
    server_parser.set_defaults(func=run_server)

    # --- Train Teacher Command ---
    teacher_parser = subparsers.add_parser(
        "train-teacher", help="Train the teacher model"
    )
    teacher_parser.set_defaults(func=run_teacher)

    # --- Train Student Command ---
    student_parser = subparsers.add_parser(
        "train-student", help="Distill into student model"
    )
    student_parser.set_defaults(func=run_student)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
