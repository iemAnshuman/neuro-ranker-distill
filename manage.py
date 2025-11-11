import argparse
import uvicorn
import sys
import os

def run_server(args):
    """Starts the FastAPI development server."""
    print(f"噫 Starting NeuroRank API on {args.host}:{args.port}...")
    uvicorn.run(
        "ranker_service.main:app", host=args.host, port=args.port, reload=args.reload
    )


def run_teacher(args):
    """Runs the teacher training pipeline."""
    import train_teacher
    print("捉窶昨沛ｫ Starting Teacher (BERT) training...")
    train_teacher.run_training(args)


def run_student(args):
    """Runs the student distillation pipeline."""
    import distill_student
    print("雌 Starting Student (MiniLM) distillation...")
    distill_student.run_distillation(args)


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
    teacher_parser.add_argument(
        "--data_dir",
        default="/content/drive/MyDrive/ms_marco_project",
        help="Path to the MS MARCO dataset"
    )
    teacher_parser.add_argument("--model", default="microsoft/MiniLM-L12-H384-uncased")
    teacher_parser.add_argument("--epochs", type=int, default=1)
    teacher_parser.add_argument("--lr", type=float, default=2e-5)
    teacher_parser.add_argument("--batch", type=int, default=16)
    teacher_parser.add_argument("--max_len", type=int, default=256)
    # --- START OF MODIFICATION ---
    teacher_parser.add_argument(
        "--out_dir", 
        default="/content/drive/MyDrive/ms_marco_project/models/teacher",
        help="Output directory for teacher model"
    )
    # --- END OF MODIFICATION ---
    teacher_parser.set_defaults(func=run_teacher)

    # --- Train Student Command ---
    student_parser = subparsers.add_parser(
        "train-student", help="Distill into student model"
    )
    student_parser.add_argument(
        "--data_dir", 
        default="/content/drive/MyDrive/ms_marco_project", 
        help="Path to the MS MARCO dataset"
    )
    student_parser.add_argument(
        "--teacher", 
        # Set default teacher path to match new teacher output
        default="/content/drive/MyDrive/ms_marco_project/models/teacher/best.pt",
        help="Path to teacher best.pt"
    )
    student_parser.add_argument("--student", default="sentence-transformers/all-MiniLM-L6-v2")
    student_parser.add_argument("--epochs", type=int, default=1)
    student_parser.add_argument("--lr", type=float, default=3e-5)
    student_parser.add_argument("--batch", type=int, default=64)
    student_parser.add_argument("--max_len", type=int, default=256)
    student_parser.add_argument("--temp", type=float, default=3.0)
    # --- START OF MODIFICATION ---
    student_parser.add_argument(
        "--out_dir", 
        default="/content/drive/MyDrive/ms_marco_project/models/student",
        help="Output directory for student model"
    )
    # --- END OF MODIFICATION ---
    student_parser.set_defaults(func=run_student)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()