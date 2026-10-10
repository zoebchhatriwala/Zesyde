"""
Package the Zesyde training workspace into a clean zip archive for Lightning.ai.

Includes:
  - src/ (Zesyde library code)
  - scripts/ (pipeline, pause, training scripts)
  - data/ (eval and eval_multilingual test suites)
  - .env (environment variables & API keys)
  - pyproject.toml (package metadata & dependencies)
  - tests/ (test suite)
  - README.md, SPECS.md (specifications)

Excludes:
  - .venv/ (Windows virtual environment)
  - .git/ (version control objects)
  - .pytest_cache/, __pycache__/, *.pyc
  - models/*.onnx, models/*.pt (local models/checkpoints)
  - *.zip (existing archives)
"""
import argparse
import os
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Files and directories required for remote training
INCLUDED_PATHS = [
    "src",
    "scripts",
    "data",
    "tests",
    ".env",
    "pyproject.toml",
    "README.md",
    "SPECS.md",
]

# Explicit exclusions
EXCLUDED_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    ".pytest_cache",
    "__pycache__",
    ".ruff_cache",
    ".mypy_cache",
}

EXCLUDED_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".zip",
    ".tar",
    ".gz",
}


def should_exclude(rel_path: Path) -> bool:
    """Check if relative path should be omitted from the deployment package."""
    for part in rel_path.parts:
        if part in EXCLUDED_DIR_NAMES:
            return True
        if part.startswith(".git"):
            return True

    if rel_path.suffix.lower() in EXCLUDED_EXTENSIONS:
        return True

    if rel_path.name in {".DS_Store", "Thumbs.db"}:
        return True

    return False


def package_workspace(output_zip: Path, include_models: bool = False) -> None:
    output_zip = output_zip.resolve()
    print(f"\nPackaging Zesyde for Lightning.ai deployment...")
    print(f"Target archive: {output_zip}")

    total_files = 0
    total_uncompressed_bytes = 0

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for item_name in INCLUDED_PATHS:
            item_path = PROJECT_ROOT / item_name
            if not item_path.exists():
                print(f"  [Skipped] (Not found): {item_name}")
                continue

            if item_path.is_file():
                rel_path = Path(item_name)
                if not should_exclude(rel_path):
                    zf.write(item_path, arcname=str(rel_path).replace("\\", "/"))
                    total_files += 1
                    total_uncompressed_bytes += item_path.stat().st_size
                    print(f"  [File] {rel_path}")
            elif item_path.is_dir():
                for root, dirs, files in os.walk(item_path):
                    # Filter subdirectories in-place
                    dirs[:] = [d for d in dirs if d not in EXCLUDED_DIR_NAMES and not d.startswith(".git")]
                    for file_name in files:
                        abs_file = Path(root) / file_name
                        rel_path = abs_file.relative_to(PROJECT_ROOT)
                        if should_exclude(rel_path):
                            continue
                        zf.write(abs_file, arcname=str(rel_path).replace("\\", "/"))
                        total_files += 1
                        total_uncompressed_bytes += abs_file.stat().st_size

                print(f"  [Dir]  {item_name}/")

        if include_models:
            models_dir = PROJECT_ROOT / "models"
            if models_dir.exists():
                for root, dirs, files in os.walk(models_dir):
                    dirs[:] = [d for d in dirs if d != "checkpoints"]
                    for file_name in files:
                        abs_file = Path(root) / file_name
                        rel_path = abs_file.relative_to(PROJECT_ROOT)
                        if should_exclude(rel_path):
                            continue
                        zf.write(abs_file, arcname=str(rel_path).replace("\\", "/"))
                        total_files += 1
                        total_uncompressed_bytes += abs_file.stat().st_size
                print(f"  [Dir]  models/ (included via flag)")

    zip_size_mb = output_zip.stat().st_size / (1024 * 1024)
    raw_size_mb = total_uncompressed_bytes / (1024 * 1024)

    print("\n" + "=" * 60)
    print("PACKAGE CREATED SUCCESSFULLY")
    print("=" * 60)
    print(f"Archive:       {output_zip.name}")
    print(f"Location:      {output_zip}")
    print(f"Total files:   {total_files:,}")
    print(f"Raw size:      {raw_size_mb:.2f} MB")
    print(f"Archive size:  {zip_size_mb:.2f} MB")
    print("\nTo upload and unpack on Lightning.ai:")
    print(f"  1. scp {output_zip.name} <lightning-ssh-host>:/teamspace/studios/this_studio/")
    print(f"  2. In Lightning Terminal:")
    print(f"     unzip {output_zip.name} -d .")
    print(f"     pip install -e .")
    print(f"     python scripts/build_prod_model.py --multilingual --train-limit 150000\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create zip archive for Lightning.ai training deployment.")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=PROJECT_ROOT / "model.Zesyde.zip",
        help="Output zip path (default: model.Zesyde.zip in project root)",
    )
    parser.add_argument(
        "--include-models",
        action="store_true",
        default=False,
        help="Include existing trained weights/onnx files in models/ (default: False)",
    )
    args = parser.parse_args()
    package_workspace(args.output, include_models=args.include_models)


if __name__ == "__main__":
    main()
