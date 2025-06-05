import os
import pathlib
import logging

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- Define Project Name ---
PROJECT_NAME = "Image_Captioning"

# --- Define Directories to Create ---
dirs_to_create = [
    PROJECT_NAME,  # Root project directory
    f"{PROJECT_NAME}/config",
    f"{PROJECT_NAME}/data/01_raw",
    f"{PROJECT_NAME}/data/02_interim",
    f"{PROJECT_NAME}/data/03_processed",
    f"{PROJECT_NAME}/data/04_external",
    f"{PROJECT_NAME}/docs",
    f"{PROJECT_NAME}/logs",
    f"{PROJECT_NAME}/models/evaluation",
    f"{PROJECT_NAME}/notebooks",
    f"{PROJECT_NAME}/reports/figures",
    f"{PROJECT_NAME}/tests",
    f"{PROJECT_NAME}/src/data",
    f"{PROJECT_NAME}/src/features",
    f"{PROJECT_NAME}/src/models",
    f"{PROJECT_NAME}/src/evaluation",
    f"{PROJECT_NAME}/src/utils",
    f"{PROJECT_NAME}/src/components",
    f"{PROJECT_NAME}/src/pipeline",
    f"{PROJECT_NAME}/src/constants",
    f"{PROJECT_NAME}/src/config",
    f"{PROJECT_NAME}/src/entity",
]

# --- Define Files to Create ---
files_to_create = [
    # Config
    f"{PROJECT_NAME}/config/config.yaml",
    f"{PROJECT_NAME}/config/logging_config.yaml",
    # Src package init files and modules
    f"{PROJECT_NAME}/src/__init__.py",
    f"{PROJECT_NAME}/src/data/__init__.py",
    f"{PROJECT_NAME}/src/data/make_dataset.py",
    f"{PROJECT_NAME}/src/features/__init__.py",
    f"{PROJECT_NAME}/src/features/build_features.py",
    f"{PROJECT_NAME}/src/models/__init__.py",
    f"{PROJECT_NAME}/src/models/train_model.py",
    f"{PROJECT_NAME}/src/models/predict_model.py",
    f"{PROJECT_NAME}/src/evaluation/__init__.py",
    f"{PROJECT_NAME}/src/evaluation/evaluate.py",
    f"{PROJECT_NAME}/src/utils/__init__.py",
    f"{PROJECT_NAME}/src/utils/logging_setup.py",
    f"{PROJECT_NAME}/src/utils/helpers.py",
    f"{PROJECT_NAME}/src/utils/file_io.py",
    f"{PROJECT_NAME}/src/utils/exceptions.py",
    # Test files
    f"{PROJECT_NAME}/tests/__init__.py",
    f"{PROJECT_NAME}/tests/test_data_processing.py",
    f"{PROJECT_NAME}/tests/test_feature_engineering.py",
    f"{PROJECT_NAME}/tests/test_model_training.py",
    f"{PROJECT_NAME}/tests/test_utils.py",
    # Root files
    f"{PROJECT_NAME}/.env.example",
    f"{PROJECT_NAME}/Dockerfile",
    f"{PROJECT_NAME}/environment.yml",
    f"{PROJECT_NAME}/main.py",
    f"Makefile",
    f"requirements.txt",
    f"setup.py",
    # Components
    f"{PROJECT_NAME}/src/components/__init__.py",
    f"{PROJECT_NAME}/src/components/data_ingestion.py",
    f"{PROJECT_NAME}/src/components/data_validation.py",
    f"{PROJECT_NAME}/src/components/data_transformation.py",
    f"{PROJECT_NAME}/src/components/model_trainer.py",
    f"{PROJECT_NAME}/src/components/model_evaluation.py",
    # Pipeline
    f"{PROJECT_NAME}/src/pipeline/__init__.py",
    f"{PROJECT_NAME}/src/pipeline/stage_01_data_ingestion.py",
    f"{PROJECT_NAME}/src/pipeline/stage_02_data_validation.py",
    f"{PROJECT_NAME}/src/pipeline/stage_03_data_transformation.py",
    f"{PROJECT_NAME}/src/pipeline/stage_04_model_trainer.py",
    f"{PROJECT_NAME}/src/pipeline/stage_05_model_evaluation.py",
    # Constants
    f"{PROJECT_NAME}/src/constants/__init__.py",
    f"{PROJECT_NAME}/src/constants/constants.py",
    # Config
    f"{PROJECT_NAME}/src/config/__init__.py",
    f"{PROJECT_NAME}/src/config/configuration.py",
    # params.yaml
    f"{PROJECT_NAME}/params.yaml",
    # Entity
    f"{PROJECT_NAME}/src/entity/__init__.py",
    f"{PROJECT_NAME}/src/entity/config_entity.py",
]

# --- Basic Gitignore Content ---
gitignore_content = """
# Standard Python ignores...
__pycache__/
*.py[cod]
*.so

# Environment stuff...
.env
.venv
env/
venv/

# Data (usually managed outside git or with LFS/DVC)
# data/

# Logs
logs/
*.log

# Models (usually large)
models/*.pkl
models/*.h5
models/*.onnx

# Notebook checkpoints
.ipynb_checkpoints

# IDE folders
.vscode/
.idea/

# OS files
.DS_Store
Thumbs.db
"""

# --- Create Structure ---
logging.info(f"Starting project structure creation for: {PROJECT_NAME}")

# Create directories
for dir_path_str in dirs_to_create:
    path = pathlib.Path(dir_path_str)
    try:
        path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created directory (or verified exists): {path}")
    except OSError as e:
        logging.error(f"Failed to create directory {path}: {e}")

# Create files
for file_path_str in files_to_create:
    file_path = pathlib.Path(file_path_str)
    try:
        # Ensure parent directory exists before touching the file
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.touch(exist_ok=True)
        logging.info(f"Created file (or verified exists): {file_path}")

        # Add initial content to specific files
        if file_path.name == "README.md":
            # Avoid overwriting if file already existed and had content
            if file_path.stat().st_size == 0:
                file_path.write_text(f"# {PROJECT_NAME}\n", encoding='utf-8')
                logging.info(f"Added title to {file_path.name}")
    except OSError as e:
        logging.error(f"Failed to create or write to file {file_path}: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred for file {file_path}: {e}")

# Create or update .gitignore in the root directory
gitignore_path = pathlib.Path(".gitignore")
try:
    if not gitignore_path.exists():
        gitignore_path.write_text(gitignore_content.strip(), encoding='utf-8')
        logging.info(f"Created .gitignore file in the root directory.")
    else:
        with open(gitignore_path, "r", encoding='utf-8') as f:
            existing_content = f.read()

        new_lines = [line.strip() for line in gitignore_content.strip().splitlines()]
        existing_lines = [line.strip() for line in existing_content.splitlines()]

        for line in new_lines:
            if line not in existing_lines:
                with open(gitignore_path, "a", encoding='utf-8') as f:
                    f.write(f"\n{line}")
                logging.info(f"Added '{line}' to .gitignore.")
        logging.info(".gitignore file updated.")

except OSError as e:
    logging.error(f"Failed to create or write to .gitignore: {e}")
except Exception as e:
    logging.error(f"An unexpected error occurred for .gitignore: {e}")

# Create the readme file on the root of the project.
readme_path = pathlib.Path("README.md")
try:
    if not readme_path.exists():
        readme_path.write_text(f"# {PROJECT_NAME}\n", encoding='utf-8')
        logging.info("Created README.md file.")
    else:
        if readme_path.stat().st_size == 0:
            readme_path.write_text(f"# {PROJECT_NAME}\n", encoding='utf-8')
            logging.info("Added title to README.md")

except OSError as e:
    logging.error(f"Failed to create or write to README.md: {e}")
except Exception as e:
    logging.error(f"An unexpected error occurred for README.md: {e}")

logging.info("Project structure creation process finished.")