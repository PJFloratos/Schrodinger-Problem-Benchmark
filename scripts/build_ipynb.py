import os
import json
import re


def format_source(text):
    """Formats text into a list of strings with newlines for Jupyter JSON."""
    lines = text.split("\n")
    return [line + "\n" for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])


def create_flat_notebook(output_filename="colab_training.ipynb"):
    cells = []

    # --- NEW: 1. Add a cell to install Hydra in Colab ---
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": format_source("!pip install hydra-core omegaconf --quiet"),
        }
    )
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": format_source("!pip install torchmetrics[image] --quiet"),
        }
    )

    def add_cell(filepath, section_title):
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found. Skipping.")
            return

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        cells.append(
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"### {section_title}\n", f"`{filepath}`"],
            }
        )

        cell_content = f"# Original file: {filepath}\n\n{content}"
        cells.append(
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": format_source(cell_content),
            }
        )

    # --- NEW: 2. Function to recreate YAML config files on Colab's disk ---
    def add_yaml_cell(filepath):
        if not os.path.exists(filepath):
            return

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract directory to create it in Colab first
        dir_name = os.path.dirname(filepath)
        mkdir_cmd = f"!mkdir -p {dir_name}" if dir_name else ""

        # %%writefile writes the cell contents to the specified path
        cell_content = f"%%writefile {filepath}\n{content}"

        cells.append(
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": format_source(
                    f"### Config: Writes `{filepath}` to Colab disk"
                ),
            }
        )
        cells.append(
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": format_source(mkdir_cmd),
            }
        )
        cells.append(
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": format_source(cell_content),
            }
        )

    # 3. Walk through the config directory (now at root level next to main.py)
    config_dir = "configs"
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": format_source("# configs"),
        }
    )
    if os.path.exists(config_dir):
        for root, dirs, files in os.walk(config_dir):
            root_name = root.split("configs")[-1]
            if root_name != "":
                cells.append(
                    {
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": format_source(f"## {root_name}"),
                    }
                )
            for file in files:
                if file.endswith((".yaml", ".yml")):
                    full_path = os.path.join(root, file)
                    add_yaml_cell(full_path)
    else:
        print(f"Warning: Config directory '{config_dir}' not found.")

    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": format_source("# src/"),
        }
    )

    # 4. Iterate through the rest of the src/ modules (Python files)
    module_folders = [
        "utils",
        "datasets",
        "core",
        os.path.join("models", "components"),
        "models",
        "metrics",
        "training",
    ]

    for folder in module_folders:
        if folder.startswith("__"):
            continue
        folder_path = os.path.join("src", folder)
        if os.path.exists(folder_path):
            file_list = os.listdir(folder_path)
            file_list.remove("__pycache__")

            if folder == "training":
                file_list.remove("base_trainer.py")
                file_list.remove("trainer.py")
                file_list = ["base_trainer.py", "trainer.py"] + file_list

            if folder == "utils":
                file_list.remove("log.py")
                file_list = ["log.py"] + file_list

            cells.append(
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": format_source(f"## {folder}"),
                }
            )

            for file in file_list:
                if file.endswith(".py") and not file.startswith("__"):
                    add_cell(
                        os.path.join(folder_path, file),
                        f"Module: {folder.capitalize()} - {file}",
                    )

    if os.path.exists("src/pipeline.py"):
        add_cell("src/pipeline.py", "Pipeline")

    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": format_source("# main"),
        }
    )

    # 5. Handle main.py execution with environment mocking
    main_path = "main.py"
    if os.path.exists(main_path):
        cells.append(
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": format_source(f"### Main Training Script\n`{main_path}`"),
            }
        )
        with open(main_path, "r", encoding="utf-8") as f:
            main_content = f.read()

        # Hydra resolves config_path relative to the *cell's temp file* in Jupyter,
        # so rewrite relative config_path values to absolute Colab paths.
        def to_abs(match):
            quote, path = match.group(1), match.group(2)
            if os.path.isabs(path):
                return match.group(0)
            abs_path = os.path.normpath(os.path.join("/content", path))
            return f"config_path={quote}{abs_path}{quote}"

        main_content = re.sub(r"config_path\s*=\s*([\"'])(.*?)\1", to_abs, main_content)

        safe_main_content = (
            "# Original file: main.py\n"
            "import sys\n\n"
            "# Mock command line to prevent crash from Jupyter's background arguments\n"
            "sys.argv = ['main.py']\n\n"
            f"{main_content}"
        )

        cells.append(
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": format_source(safe_main_content),
            }
        )

    # 6. Construct Notebook JSON
    notebook = {
        "cells": cells,
        "metadata": {"language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    # 7. Write to .ipynb
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)

    print(f"Successfully generated {output_filename}")


if __name__ == "__main__":
    create_flat_notebook()
