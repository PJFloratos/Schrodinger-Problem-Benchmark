import os
import json


def format_source(text):
    """Formats text into a list of strings with newlines for Jupyter JSON."""
    lines = text.split("\n")
    return [line + "\n" for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])


def create_flat_notebook(output_filename="colab_training.ipynb"):
    cells = []

    def add_cell(filepath, section_title):
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found. Skipping.")
            return

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Add a markdown header for organization
        cells.append(
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"### {section_title}\n", f"`{filepath}`"],
            }
        )

        # Add the code cell with a comment at the top
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

    # 1. Force 'utils' first, starting specifically with log.py[cite: 1]
    utils_dir = os.path.join("src", "utils")
    if os.path.exists(utils_dir):
        # Add logger first
        log_path = os.path.join(utils_dir, "log.py")
        if os.path.exists(log_path):
            add_cell(log_path, "Utils: Logger")

        # Add remaining utils
        for file in sorted(os.listdir(utils_dir)):
            if file.endswith(".py") and file != "log.py" and not file.startswith("__"):
                add_cell(os.path.join(utils_dir, file), f"Utils: {file}")

    # 2. Iterate through the rest of the src/ modules[cite: 1]
    # Defining an order that typically helps with dependencies when flattened
    module_folders = ["dataset", "models", "evaluation", "training"]

    for folder in module_folders:
        folder_path = os.path.join("src", folder)
        if os.path.exists(folder_path):
            for file in sorted(os.listdir(folder_path)):
                if file.endswith(".py") and not file.startswith("__"):
                    add_cell(
                        os.path.join(folder_path, file),
                        f"Module: {folder.capitalize()} - {file}",
                    )

    # 3. Add the main training script last[cite: 1]
    add_cell("train.py", "Main Training Script")

    # 4. Construct Notebook JSON
    notebook = {
        "cells": cells,
        "metadata": {"language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    # 5. Write to .ipynb
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)

    print(f"Successfully generated {output_filename}")


if __name__ == "__main__":
    create_flat_notebook()
