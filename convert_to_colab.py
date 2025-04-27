"""
bulk_colab_prepare.py
──────────────────────
• Adds minimal Colab metadata to every *.ipynb file
• Inserts/updates an “Open in Colab” badge in the first markdown cell
"""

import pathlib
import urllib.parse
import nbformat

# -------- repo-specific settings -----------------------------------------
GITHUB_USER   = "NirDiamant"
GITHUB_REPO   = "RAG_Techniques"
GITHUB_BRANCH = "main"              # or "master" etc.
# -------------------------------------------------------------------------

badge_template = (
    "[![Open in Colab]"
    "(https://colab.research.google.com/assets/colab-badge.svg)]"
    "({url})"
)

def make_colab_url(notebook_path: pathlib.Path) -> str:
    """Return the full Colab launch URL for a notebook"""
    blob_path = urllib.parse.quote(str(notebook_path).replace("\\", "/"))
    return (
        f"https://colab.research.google.com/github/"
        f"{GITHUB_USER}/{GITHUB_REPO}/blob/{GITHUB_BRANCH}/{blob_path}"
    )

def process_notebook(nb_path: pathlib.Path) -> bool:
    nb = nbformat.read(nb_path, as_version=4)
    changed = False

    # 1. ensure "colab" metadata exists
    if "colab" not in nb.metadata:
        nb.metadata["colab"] = {
            "name": nb_path.name,
            "provenance": [],
            "private_outputs": True,
        }
        changed = True

    # 2. add or update the badge in first markdown cell
    badge_md = badge_template.format(url=make_colab_url(nb_path))
    if nb.cells and nb.cells[0].cell_type == "markdown":
        md = nb.cells[0].source
        if "colab-badge.svg" in md:
            if badge_md not in md:
                nb.cells[0].source = badge_md  # replace old badge line
                changed = True
        else:
            nb.cells[0].source = badge_md + "\n\n" + md
            changed = True
    else:
        nb.cells.insert(
            0,
            nbformat.v4.new_markdown_cell(badge_md)
        )
        changed = True

    if changed:
        nbformat.write(nb, nb_path)
    return changed

def main():
    count = 0
    for nb_path in pathlib.Path(".").rglob("*.ipynb"):
        if process_notebook(nb_path):
            count += 1
            print("updated", nb_path)
    print(f"Finished. {count} notebooks modified.")

if __name__ == "__main__":
    main()
