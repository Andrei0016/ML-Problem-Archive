import logging
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import nbformat

# ─────────────── CONFIGURATION ────────────────────

# Directory to scan for .ipynb files (relative to this script)
BASE_DIR = Path("..").resolve()

# Path to README to update
README_PATH = BASE_DIR / "README.md"

# Markers in README.md between which the TOC will be injected
START_MARKER = "<!-- NOTEBOOK-TOC-START -->"
END_MARKER = "<!-- NOTEBOOK-TOC-END -->"

# GitHub repository slug in the form "owner/repo", used to construct nbviewer/raw URLs
GITHUB_REPO = "Andrei0016/ML-Problem-Archive"

# ─────────────── SETUP LOGGING ────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────── NOTEBOOK DISCOVERY ────────────────────


def get_notebook_paths(base_dir: Path) -> List[Path]:
    """
    Recursively find all .ipynb files under base_dir, skipping hidden directories.
    Returns a list of Paths, relative to base_dir.
    """
    notebooks: List[Path] = []
    for path in base_dir.rglob("*.ipynb"):
        # Skip any path that has a hidden directory in its parts
        if any(part.startswith(".") for part in path.parts):
            continue
        notebooks.append(path.relative_to(base_dir))
    logger.info("Found %d notebooks under %s", len(notebooks), base_dir)
    return notebooks


# ─────────────── TAG & TITLE EXTRACTION ────────────────────


def extract_tags_and_title(nb_path: Path) -> Tuple[List[str], str]:
    """
    Open the notebook at nb_path (absolute), then:
      1) Find the first Markdown cell, scan it for the first top-level heading '# ...' → title.
      2) Find the last Markdown cell, scan it for a fenced JSON block containing {"Tags": [...]}. → tags.
    Returns (tags_list, title_string).
    If title not found, fall back to parent folder name.
    If tags not found or invalid, return empty list for tags.
    """
    tags: List[str] = []
    title: Optional[str] = None

    try:
        nb = nbformat.read(nb_path, as_version=4)
    except Exception as e:
        logger.warning("Could not parse '%s': %s", nb_path, e)
        # Fallback: title is parent folder name
        return [], nb_path.parent.name

    cells = nb.get("cells", [])
    # 1) Extract title from the first Markdown cell that contains a '# ' heading
    for cell in cells:
        if cell.get("cell_type") != "markdown":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        # Look for a line starting with "# " (not "##" or "###") at the beginning
        for line in source.splitlines():
            if line.startswith("# "):
                title = line.lstrip("# ").strip()
                break
        if title:
            break
    if not title:
        # Use parent folder name instead of file stem
        title = nb_path.parent.name

    # Rest of the function remains the same
    for cell in reversed(cells):
        if cell.get("cell_type") != "markdown":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        pattern = re.compile(r"```json\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE)
        match = pattern.search(source)
        if not match:
            continue
        json_text = match.group(1)
        try:
            data = json.loads(json_text)
        except json.JSONDecodeError as e:
            logger.warning("Invalid JSON in '%s' last MD cell: %s", nb_path, e)
            continue
        raw_tags = data.get("Tags")
        if isinstance(raw_tags, list):
            for t in raw_tags:
                if isinstance(t, str) and t.strip():
                    tags.append(t.strip())
        break

    return tags, title


# ─────────────── BUILD TAG INDEX ────────────────────


def build_tag_index(nb_paths: List[Path]) -> Dict[str, List[Tuple[Path, str]]]:
    """
    Given a list of notebook paths (relative to BASE_DIR), return a mapping:
      { tag: [ (relative_path, title), ... ] }
    Notebooks with no tags go under "Untagged".
    """
    index: Dict[str, List[Tuple[Path, str]]] = {}
    for rel_path in nb_paths:
        abs_path = BASE_DIR / rel_path
        tags, title = extract_tags_and_title(abs_path)
        if not tags:
            index.setdefault("Untagged", []).append((rel_path, title))
        else:
            for raw_tag in tags:
                normalized = raw_tag.strip()
                index.setdefault(normalized, []).append((rel_path, title))
    logger.info(
        "Built tag index with %d tags (including 'Untagged' if any)",
        len(index),
    )
    return index


# ─────────────── MARKDOWN RENDERING ────────────────────


def render_tag_index(tags: List[str]) -> str:
    """
    Generate a small bullet list of tags, each linking to its section anchor.
    E.g.:
      - [Classification](#classification)
      - [Clustering](#clustering)
    """
    lines: List[str] = ["## Notebook Tags", ""]
    for tag in sorted(tags, key=str.lower):
        anchor = tag.lower().replace(" ", "-")
        lines.append(f"- [{tag}](#{anchor})")
    lines.append("")  # blank line after
    return "\n".join(lines)


def render_tag_section(
        tag: str, notebooks: List[Tuple[Path, str]], repo: str
) -> str:
    """
    Render a Markdown table for a given tag. Each row has:
      | Title | Path & Links |
    """
    lines: List[str] = []
    lines.append(f"### {tag}")
    lines.append("")
    lines.append("| Title | Path & Links |")
    lines.append("|-------|--------------|")
    for rel_path, title in sorted(notebooks, key=lambda x: x[1].lower()):
        nb_posix = rel_path.as_posix()
        nb_posix_w_content = nb_posix.replace("content", '')
        nbviewer_url = f"https://andrei0016.github.io/ML-Problem-Archive/lab?path={nb_posix_w_content}"
        raw_url = f"https://github.com/{repo}/blob/master/{nb_posix}"
        link_md = f"[view]({nbviewer_url})<br>[(raw)]({raw_url})"
        lines.append(f"| {title} | {link_md} |")
    lines.append("")  # blank line after table
    return "\n".join(lines)


def render_untagged(
        notebooks: List[Tuple[Path, str]], repo: str
) -> str:
    """
    Wrap the list of untagged notebooks in a collapsible <details> block.
    """
    count = len(notebooks)
    lines: List[str] = []
    lines.append("<details>")
    lines.append(f"<summary>Show {count} untagged notebook{'s' if count != 1 else ''}</summary>")
    lines.append("")
    for rel_path, title in sorted(notebooks, key=lambda x: x[1].lower()):
        nbviewer_url = f"https://nbviewer.org/github/{repo}/blob/master/{rel_path.as_posix()}"
        lines.append(f"- [{title}]({nbviewer_url})  ")
    lines.append("")
    lines.append("</details>")
    lines.append("")
    return "\n".join(lines)


def generate_markdown(tag_index: Dict[str, List[Tuple[Path, str]]], repo: str) -> str:
    """
    Combine the tag index, individual tag tables, and a collapsible 'Untagged' section
    into one Markdown string (without the START/END markers).
    """
    # Separate out "Untagged" if present, and all other tags
    all_tags = [t for t in tag_index.keys() if t.lower() != "untagged"]
    md_parts: List[str] = []

    # 1) Top-level tag index
    md_parts.append(render_tag_index(all_tags))

    # 2) Tables for each tag
    for tag in sorted(all_tags, key=str.lower):
        md_parts.append(render_tag_section(tag, tag_index[tag], repo))

    # 3) Collapsible "Untagged" section (if any)
    if "Untagged" in tag_index:
        md_parts.append("### Untagged")
        md_parts.append("")
        md_parts.append(render_untagged(tag_index["Untagged"], repo))

    return "\n".join(md_parts).strip()


# ─────────────── README UPDATE ────────────────────


def replace_between_markers(
        readme_text: str, start_marker: str, end_marker: str, new_block_body: str
) -> str:
    """
    Find the lines containing start_marker and end_marker in readme_text.
    Replace everything between (exclusive) with new_block_body (split into lines),
    but keep the markers themselves. Return the updated README content.
    """
    lines = readme_text.splitlines()
    try:
        start_idx = next(i for i, line in enumerate(lines) if start_marker in line)
        end_idx = next(i for i, line in enumerate(lines) if end_marker in line)
    except StopIteration:
        raise RuntimeError(
            f"Could not find both markers ({start_marker} / {end_marker}) in {README_PATH}"
        )

    if start_idx >= end_idx:
        raise RuntimeError(f"START marker appears after END marker in {README_PATH}")

    new_block_lines = [start_marker] + new_block_body.splitlines() + [end_marker]
    updated_lines = lines[:start_idx] + new_block_lines + lines[end_idx + 1 :]
    return "\n".join(updated_lines) + "\n"


def update_readme_file(
        readme_path: Path, start_marker: str, end_marker: str, new_block_body: str
) -> None:
    """
    Read (or create) README.md, inject the new TOC between the markers, and write it back.
    """
    if readme_path.exists():
        content = readme_path.read_text(encoding="utf-8")
    else:
        logger.info("README not found. Creating a new one at %s", readme_path)
        content = f"# Project Notebooks\n{start_marker}\n{end_marker}\n"

    updated = replace_between_markers(content, start_marker, end_marker, new_block_body)
    readme_path.write_text(updated, encoding="utf-8")
    logger.info("README.md successfully updated.")


# ─────────────── MAIN EXECUTION ────────────────────


def main() -> None:
    logger.info("Starting notebook TOC generation...")

    # 1) Find all notebooks
    notebook_paths = get_notebook_paths(BASE_DIR)

    if not notebook_paths:
        logger.warning("No notebooks found under %s. Exiting.", BASE_DIR)
        return

    # 2) Build tag → [ (path, title) ] index
    tag_index = build_tag_index(notebook_paths)

    total_notebooks = sum(len(v) for v in tag_index.values())
    total_tags = len(tag_index)
    logger.info(
        "Discovered %d notebooks across %d tag groups.", total_notebooks, total_tags
    )

    # 3) Generate the Markdown snippet (without markers)
    toc_md = generate_markdown(tag_index, GITHUB_REPO)

    # 4) Inject between markers and update README.md
    try:
        update_readme_file(
            readme_path=README_PATH,
            start_marker=START_MARKER,
            end_marker=END_MARKER,
            new_block_body=toc_md,
        )
    except RuntimeError as e:
        logger.error(str(e))
        logger.error(
            "Please ensure that README.md contains the markers:\n%s\n%s",
            START_MARKER,
            END_MARKER,
        )
        return

    logger.info(
        "✅ Updated %s with %d notebooks across %d tags.", README_PATH, total_notebooks, total_tags
    )


if __name__ == "__main__":
    main()
