import os
import re
import json
import nbformat

# Path to README and markers:
README_PATH = "README.md"
START_MARKER = "<!-- NOTEBOOK-TOC-START -->"
END_MARKER = "<!-- NOTEBOOK-TOC-END -->"

def find_notebooks(base_dir="."):
    notebooks = []
    for root, dirs, files in os.walk(base_dir):
        # Skip hidden folders
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for fname in files:
            if fname.endswith(".ipynb"):
                full_path = os.path.join(root, fname)
                rel_path = os.path.relpath(full_path, base_dir)
                notebooks.append(rel_path)
    return notebooks

def extract_tags_from_last_md(nb_path):
    try:
        nb = nbformat.read(nb_path, as_version=4)
    except Exception as e:
        print(f"Warning: could not parse '{nb_path}': {e}")
        return []

    # Walk cells in reverse, stopping at the first Markdown cell
    for cell in reversed(nb.get("cells", [])):
        if cell.get("cell_type") == "markdown":
            # A Markdown cell's "source" may be a single string or a list of strings
            source = cell.get("source", "")
            if isinstance(source, list):
                source = "".join(source)

            # Look for a fenced JSON block: ```json ... ```
            # The regex captures everything between ```json and the next ```
            pattern = re.compile(r"```json\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE)
            match = pattern.search(source)
            if not match:
                # No JSON fence in this Markdown cell → no tags here
                return []

            json_text = match.group(1)
            try:
                data = json.loads(json_text)
            except json.JSONDecodeError as e:
                print(f"Warning: invalid JSON in '{nb_path}' last MD cell: {e}")
                return []

            # Expecting a key "Tags" whose value is a list of strings
            raw_tags = data.get("Tags", [])
            if not isinstance(raw_tags, list):
                return []
            tags = []
            for t in raw_tags:
                if isinstance(t, str) and t.strip():
                    tags.append(t.strip())
            return tags

    # If we never found a Markdown cell at all:
    return []

def build_tag_index(nb_paths):
    tag_index = {}
    for nb in nb_paths:
        tags = extract_tags_from_last_md(nb)
        if not tags:
            tag_index.setdefault("Untagged", []).append(nb)
        else:
            for tag in tags:
                tag_index.setdefault(tag, []).append(nb)
    return tag_index

def generate_markdown(tag_index):
    GITHUB_REPO = "Andrei0016/MachineLearning-Problem-Collection"
    
    def get_display_name(path):
        dirname = os.path.dirname(path)
        if dirname:
            # Use folder name, replacing directory separators with '/'
            return dirname.replace('\\', '/')
        # Fallback to filename without extension if no folder
        return os.path.splitext(os.path.basename(path))[0]
    
    md_lines = []
    for tag in sorted(tag_index.keys(), key=lambda s: s.lower()):
        md_lines.append(f"### {tag}")
        for nb_path in sorted(tag_index[tag], key=lambda p: p.lower()):
            display_name = get_display_name(nb_path)
            
            # Create nbviewer link
            nbviewer_url = f"https://nbviewer.org/github/{GITHUB_REPO}/blob/master/{nb_path}"
            md_lines.append(f"- [{display_name}]({nbviewer_url}) [(raw)]({nb_path})")
                
        md_lines.append("")  # blank line between each tag group

    return "\n".join(md_lines).strip()

def replace_between_markers(readme_content, new_toc_md):
    lines = readme_content.splitlines()
    try:
        start_idx = next(i for i, line in enumerate(lines) if START_MARKER in line)
        end_idx = next(i for i, line in enumerate(lines) if END_MARKER in line)
    except StopIteration:
        raise RuntimeError(f"Could not find both markers ({START_MARKER} / {END_MARKER}) in {README_PATH}")

    if start_idx >= end_idx:
        raise RuntimeError(f"START marker appears after END marker in {README_PATH}")

    # Build the new block (markers included)
    new_block = [START_MARKER] + new_toc_md.splitlines() + [END_MARKER]

    updated = lines[:start_idx] + new_block + lines[end_idx+1:]
    return "\n".join(updated) + "\n"

def main():
    # 1) Find all notebooks in repo
    notebooks = find_notebooks()

    # 2) Build { tag → [list of notebooks] }
    tag_index = build_tag_index(notebooks)

    # 3) Generate the Markdown snippet
    toc_md = generate_markdown(tag_index)

    # 4) Read existing README.md or create new one if it doesn't exist
    try:
        with open(README_PATH, "r", encoding="utf-8") as f:
            readme_text = f.read()
    except FileNotFoundError:
        # Create a new README.md with markers if it doesn't exist
        readme_text = f"""# Project Notebooks
                    {START_MARKER}
                    {END_MARKER}"""
        print(f"Created new {README_PATH} file as it didn't exist.")

    # 5) Inject the new TOC between markers
    updated = replace_between_markers(readme_text, toc_md)

    # 6) Overwrite README.md
    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(updated)

    total_nb = sum(len(lst) for lst in tag_index.values())
    total_tags = len(tag_index)
    print(f"✅ Updated {README_PATH} with {total_nb} notebooks across {total_tags} tags.")

if __name__ == "__main__":
    main()