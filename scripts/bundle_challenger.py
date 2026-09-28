"""Package current local experiment code/data for Colab without publishing it."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / "colab_bundle" / "challenger.zip"
    output.parent.mkdir(exist_ok=True)
    paths = [ROOT / name for name in ("pyproject.toml", "uv.lock", "README.md", "LICENSE", "data/grampa.csv")]
    paths += sorted((ROOT / "src").rglob("*.py"))
    paths += [ROOT / "docs/CHALLENGER_EXPERIMENT.md"]
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(ROOT))
    print(f"Colab upload bundle: {output} ({output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
