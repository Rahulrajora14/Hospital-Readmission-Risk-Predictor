"""
Step 1 - Download the raw dataset.

Dataset: "Diabetes 130-US Hospitals for Years 1999-2008" (UCI ML Repository)
~101k hospital encounters of diabetic patients, 50 columns.

We first try the official UCI zip. If that fails (network, firewall),
we fall back to a public GitHub mirror of the same CSV.
"""
import io
import urllib.request
import zipfile

from src import config


def _download_from_uci() -> bytes:
    print(f"  Trying official UCI source...\n  {config.UCI_ZIP_URL}")
    with urllib.request.urlopen(config.UCI_ZIP_URL, timeout=60) as resp:
        zip_bytes = resp.read()
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        # the CSV sits inside a folder in the zip, so search for it by name
        name = next(n for n in zf.namelist() if n.endswith("diabetic_data.csv"))
        return zf.read(name)


def _download_from_mirror() -> bytes:
    print(f"  Trying GitHub mirror...\n  {config.MIRROR_CSV_URL}")
    with urllib.request.urlopen(config.MIRROR_CSV_URL, timeout=60) as resp:
        return resp.read()


def download_dataset(force: bool = False) -> None:
    """Download diabetic_data.csv into data/raw/ (skips if already there)."""
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)

    if config.RAW_CSV.exists() and not force:
        print(f"  Dataset already present: {config.RAW_CSV}")
        return

    for source in (_download_from_uci, _download_from_mirror):
        try:
            csv_bytes = source()
            config.RAW_CSV.write_bytes(csv_bytes)
            print(f"  Saved {len(csv_bytes) / 1e6:.1f} MB -> {config.RAW_CSV}")
            return
        except Exception as err:  # noqa: BLE001 - we just want to try the next source
            print(f"  Failed: {err}")

    raise RuntimeError(
        "Could not download the dataset automatically.\n"
        "Download it manually from "
        "https://archive.ics.uci.edu/dataset/296 and place "
        f"diabetic_data.csv in {config.RAW_DIR}"
    )


if __name__ == "__main__":
    download_dataset()
