"""通过 PyTrio checkpoint ID 下载 PEFT LoRA adapter 归档。"""
from __future__ import annotations

import argparse
from pathlib import Path

import pytrio as trio
import requests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_id", help="例如 ckpt_dgwydrq0t4zg")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "weights")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / f"{args.checkpoint_id}.zip"
    rest = trio.ServiceClient().create_rest_client()
    url = rest.get_checkpoint_archive_url(args.checkpoint_id).result().url
    temporary = destination.with_suffix(".zip.part")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with temporary.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)
    temporary.replace(destination)
    print(f"Adapter archive saved to: {destination.resolve()}")


if __name__ == "__main__":
    main()

