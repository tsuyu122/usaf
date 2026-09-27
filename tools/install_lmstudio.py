"""Install a finished USAF export where LM Studio will find it.

LM Studio reads two shapes out of its models folder: a .gguf file, or a
directory holding the huggingface files. This handles the second, because the
second is what a USAF export actually is - the export writes an ordinary
huggingface model directory, with the trained experts and the trained routers
under the names that architecture declares. Turning that into a .gguf is a
separate step, done by a converter that has to map every expert tensor by
name, and this script does not pretend that step has been taken.

So it copies, and then it checks the copy: the config parses and names the
architecture, the weights open and hold as many tensors as the origin did, and
the tokenizer is there, because a model directory without one still loads in
LM Studio and then answers with gibberish, which is worse than an error.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt")
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "special_tokens_map.json",
)


def survey(src: Path) -> dict:
    """What this directory holds, and everything wrong with it, before copying.

    Every problem is collected rather than raised, because a directory missing
    three files and one missing file are one run to fix and not three, and the
    first one to fail would have hidden the other two.
    """
    if not src.is_dir():
        return {"ok": False, "problems": [f"nao e um diretorio: {src}"],
                "architectures": [], "n_tensors": None, "weights": [],
                "bytes": 0}

    problems = []
    arch, model_type = [], None

    cfg_path = src / "config.json"
    if not cfg_path.is_file():
        problems.append("falta config.json")
    else:
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            arch = cfg.get("architectures") or []
            model_type = cfg.get("model_type")
        except Exception as e:  # noqa: BLE001 - a config that will not read is the finding
            problems.append(f"config.json nao le: {type(e).__name__}: {e}")
        if not arch:
            problems.append("config.json nao declara architectures")

    weights = sorted(f for f in src.iterdir()
                     if f.is_file() and f.suffix in WEIGHT_SUFFIXES)
    if not weights:
        problems.append("nenhum arquivo de pesos (" + ", ".join(WEIGHT_SUFFIXES) + ")")

    n_tensors = None
    if weights and weights[0].suffix == ".safetensors":
        try:
            from safetensors import safe_open

            with safe_open(str(weights[0]), framework="pt") as f:
                n_tensors = len(list(f.keys()))
        except Exception as e:  # noqa: BLE001 - weights that will not open is the finding
            problems.append(f"pesos nao abrem: {type(e).__name__}: {e}")

    if not any((src / f).is_file() for f in TOKENIZER_FILES):
        problems.append("nenhum arquivo de tokenizer")

    nbytes = sum(f.stat().st_size for f in src.iterdir() if f.is_file())
    return {
        "ok": not problems,
        "problems": problems,
        "architectures": arch,
        "model_type": model_type,
        "weights": [f.name for f in weights],
        "n_tensors": n_tensors,
        "bytes": nbytes,
    }


def install(src: Path, name: str, models_root: Path, publisher: str,
             force: bool = False) -> Path:
    """Copy a checked origin into the models folder and return where it went."""
    info = survey(src)
    if not info["ok"]:
        raise SystemExit("o diretorio de origem nao serve:\n  - "
                         + "\n  - ".join(info["problems"]))

    dest = models_root / publisher / name
    if dest.exists():
        if not force:
            raise SystemExit(f"ja existe {dest} - use --force para substituir")
        shutil.rmtree(dest)

    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dest)
    return dest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", type=Path, help="o export do usaf")
    ap.add_argument("name", help="nome do modelo no lm studio")
    ap.add_argument("--models-root", type=Path,
                default=Path.home() / ".lmstudio" / "models",
                help="raiz de modelos do lm studio")
    ap.add_argument("--publisher", default="usaf",
                    help="pasta de autoria, junto com os outros modelos")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)

    info = survey(a.src)
    if not info["ok"]:
        print("o diretorio de origem nao serve:", file=sys.stderr)
        for x in info["problems"]:
            print("  -", x, file=sys.stderr)
        return 1

    arch = ", ".join(info["architectures"])
    n = info["n_tensors"] if info["n_tensors"] is not None else -1
    print(f"origem : {a.src}")
    print(f"        {arch}, {n} tensores, {info['bytes'] / 2 ** 30:.2f} GB")
    print(f"pesos  : {', '.join(info['weights'])}")

    dest = install(a.src, a.name, a.models_root, a.publisher, a.force)

    after = survey(dest)
    if not after["ok"] or after["n_tensors"] != info["n_tensors"]:
        print(f"a copia nao bate com a origem: {after['problems'] or 'n_de_tensores'}",
              file=sys.stderr)
        return 1

    print(f"instalado: {dest}")
    print(f"verificado apos copiar: {after['n_tensors']} tensores, {arch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
