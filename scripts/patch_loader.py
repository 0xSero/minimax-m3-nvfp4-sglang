#!/usr/bin/env python3
from pathlib import Path
import sys


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: patch_loader.py /path/to/minimax_m3_vl.py")

    path = Path(sys.argv[1])
    text = path.read_text()
    if 'if name.startswith("model."):' in text:
        return

    old = '''        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            if name.startswith("language_model."):
'''
    new = '''        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            # HF checkpoints may include a root model. prefix; normalize it
            # before dispatch so text self_attn q/k/v does not enter VIT QKV merge.
            if name.startswith("model."):
                name = name[len("model.") :]

            if name.startswith("language_model."):
'''
    if old not in text:
        raise SystemExit(f"MiniMax M3 loader patch anchor not found in {path}")
    path.write_text(text.replace(old, new, 1))


if __name__ == "__main__":
    main()

