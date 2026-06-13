#!/usr/bin/env python3
import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: patch_model_config.py /path/to/config.json")

    path = Path(sys.argv[1])
    config = json.load(path.open())

    for obj in (config, config.setdefault("text_config", {})):
        layer_types = obj.get("layer_types")
        if isinstance(layer_types, list):
            obj["layer_types"] = [
                "sparse" if value == "minimax_m3_sparse" else value
                for value in layer_types
            ]

    vision_config = config.setdefault("vision_config", {})
    if "rope_theta" not in vision_config:
        vision_config["rope_theta"] = float(
            (vision_config.get("rope_parameters") or {}).get("rope_theta", 10000.0)
        )

    text_config = config.setdefault("text_config", {})
    layer_types = text_config.get("layer_types") or []
    sparse_frequency = [0 if value == "full_attention" else 1 for value in layer_types]

    sparse_config = text_config.setdefault("sparse_attention_config", {})
    sparse_config.update(
        {
            "sparse_attention_freq": sparse_frequency,
            "sparse_num_index_heads": int(
                text_config.get(
                    "index_n_heads",
                    sparse_config.get("sparse_num_index_heads", 4),
                )
            ),
            "sparse_index_dim": int(
                text_config.get(
                    "index_head_dim",
                    sparse_config.get("sparse_index_dim", 128),
                )
            ),
            "sparse_block_size": int(
                text_config.get(
                    "index_block_size",
                    sparse_config.get("sparse_block_size", 128),
                )
            ),
            "sparse_topk_blocks": int(
                text_config.get(
                    "index_topk_blocks",
                    sparse_config.get("sparse_topk_blocks", 16),
                )
            ),
            "sparse_topk": int(
                text_config.get(
                    "index_topk_blocks",
                    sparse_config.get("sparse_topk", 16),
                )
            ),
            "sparse_local_block": int(
                text_config.get(
                    "index_local_blocks",
                    sparse_config.get("sparse_local_block", 1),
                )
            ),
            "sparse_local_blocks": int(
                text_config.get(
                    "index_local_blocks",
                    sparse_config.get("sparse_local_blocks", 1),
                )
            ),
            "sparse_init_tokens": int(
                sparse_config.get(
                    "sparse_init_tokens",
                    text_config.get("index_block_size", 128),
                )
            ),
            "sparse_score_type": sparse_config.get("sparse_score_type", "max"),
        }
    )
    config["sparse_attention_config"] = sparse_config

    json.dump(config, path.open("w"), indent=2)
    path.open("a").write("\n")


if __name__ == "__main__":
    main()

