from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

import maf_ood_notebook_utils as nb


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the seed=42 dual-track MAF-OOD evaluation without Jupyter."
    )
    p.add_argument("--save-root", default=str(Path("~/maf_ood_v51").expanduser()))
    p.add_argument("--data-src", default=str(Path("~/WILD_DATA/splits").expanduser()))
    p.add_argument("--backbones", nargs="+", default=["imagenet_vit", "bioclip"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--force-reextract", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--official-track-i-csv", default=None)
    p.add_argument("--include-approx-track-i", action="store_true")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--tau", type=float, default=0.5)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = nb.Cfg(
        lr=args.lr,
        wd=args.wd,
        epochs=args.epochs,
        bs=args.batch_size,
        nw=args.num_workers,
        lam=args.lam,
        tau=args.tau,
    )

    all_results = {}
    for backbone in args.backbones:
        print(f"\n{'=' * 120}")
        print(f"Running {backbone} / seed=42")
        print(f"{'=' * 120}")

        result = nb.evaluate_backbone_seed42(
            backbone=backbone,
            data_src=args.data_src,
            save_root=args.save_root,
            cfg=cfg,
            device=args.device,
            force_reextract=args.force_reextract,
            eval_only=args.eval_only,
            official_track_i_csv=args.official_track_i_csv,
            include_approx_track_i=args.include_approx_track_i,
        )
        artifact_paths = nb.save_backbone_artifacts(result)
        all_results[backbone] = {
            "result": result,
            "artifacts": artifact_paths,
        }

        print(result["results"].to_string(index=False))
        official_meta = result.get("official_track_i", {})
        if official_meta.get("imported"):
            print(
                "\nImported official Track I rows: "
                f"{official_meta['imported_rows']} from {official_meta['csv']}"
            )
            print("Methods:", ", ".join(official_meta["methods"]))
        elif args.official_track_i_csv:
            print(f"\nNo official Track I rows were imported for backbone={backbone} from {args.official_track_i_csv}")
        print("\nSaved artifacts:")
        for key, value in artifact_paths.items():
            print(f"- {key}: {value}")

        torch.cuda.empty_cache()

    combined_results = pd.concat(
        [all_results[bb]["result"]["results"] for bb in args.backbones],
        ignore_index=True,
    )
    if args.official_track_i_csv:
        imported = nb.load_external_track_i_csv(args.official_track_i_csv)
        extra_rows = imported[~imported["backbone"].isin(args.backbones)].copy()
        if not extra_rows.empty:
            for col in combined_results.columns:
                if col not in extra_rows.columns:
                    extra_rows[col] = ""
            extra_rows = extra_rows[combined_results.columns]
            combined_results = pd.concat([combined_results, extra_rows], ignore_index=True)
            combined_results = combined_results.sort_values(
                by=["AUROC", "FPR95", "AUPR-OUT", "AUTC"],
                ascending=[False, True, False, True],
            ).reset_index(drop=True)
            print(
                "\nIncluded official-only Track I rows in the combined summary: "
                + ", ".join(extra_rows["display_name"].astype(str).tolist())
            )
    combined_alpha = pd.concat(
        [all_results[bb]["result"]["alpha_sweep"] for bb in args.backbones],
        ignore_index=True,
    )

    summary_root = Path(args.save_root).expanduser() / "seed42_dualtrack_summary"
    summary_root.mkdir(parents=True, exist_ok=True)

    combined_results_path = summary_root / "combined_results_seed42.csv"
    combined_alpha_path = summary_root / "combined_alpha_sweep_seed42.csv"
    run_meta_path = summary_root / "run_meta_seed42.json"

    combined_results.to_csv(combined_results_path, index=False)
    combined_alpha.to_csv(combined_alpha_path, index=False)
    run_meta_path.write_text(
        json.dumps(
            {
                "save_root": str(Path(args.save_root).expanduser()),
                "data_src": str(Path(args.data_src).expanduser()),
                "backbones": list(args.backbones),
                "device": args.device,
                "force_reextract": bool(args.force_reextract),
                "eval_only": bool(args.eval_only),
                "official_track_i_csv": args.official_track_i_csv,
                "include_approx_track_i": bool(args.include_approx_track_i),
                "cfg": {
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "num_workers": args.num_workers,
                    "lr": args.lr,
                    "wd": args.wd,
                    "lam": args.lam,
                    "tau": args.tau,
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    print(f"\nCombined results saved to: {combined_results_path}")
    print(f"Combined alpha sweep saved to: {combined_alpha_path}")
    print(f"Run metadata saved to: {run_meta_path}")


if __name__ == "__main__":
    main()
