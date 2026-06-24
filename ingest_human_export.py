"""
Split a /api/games/export bundle into individual game_*.json files that
human_model.load_decisions() consumes.

Usage:
  python3 ingest_human_export.py runs/human_export.json [--out-dir runs/human_games]

Each record in the export's "games" list is already in the per-game schema the
logger wrote (human_seat + decisions[...]), so we just write them out one file
each. Reports how many human decisions are usable for behavior cloning.
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export_json")
    ap.add_argument("--out-dir", default="runs/human_games")
    args = ap.parse_args()

    with open(args.export_json) as f:
        bundle = json.load(f)
    games = bundle["games"] if isinstance(bundle, dict) and "games" in bundle else bundle
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    written = 0
    human_decisions = usable = 0
    human_wins = 0
    for i, rec in enumerate(games):
        # accept either {decisions, human_seat,...} or already-wrapped records
        if "decisions" not in rec:
            continue
        fname = out / f"game_{i:04d}_{rec.get('session_id', 'x')[:8]}.json"
        with open(fname, "w") as f:
            json.dump(rec, f)
        written += 1
        human_wins += int(rec.get("human_won", 0))
        for d in rec.get("decisions", []):
            if d.get("is_human"):
                human_decisions += 1
                if d.get("chosen_idx", -1) >= 0 and d.get("n_legal", 0) > 1 \
                        and "state" in d and "action_feats" in d:
                    usable += 1

    print(f"wrote {written} games to {out}/")
    print(f"human decisions: {human_decisions}  |  usable for BC (n_legal>1, encoded): {usable}")
    print(f"human win-rate in logged games: {human_wins}/{written}"
          + (f" = {human_wins/written:.2f}" if written else ""))
    if usable < 200:
        print(f"NOTE: {usable} usable decisions is on the small side; expect modest BC signal. "
              f"KL-regularization toward the warm-start checkpoint guards against overfitting.")


if __name__ == "__main__":
    main()
