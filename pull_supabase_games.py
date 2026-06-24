"""
Pull all logged games from Supabase down to local game_*.json files that
human_model.load_decisions() consumes.

Usage (set SUPABASE_URL / SUPABASE_KEY in env first):
  SUPABASE_URL=https://xxxx.supabase.co SUPABASE_KEY=... \
    python3 pull_supabase_games.py --out-dir runs/human_games

Then behavior-clone:
  python3 human_model.py --games-dir runs/human_games \
      --bot-ckpt runs/local_v9c_conserveC2_30/C2c_ep1330.pt
"""
import argparse
import json
from pathlib import Path

from supabase_store import SupabaseGameStore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="runs/human_games")
    args = ap.parse_args()

    store = SupabaseGameStore()
    if not store.enabled:
        raise SystemExit("SUPABASE_URL / SUPABASE_KEY not set in env; nothing to pull.")

    records = store.fetch_all()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    written = human_decisions = usable = human_wins = 0
    for i, rec in enumerate(records):
        if "decisions" not in rec:
            continue
        fname = out / f"game_{i:04d}_{(rec.get('session_id') or 'x')[:8]}.json"
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

    print(f"pulled {written} games -> {out}/")
    print(f"human decisions: {human_decisions}  |  usable for BC (n_legal>1, encoded): {usable}")
    if written:
        print(f"human win-rate in logged games: {human_wins}/{written} = {human_wins/written:.2f}")
    if 0 < usable < 200:
        print(f"NOTE: {usable} usable decisions is small; KL-regularized BC guards against overfitting.")


if __name__ == "__main__":
    main()
