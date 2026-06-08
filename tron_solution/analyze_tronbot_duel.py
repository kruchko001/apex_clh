import json
import sys
from collections import Counter, defaultdict

DIR = ["UP", "RIGHT", "DOWN", "LEFT"]


def load_records(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def analyze(path):
    moves = []
    ends = []
    for rec in load_records(path):
        if rec.get("type") == "move":
            moves.append(rec)
        elif rec.get("type") == "end":
            ends.append(rec)

    print(f"Duels: {len(ends)}  Moves logged: {len(moves)}")
    results = Counter(e["winner"] for e in ends)
    print(f"Results: {dict(results)}")

    losses = [e for e in ends if e["winner"] == "tronbot"]
    wins = [e for e in ends if e["winner"] == "minimax"]
    print(f"\nMinimax losses ({len(losses)}):")
    print(f"  death cause: {dict(Counter(e.get('minimax_death', '?') for e in losses))}")
    print(f"  avg steps: {sum(e['steps'] for e in losses)/max(len(losses),1):.0f}")

    if losses:
        loss_ids = {e["duel"] for e in losses}
        loss_moves = [m for m in moves if m["duel"] in loss_ids]
        not_top = [m for m in loss_moves if not m.get("mm_chose_top", True)]
        print(f"  non-top-ranked moves in losses: {len(not_top)}/{len(loss_moves)} ({100*len(not_top)/max(len(loss_moves),1):.1f}%)")

        late = [m for m in loss_moves if m["step"] >= m.get("death_step", 999) - 10]
        print(f"  last-10-move stats (losses):")
        print(f"    avg mm_voronoi: {sum(m['mm_voronoi'] for m in late)/max(len(late),1):.1f}")
        print(f"    avg space delta (mm-tb): {sum(m['mm_space']-m['tb_space'] for m in late)/max(len(late),1):.1f}")
        print(f"    partitioned: {sum(1 for m in late if m.get('partitioned'))}/{len(late)}")

        head_on = [m for m in loss_moves if m.get("dist", 99) <= 4]
        print(f"  close-range moves (dist<=4): {len(head_on)}")

    print(f"\nMinimax wins ({len(wins)}):")
    if wins:
        print(f"  avg steps: {sum(e['steps'] for e in wins)/len(wins):.0f}")
        print(f"  tronbot death: {dict(Counter(e.get('tronbot_death', '?') for e in wins))}")

    bad = defaultdict(int)
    for m in moves:
        if m.get("mm_chose_top"):
            continue
        if m["duel"] in {e["duel"] for e in losses}:
            ranked = m.get("mm_ranked", [])
            if ranked:
                best = ranked[0][1]
                bad[f"chose_{DIR[m['minimax_action']]}_over_{DIR[best]}"] += 1
    if bad:
        print(f"\nCommon wrong choices in losses:")
        for k, v in sorted(bad.items(), key=lambda x: -x[1])[:8]:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "tron_solution/tronbot_duel.jsonl"
    analyze(path)
