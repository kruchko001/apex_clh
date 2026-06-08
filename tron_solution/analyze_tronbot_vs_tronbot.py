import json
import sys
from collections import Counter

DIR = ["UP", "RIGHT", "DOWN", "LEFT"]


def analyze(path):
    moves, ends = [], []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if rec["type"] == "move":
                moves.append(rec)
            else:
                ends.append(rec)

    same = sum(1 for m in moves if m["same_action"])
    print(f"Duels: {len(ends)}  Moves: {len(moves)}")
    print(f"Same action both bots: {same}/{len(moves)} ({100*same/max(len(moves),1):.1f}%)")

    if moves:
        pairs = Counter((m["p0_action_name"], m["p1_action_name"]) for m in moves)
        print(f"Top action pairs: {pairs.most_common(5)}")

    steps = [e["steps"] for e in ends]
    print(f"Steps per duel: min={min(steps)} max={max(steps)} avg={sum(steps)/len(steps):.0f}")
    print(f"Winners: {dict(Counter(e['winner'] for e in ends))}")
    print(f"P0 deaths: {dict(Counter(e.get('p0_death') or 'survived' for e in ends))}")
    print(f"P1 deaths: {dict(Counter(e.get('p1_death') or 'survived' for e in ends))}")

    early = [e for e in ends if e["steps"] <= 10]
    if early:
        print(f"\nShort duels (<=10 steps): {len(early)}")
        for e in early:
            dm = [m for m in moves if m["duel"] == e["duel"]]
            sm = sum(1 for m in dm if m["same_action"])
            print(f"  duel {e['duel']}: {e['steps']} steps, same_action {sm}/{len(dm)}")

    early_moves = [m for m in moves if m["step"] < 8]
    early_same = sum(1 for m in early_moves if m["same_action"])
    print(f"Same action in first 8 moves: {early_same}/{len(early_moves)}")

    ok = early_same == 0 and min(steps) > 10
    print(f"\nEncoding fix verified: {'YES' if ok else 'NO'}")
    return ok


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "tron_solution/tronbot_vs_tronbot.jsonl"
    analyze(path)
