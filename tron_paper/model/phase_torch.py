import torch


@torch.jit.script
def phase_separated(obs: torch.Tensor) -> torch.Tensor:
    h = obs.size(2)
    w = obs.size(3)
    blocked = obs[0, 0] > 0.5
    for c in range(1, 5):
        blocked = blocked | (obs[0, c] > 0.5)

    my_y = 0
    my_x = 0
    opp_y = 0
    opp_x = 0
    found_my = False
    found_opp = False
    for y in range(h):
        for x in range(w):
            if obs[0, 3, y, x] > 0.5:
                my_y = y
                my_x = x
                found_my = True
            if obs[0, 4, y, x] > 0.5:
                opp_y = y
                opp_x = x
                found_opp = True

    if not found_my or not found_opp:
        return torch.zeros((), dtype=torch.float32)

    seen_my = torch.zeros((h, w), dtype=torch.bool)
    seen_opp = torch.zeros((h, w), dtype=torch.bool)
    my_front = torch.zeros((h, w), dtype=torch.bool)
    opp_front = torch.zeros((h, w), dtype=torch.bool)

    if my_y > 0 and not blocked[my_y - 1, my_x]:
        my_front[my_y - 1, my_x] = True
    if my_y + 1 < h and not blocked[my_y + 1, my_x]:
        my_front[my_y + 1, my_x] = True
    if my_x > 0 and not blocked[my_y, my_x - 1]:
        my_front[my_y, my_x - 1] = True
    if my_x + 1 < w and not blocked[my_y, my_x + 1]:
        my_front[my_y, my_x + 1] = True

    if opp_y > 0 and not blocked[opp_y - 1, opp_x]:
        opp_front[opp_y - 1, opp_x] = True
    if opp_y + 1 < h and not blocked[opp_y + 1, opp_x]:
        opp_front[opp_y + 1, opp_x] = True
    if opp_x > 0 and not blocked[opp_y, opp_x - 1]:
        opp_front[opp_y, opp_x - 1] = True
    if opp_x + 1 < w and not blocked[opp_y, opp_x + 1]:
        opp_front[opp_y, opp_x + 1] = True

    for _ in range(h * w):
        if not my_front.any() and not opp_front.any():
            break
        seen_my = seen_my | my_front
        seen_opp = seen_opp | opp_front
        new_my = torch.zeros((h, w), dtype=torch.bool)
        new_opp = torch.zeros((h, w), dtype=torch.bool)
        for y in range(h):
            for x in range(w):
                if my_front[y, x]:
                    if y > 0 and not blocked[y - 1, x] and not seen_my[y - 1, x]:
                        new_my[y - 1, x] = True
                    if y + 1 < h and not blocked[y + 1, x] and not seen_my[y + 1, x]:
                        new_my[y + 1, x] = True
                    if x > 0 and not blocked[y, x - 1] and not seen_my[y, x - 1]:
                        new_my[y, x - 1] = True
                    if x + 1 < w and not blocked[y, x + 1] and not seen_my[y, x + 1]:
                        new_my[y, x + 1] = True
                if opp_front[y, x]:
                    if y > 0 and not blocked[y - 1, x] and not seen_opp[y - 1, x]:
                        new_opp[y - 1, x] = True
                    if y + 1 < h and not blocked[y + 1, x] and not seen_opp[y + 1, x]:
                        new_opp[y + 1, x] = True
                    if x > 0 and not blocked[y, x - 1] and not seen_opp[y, x - 1]:
                        new_opp[y, x - 1] = True
                    if x + 1 < w and not blocked[y, x + 1] and not seen_opp[y, x + 1]:
                        new_opp[y, x + 1] = True
        my_front = new_my
        opp_front = new_opp

    overlap = (seen_my & seen_opp).any()
    if overlap:
        return torch.zeros((), dtype=torch.float32)
    return torch.ones((), dtype=torch.float32)
