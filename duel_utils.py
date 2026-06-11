CORNER_FIRST_ACTION = {
    (1, 1): 2,
    (30, 30): 0,
}

CHALLENGER_SIDES = (0, 1, 0)
CHALLENGER_CORNERS = ((1, 1), (30, 30), (1, 1))
DEFAULT_DUEL_RUNS = 3

HEAD_TAG_SHORT = {
    "Spacefill": "SF",
    "Greedy": "Gr",
    "Enhanced": "New",
    "Previous": "Old",
    "King": "K",
    "TronBot": "TB",
    "PyTronBot": "Py",
    "Minimax": "MM",
}


def head_tag(name, max_len=3):
    return HEAD_TAG_SHORT.get(name, name[:max_len])


def head_label_font(size=10):
    import pygame
    return pygame.font.SysFont("consolas", size, bold=True)


def blit_head_label(screen, font, px, py, cell, tag, color=(0, 0, 0)):
    s = font.render(tag, True, color)
    screen.blit(s, s.get_rect(center=(px * cell + cell // 2, py * cell + cell // 2)))


def challenger_side(ep):
    return CHALLENGER_SIDES[ep % 3]


def challenger_corner(ep):
    return CHALLENGER_CORNERS[ep % 3]


def corner_first_actions(g):
    if g.step_count != 0:
        return None
    actions = {}
    for pid in (0, 1):
        p = g.get_player(pid)
        if not p.alive:
            continue
        a = CORNER_FIRST_ACTION.get((p.y, p.x))
        if a is None:
            return None
        valid = g.get_valid_actions(pid)
        if not valid:
            actions[pid] = int(p.direction)
        elif a in valid:
            actions[pid] = a
        else:
            actions[pid] = valid[0]
    return actions


def apply_opening_cross(g):
    opening = corner_first_actions(g)
    if opening:
        g.step(opening)
    return opening


def opening_cross_tronenv(env):
    my_a = CORNER_FIRST_ACTION.get(env.my_head)
    opp_a = CORNER_FIRST_ACTION.get(env.opponent_head)
    if my_a is None or opp_a is None:
        return False
    env.step_dual(my_a, opp_a)
    return True
