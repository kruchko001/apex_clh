import os
import shutil
import sys
import traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

PASS = 0
FAIL = 0
RESULTS = []


def ok(name):
    global PASS
    PASS += 1
    RESULTS.append(("PASS", name))
    print(f"  PASS  {name}")


def fail(name, err):
    global FAIL
    FAIL += 1
    RESULTS.append(("FAIL", name, str(err)))
    print(f"  FAIL  {name}: {err}")


def run(name, fn):
    try:
        fn()
        ok(name)
    except Exception as e:
        fail(name, e)
        traceback.print_exc()


def test_imports():
    import tron_solution
    import tron_paper
    import tron_paper_dqn
    import tron_solution_dqn
    from tron_solution.env.tron_env import TronEnv
    from tron_paper.env.stationary_env import StationaryTronEnv
    from tron_paper.env.duel_env import MRLDuelEnv
    from competition.tron.tron import TronGame, GameConfig


def test_tron_solution_env():
    from tron_solution.env.tron_env import TronEnv
    from tron_solution.env.frame_stack_wrapper import GridFrameStackWrapper
    env = GridFrameStackWrapper(TronEnv(opponent_type="random"))
    obs, _ = env.reset(seed=0)
    assert "grid" in obs and "valid" in obs
    obs, r, term, trunc, info = env.step(1)
    assert obs["grid"].shape[0] == 16


def test_tron_paper_envs():
    from tron_paper.env.stationary_env import StationaryTronEnv
    from tron_paper.env.duel_env import MRLDuelEnv
    e1 = StationaryTronEnv()
    o1, _ = e1.reset(seed=1)
    assert o1.shape == (5, 32, 32)
    e1.step(1)
    e2 = MRLDuelEnv()
    o2, _ = e2.reset(seed=1)
    assert o2.shape == (5, 32, 32)
    e2.step(0)


def test_tron_paper_dqn_stationary_mask():
    import torch
    from tron_paper.model.phase_torch import extract_stationary_input
    from tron_paper_dqn.model.q_net import MRLQNet
    from tron_paper.model.mrl_net import MRLActorCritic
    obs = torch.zeros(1, 5, 32, 32)
    obs[0, 0, 0, :] = 1.0
    obs[0, 0, -1, :] = 1.0
    obs[0, 0, :, 0] = 1.0
    obs[0, 0, :, -1] = 1.0
    obs[0, 3, 10, 10] = 1.0
    obs[0, 2, 20, 20] = 1.0
    model_in = extract_stationary_input(obs)
    assert model_in.shape == (1, 2, 30, 30)
    assert model_in[0, 0, 19, 19] > 0.5
    assert MRLQNet(stationary=True)(obs).shape == (1, 4)
    assert MRLActorCritic(stationary=True)(obs)[0].shape == (1, 4)


def test_tron_paper_models():
    import torch
    from tron_paper.model.mrl_net import MRLActorCritic
    from tron_paper_dqn.model.q_net import MRLQNet
    x = torch.zeros(1, 5, 32, 32)
    ac = MRLActorCritic(stationary=True)
    logits, v = ac(x)
    assert logits.shape == (1, 4)
    from tron_paper.model.phase_torch import extract_non_stationary_input
    assert extract_non_stationary_input(x).shape == (1, 5, 30, 30)
    ac_ns = MRLActorCritic(stationary=False)
    assert ac_ns(x)[0].shape == (1, 4)
    qn = MRLQNet(stationary=True)
    assert qn(x).shape == (1, 4)


def test_tron_solution_model():
    import torch
    from tron_solution.model.tron_cnn import TronCNN
    from tron_solution.model.obs import INPUT_CHANNELS, PLAY_SIZE
    m = TronCNN(INPUT_CHANNELS, PLAY_SIZE)
    x = torch.zeros(1, INPUT_CHANNELS, PLAY_SIZE, PLAY_SIZE)
    logits, v = m(x)
    assert logits.shape == (1, 4)


def test_tron_paper_ppo_smoke():
    from tron_paper.training.train_stationary import train_stationary
    d = "_test_out/tron_paper"
    if os.path.isdir(d):
        shutil.rmtree(d)
    train_stationary(total_timesteps=200, save_dir=d, n_envs=1, verbose=0)
    assert os.path.isfile(os.path.join(d, "stationary_agent.pt"))


def test_tron_paper_dqn_smoke():
    from tron_paper_dqn.training.train_stationary import train_stationary
    d = "_test_out/tron_paper_dqn"
    if os.path.isdir(d):
        shutil.rmtree(d)
    train_stationary(total_timesteps=200, save_dir=d, n_envs=1, learning_starts=50, verbose=0)
    assert os.path.isfile(os.path.join(d, "stationary_agent.pt"))


def test_tron_solution_dqn_smoke():
    from tron_solution_dqn.training.train_dqn import train as train_dqn
    d = "_test_out/tron_solution_dqn"
    if os.path.isdir(d):
        shutil.rmtree(d)
    train_dqn(
        total_timesteps=200, save_dir=d, n_envs=1, opponent_type="random",
        learning_starts=50, eval_freq=100000, early_stop=False, verbose=0,
    )


def test_tron_solution_ppo_smoke():
    from tron_solution.training.train_ppo import train as train_ppo
    d = "_test_out/tron_solution_ppo"
    if os.path.isdir(d):
        shutil.rmtree(d)
    train_ppo(
        total_timesteps=256, save_dir=d, n_envs=1, opponent_type="random",
        eval_freq=100000, early_stop=False, verbose=0,
    )


def test_tron_paper_export_and_test():
    from tron_paper.export.export_model import export_mrl
    from tron_paper.test.test_model import test_paper_model
    d = "_test_out/tron_paper"
    st = os.path.join(d, "stationary_agent.pt")
    ns = os.path.join(d, "non_stationary_agent.pt")
    if not os.path.isfile(ns):
        shutil.copy(st, ns)
    out = "_test_out/tron_paper_model.pt"
    export_mrl(ns, st, out)
    test_paper_model(out, episodes=2, seed=0)


def test_tron_paper_dqn_export_and_test():
    from tron_paper_dqn.export.export_model import export_mrl_dqn
    from tron_paper.test.test_model import test_paper_model
    d = "_test_out/tron_paper_dqn"
    st = os.path.join(d, "stationary_agent.pt")
    ns = os.path.join(d, "non_stationary_agent.pt")
    if not os.path.isfile(ns):
        shutil.copy(st, ns)
    out = "_test_out/tron_paper_dqn_model.pt"
    export_mrl_dqn(ns, st, out)
    test_paper_model(out, episodes=2, seed=0)


def test_tron_solution_dqn_export_and_test():
    import glob
    from tron_solution_dqn.export.export_model import export_model
    from tron_solution.test.test_model import test_model
    d = "_test_out/tron_solution_dqn"
    zips = glob.glob(os.path.join(d, "tron_dqn_final_*.zip"))
    assert zips, "no dqn checkpoint"
    out = "_test_out/tron_solution_dqn_model.pt"
    export_model(zips[-1], out)
    test_model(out, num_episodes=2)


def test_tron_solution_ppo_export_and_test():
    import glob
    from tron_solution.export.export_model import export_model
    from tron_solution.test.test_model import test_model
    d = "_test_out/tron_solution_ppo"
    zips = glob.glob(os.path.join(d, "tron_ppo_final_*.zip"))
    assert zips, "no ppo checkpoint"
    out = "_test_out/tron_solution_ppo_model.pt"
    export_model(zips[-1], out)
    test_model(out, num_episodes=2)


def test_official_tron_game():
    from competition.tron.tron import TronGame, GameConfig
    g = TronGame(GameConfig(width=32, height=32, max_steps=10, num_players=2))
    g.step({0: 1, 1: 3})


def test_show_stationary_input():
    import subprocess
    r = subprocess.run(
        [sys.executable, "tron_paper/show_stationary_input.py"],
        capture_output=True, text=True, timeout=60,
    )
    assert r.returncode == 0, r.stderr or r.stdout


def test_main_cli_help():
    import subprocess
    for cmd in ["train", "train-dqn", "train-paper", "train-paper-dqn", "export", "export-dqn", "export-paper", "export-paper-dqn", "test-paper", "test-paper-dqn"]:
        r = subprocess.run([sys.executable, "main.py", cmd, "--help"], capture_output=True, text=True)
        assert r.returncode == 0, f"{cmd}: {r.stderr}"


TESTS = [
    ("imports", test_imports),
    ("tron_solution env", test_tron_solution_env),
    ("tron_paper envs", test_tron_paper_envs),
    ("tron_paper models", test_tron_paper_models),
    ("tron_paper_dqn stationary mask", test_tron_paper_dqn_stationary_mask),
    ("tron_solution model", test_tron_solution_model),
    ("official tron game", test_official_tron_game),
    ("tron_paper PPO smoke train", test_tron_paper_ppo_smoke),
    ("tron_paper DQN smoke train", test_tron_paper_dqn_smoke),
    ("tron_solution DQN smoke train", test_tron_solution_dqn_smoke),
    ("tron_solution PPO smoke train", test_tron_solution_ppo_smoke),
    ("tron_paper export+test", test_tron_paper_export_and_test),
    ("tron_paper_dqn export+test", test_tron_paper_dqn_export_and_test),
    ("tron_solution_dqn export+test", test_tron_solution_dqn_export_and_test),
    ("tron_solution PPO export+test", test_tron_solution_ppo_export_and_test),
    ("show_stationary_input", test_show_stationary_input),
    ("main.py CLI help", test_main_cli_help),
]

if __name__ == "__main__":
    os.makedirs("_test_out", exist_ok=True)
    print("=" * 60)
    print("Project test suite")
    print("=" * 60)
    for name, fn in TESTS:
        print(f"\n[{name}]")
        run(name, fn)
    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed, {len(TESTS)} total")
    print("=" * 60)
    if FAIL:
        sys.exit(1)
