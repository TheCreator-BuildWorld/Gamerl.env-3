# Gamerl.env-3
Game rl . env
import random
import json

# ─────────────────────────────────────────────
#  Grid World RL Environment – OpenEnv Hackathon
#  Problem: Mini-Game RL Environment
# ─────────────────────────────────────────────

# Actions
UP    = 0
DOWN  = 1
LEFT  = 2
RIGHT = 3

# Difficulty config
DIFFICULTY_CONFIG = {
    "easy":   {"grid_size": 5,  "num_obstacles": 2,  "max_steps": 50},
    "medium": {"grid_size": 8,  "num_obstacles": 6,  "max_steps": 100},
    "hard":   {"grid_size": 12, "num_obstacles": 15, "max_steps": 200},
}

# ─────────────────────────────────────────────
#  Environment Class
# ─────────────────────────────────────────────

class GridWorldEnv:
    def __init__(self, difficulty="easy"):
        assert difficulty in DIFFICULTY_CONFIG, f"difficulty must be one of {list(DIFFICULTY_CONFIG.keys())}"
        config = DIFFICULTY_CONFIG[difficulty]

        self.grid_size     = config["grid_size"]
        self.num_obstacles = config["num_obstacles"]
        self.max_steps     = config["max_steps"]
        self.difficulty    = difficulty

        # Will be initialised by reset()
        self.agent_pos   = None
        self.goal_pos    = None
        self.obstacles   = []
        self.current_step = 0
        self.done        = False
        self.total_reward = 0.0

        self.reset()

    # ── Helpers ──────────────────────────────

    def _random_pos(self, exclude=None):
        """Return a random (row, col) not in exclude list."""
        exclude = exclude or []
        while True:
            pos = (random.randint(0, self.grid_size - 1),
                   random.randint(0, self.grid_size - 1))
            if pos not in exclude:
                return pos

    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # ── OpenEnv API ───────────────────────────

    def reset(self):
        """
        Reset the environment to a fresh episode.
        Returns the initial state dict.
        """
        self.current_step = 0
        self.done         = False
        self.total_reward = 0.0

        taken = []

        self.agent_pos = self._random_pos(taken)
        taken.append(self.agent_pos)

        # Ensure goal is reachable (at least 2 steps away)
        while True:
            self.goal_pos = self._random_pos(taken)
            if self._manhattan(self.agent_pos, self.goal_pos) >= 2:
                break
        taken.append(self.goal_pos)

        self.obstacles = []
        for _ in range(self.num_obstacles):
            obs = self._random_pos(taken)
            self.obstacles.append(obs)
            taken.append(obs)

        return self.state()

    def step(self, action):
        """
        Apply action and return (state, reward, done, info).

        Actions:
            0 = UP, 1 = DOWN, 2 = LEFT, 3 = RIGHT
        """
        assert not self.done, "Episode is done. Call reset() first."
        assert action in (UP, DOWN, LEFT, RIGHT), f"Invalid action {action}"

        self.current_step += 1

        row, col = self.agent_pos
        if action == UP:    row -= 1
        elif action == DOWN: row += 1
        elif action == LEFT: col -= 1
        elif action == RIGHT:col += 1

        # Clamp to grid boundaries (wall bounce = no move)
        row = max(0, min(self.grid_size - 1, row))
        col = max(0, min(self.grid_size - 1, col))
        new_pos = (row, col)

        # ── Reward logic ──────────────────────
        reward = -0.1  # small step penalty → encourages speed

        if new_pos in self.obstacles:
            # Hit obstacle → stay in place, penalty
            reward = -5.0
            new_pos = self.agent_pos
        elif new_pos == self.goal_pos:
            # Reached goal → big positive reward
            reward = +10.0
            self.done = True
        
        self.agent_pos = new_pos

        # Timeout
        if self.current_step >= self.max_steps:
            self.done = True
            reward -= 2.0  # extra penalty for timing out

        self.total_reward += reward

        info = {
            "step":         self.current_step,
            "total_reward": round(self.total_reward, 3),
            "difficulty":   self.difficulty,
        }

        return self.state(), reward, self.done, info

    def state(self):
        """
        Return the current environment state as a dict.
        """
        return {
            "agent_pos":    list(self.agent_pos),
            "goal_pos":     list(self.goal_pos),
            "obstacles":    [list(o) for o in self.obstacles],
            "grid_size":    self.grid_size,
            "step":         self.current_step,
            "max_steps":    self.max_steps,
            "done":         self.done,
            "difficulty":   self.difficulty,
        }

    def render(self):
        """
        Print a human-readable ASCII grid (optional, for debugging).
        """
        print(f"\n── Step {self.current_step}/{self.max_steps}  |  Difficulty: {self.difficulty} ──")
        for r in range(self.grid_size):
            row_str = ""
            for c in range(self.grid_size):
                pos = (r, c)
                if pos == self.agent_pos:
                    row_str += " A "
                elif pos == self.goal_pos:
                    row_str += " G "
                elif pos in self.obstacles:
                    row_str += " X "
                else:
                    row_str += " . "
            print(row_str)
        print(f"Agent: {self.agent_pos}  Goal: {self.goal_pos}  Reward so far: {round(self.total_reward,2)}\n")


# ─────────────────────────────────────────────
#  Graders
# ─────────────────────────────────────────────

def grader_goal_reached(env, actions):
    """Grader 1 – Did the agent reach the goal?"""
    obs = env.reset()
    done = False
    for a in actions:
        if done:
            break
        obs, reward, done, info = env.step(a)
    reached = obs["agent_pos"] == obs["goal_pos"]
    return {
        "grader": "goal_reached",
        "passed": reached,
        "score":  1.0 if reached else 0.0,
    }

def grader_within_steps(env, actions):
    """Grader 2 – Did the agent finish within max_steps?"""
    obs = env.reset()
    done = False
    for a in actions:
        if done:
            break
        obs, reward, done, info = env.step(a)
    within = info["step"] <= env.max_steps
    return {
        "grader": "within_steps",
        "passed": within,
        "score":  1.0 if within else 0.0,
    }

def grader_no_errors(env):
    """Grader 3 – Does the environment run without errors?"""
    try:
        obs = env.reset()
        assert isinstance(obs, dict)
        obs, reward, done, info = env.step(DOWN)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        return {"grader": "no_errors", "passed": True, "score": 1.0}
    except Exception as e:
        return {"grader": "no_errors", "passed": False, "score": 0.0, "error": str(e)}

def grader_reward_positive(env, num_episodes=5):
    """Grader 4 – Does reward improve (agent gets positive reward at least once)?"""
    any_positive = False
    for _ in range(num_episodes):
        env.reset()
        # Simple greedy agent moving toward goal
        for _ in range(env.max_steps):
            s = env.state()
            ar, ac = s["agent_pos"]
            gr, gc = s["goal_pos"]
            if ar < gr:   action = DOWN
            elif ar > gr: action = UP
            elif ac < gc: action = RIGHT
            else:         action = LEFT
            _, reward, done, _ = env.step(action)
            if reward > 0:
                any_positive = True
            if done:
                break
    return {
        "grader": "reward_positive",
        "passed": any_positive,
        "score":  1.0 if any_positive else 0.0,
    }

def run_all_graders(difficulty="easy"):
    """Run all 4 graders and print results."""
    env = GridWorldEnv(difficulty=difficulty)

    # Simple action sequence: move right and down repeatedly
    actions = [RIGHT, DOWN] * 50

    results = [
        grader_no_errors(env),
        grader_goal_reached(env, actions),
        grader_within_steps(env, actions),
        grader_reward_positive(env),
    ]

    print("\n══════════════ GRADER RESULTS ══════════════")
    total = 0
    for r in results:
        status = "✅ PASS" if r["passed"] else "❌ FAIL"
        print(f"  {status}  |  {r['grader']}  |  score: {r['score']}")
        total += r["score"]
    print(f"\n  Overall Score: {total}/{len(results)}")
    print("════════════════════════════════════════════\n")
    return results


# ─────────────────────────────────────────────
#  Quick Demo
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Grid World RL Environment – Demo ===\n")

    for diff in ["easy", "medium", "hard"]:
        print(f"\n{'='*40}")
        print(f"  Difficulty: {diff.upper()}")
        print(f"{'='*40}")
        env = GridWorldEnv(difficulty=diff)
        env.render()

        # Take 5 random steps
        for _ in range(5):
            action = random.choice([UP, DOWN, LEFT, RIGHT])
            state, reward, done, info = env.step(action)
            action_name = ["UP","DOWN","LEFT","RIGHT"][action]
            print(f"Action: {action_name:6s} | Reward: {reward:+.1f} | Done: {done}")
            if done:
                print("  → Episode finished!")
                break

    print("\n\n=== Running Graders ===")
    run_all_graders("easy")
    run_all_graders("medium")
    
