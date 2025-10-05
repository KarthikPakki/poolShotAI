import gymnasium as gym
import numpy as np
import pooltool as pt

class PoolToolEnv(gym.Env):
    def __init__(self):
        super(PoolToolEnv, self).__init__()
        
        self.table = pt.Table.default()
        self.balls = pt.get_rack(pt.GameType.EIGHTBALL, table=self.table)
        self.cue = pt.Cue(cue_ball_id="cue")
        self.system = pt.System(cue=self.cue, table=self.table, balls=self.balls)
        
        table_length = self.table.l
        table_width = self.table.w
        
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([table_length, table_width, 10.0], dtype=np.float32),
            dtype=np.float32
        )
        
        self.action_space = gym.spaces.Box(
            low=np.array([0.0, 0.1], dtype=np.float32),
            high=np.array([2*np.pi, 5.0], dtype=np.float32),
            dtype=np.float32
        )

    def _get_obs(self):
        if "cue" not in self.system.balls:
            return np.zeros(3, dtype=np.float32)
        cue_ball = self.system.balls["cue"]
        cue_x = cue_ball.state.rvw[0, 0]
        cue_y = cue_ball.state.rvw[0, 1]
        speed = np.linalg.norm(cue_ball.state.rvw[1])
        return np.array([cue_x, cue_y, speed], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.table = pt.Table.default()
        self.balls = pt.get_rack(pt.GameType.EIGHTBALL, table=self.table)
        self.cue = pt.Cue(cue_ball_id="cue")
        self.system = pt.System(cue=self.cue, table=self.table, balls=self.balls)
        return self._get_obs(), {}

    def step(self, action):
        phi, force = action
        
        self.system.cue.set_state(
            V0=force,
            phi=phi,
            theta=0,
            a=0,
            b=0
        )
        self.system.strike()
        pt.simulate(self.system, inplace=True)

        reward = 0
        terminated = False
        
        pocketed_balls = pt.ruleset.utils.get_pocketed_ball_ids(self.system)
        balls_on_table = [bid for bid in self.system.balls.keys() if bid not in pocketed_balls]
        
        for ball_id in pocketed_balls:
            if ball_id != "cue":
                reward += 1
        
        if "cue" in pocketed_balls:
            reward -= 10
            terminated = True
        
        object_balls_remaining = [b for b in balls_on_table if b != "cue"]
        if len(object_balls_remaining) == 0 and "cue" in balls_on_table:
            reward += 20
            terminated = True
        
        if "cue" not in self.system.balls:
            terminated = True
            reward -= 10

        info = {"pocketed_balls": pocketed_balls, "balls_on_table": balls_on_table}
        return self._get_obs(), reward, terminated, False, info

    def render(self):
        pt.show(self.system)

    def close(self):
        pass