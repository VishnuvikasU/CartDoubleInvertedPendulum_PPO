"""
PPO — Cart Double Inverted Pendulum  |  Two-Stage Curriculum (FIXED v3)

KEY FIXES vs previous version:
  1. OBS: added sin/cos(h1+h2) world-frame angle for pole 2  [was only relative h2]
  2. OBS_DIM: 11 → 13
  3. Curriculum: W_UPRIGHT2_S1 ramps 0.1→0.5 over Stage 1   [was fixed at 0.1]
  4. Reward: added W_VEL1/W_VEL2 angular velocity penalties  [new — critical]
  5. Removed W_LAST_PEN entirely                             [fights rapid correction]
  6. UPRIGHT_BONUS raised to 3.0, angle tightened to 0.05 rad
  7. STAGE1_THRESHOLD lowered to 900 (easier graduation)
"""

import argparse
import math
import os
from collections import deque

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

import mujoco

# ─────────────────────────────────────────────────────────────────────────────
# Hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────
XML_PATH        = r"E:\Robot Motion Planning and Control\CartDoubleInvertedPendulum\cart_double_inverted_pendulum.xml"

LR              = 3e-4
GAMMA           = 0.99
GAE_LAMBDA      = 0.90
CLIP_EPS        = 0.2
ENTROPY_COEF    = 0.02
VALUE_COEF      = 0.5
MAX_GRAD_NORM   = 0.5
UPDATE_EPOCHS   = 10
MINI_BATCH_SIZE = 64
ROLLOUT_STEPS   = 2048

STAGE1_MAX_STEPS  = 5_000_000
STAGE2_MAX_STEPS  = 8_000_000
STAGE1_THRESHOLD  = 900.0    # FIX: lowered; easier to graduate with richer obs
STAGE2_THRESHOLD  = 1400.0

MAX_EP_STEPS    = 2000
SIM_STEPS       = 4

# ── Reward weights ────────────────────────────────────────────────────────────
W_UPRIGHT1          = 1.0
W_UPRIGHT2          = 1.0
W_UPRIGHT2_S1_START = 0.1    # FIX 3: start of Stage-1 ramp
W_UPRIGHT2_S1_END   = 0.5    # FIX 3: end of Stage-1 ramp (reached at STAGE1_MAX_STEPS)
W_CART_PEN          = 0.05
W_CTRL_PEN          = 0.005
W_VEL1              = 0.02   # FIX 4: penalise pole-1 angular velocity
W_VEL2              = 0.02   # FIX 4: penalise pole-2 angular velocity
# W_LAST_PEN removed entirely (FIX 5)

UPRIGHT_BONUS       = 3.0    # FIX 6: was 2.0
UPRIGHT_BONUS_ANGLE = 0.05   # FIX 6: was 0.15 rad (~8.6°) → now 0.05 rad (~2.9°)

os.makedirs("models", exist_ok=True)
STAGE1_MODEL    = "models/stage1_pole1.pth"
STAGE2_MODEL    = "models/stage2_both.pth"

# FIX 1+2: OBS_DIM increased from 11 to 13
# New obs layout:
#   [cp, cv,
#    sin(h1), cos(h1), h1v,
#    sin(h2), cos(h2), h2v,          ← pole-2 joint angle (relative, unchanged)
#    sin(h1+h2), cos(h1+h2),         ← FIX 1: pole-2 WORLD angle (new!)
#    tip_x, tip_z]                   ← replaced tip_y with tip_z only (more useful)
OBS_DIM  = 12
ACT_DIM  = 1

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────
C = {
    "bg":       (10,  12,  20),
    "panel":    (18,  22,  35),
    "border":   (40,  50,  80),
    "rail":     (60,  70, 160),
    "cart":     (30, 100, 220),
    "pole1":    (0,  200, 200),
    "pole2":    (200,  60, 220),
    "tip":      (255, 230,  50),
    "ghost":    (50,  70,  50),
    "trail":    (80, 130,  80),
    "grid":     (25,  30,  45),
    "text":     (200, 210, 230),
    "text_dim": (100, 110, 130),
    "accent":   (80, 160, 255),
    "green":    (60, 220, 120),
    "yellow":   (240, 200,  50),
    "red":      (220,  70,  60),
    "bar_bg":   (28,  34,  52),
    "s1_col":   (60, 160, 255),
    "s2_col":   (200,  80, 220),
}


# ─────────────────────────────────────────────────────────────────────────────
# Running reward normaliser
# ─────────────────────────────────────────────────────────────────────────────
class RunningMeanStd:
    def __init__(self, eps=1e-4):
        self.mean  = 0.0
        self.var   = 1.0
        self.count = eps

    def update(self, x):
        batch_mean = np.mean(x)
        batch_var  = np.var(x)
        batch_n    = len(x)
        total = self.count + batch_n
        delta = batch_mean - self.mean
        self.mean  = self.mean + delta * batch_n / total
        m_a        = self.var   * self.count
        m_b        = batch_var  * batch_n
        M2         = m_a + m_b + delta**2 * self.count * batch_n / total
        self.var   = M2 / total
        self.count = total

    def normalise(self, x, clip=10.0):
        return np.clip((x - self.mean) / (np.sqrt(self.var) + 1e-8), -clip, clip)


# ─────────────────────────────────────────────────────────────────────────────
# Neural Network — hidden widened to 512 to handle extra obs dimensions
# ─────────────────────────────────────────────────────────────────────────────
class ActorCritic(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden=512):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
        )
        self.actor_mean    = nn.Linear(hidden, act_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))
        self.critic        = nn.Linear(hidden, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)

    def forward(self, obs):
        h     = self.shared(obs)
        mean  = self.actor_mean(h)
        std   = self.actor_log_std.exp().expand_as(mean)
        value = self.critic(h).squeeze(-1)
        return Normal(mean, std), value

    def get_action(self, obs):
        dist, value = self(obs)
        action   = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value

    def evaluate(self, obs, action):
        dist, value = self(obs)
        log_prob = dist.log_prob(action).sum(-1)
        entropy  = dist.entropy().sum(-1)
        return log_prob, value, entropy


# ─────────────────────────────────────────────────────────────────────────────
# MuJoCo Environment
# ─────────────────────────────────────────────────────────────────────────────
class CartDoublePendulumEnv:
    def __init__(self, xml_path, stage=1):
        self.model      = mujoco.MjModel.from_xml_path(xml_path)
        self.data       = mujoco.MjData(self.model)
        self.stage      = stage
        self.step_count = 0
        self.last       = 0.0
        self.last_tip   = 0.0
        # w2_s1 is set externally by train_stage() during Stage 1 ramp
        self.w2_s1      = W_UPRIGHT2_S1_START
        self._sid = {
            n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, n)
            for n in ("cart_pos","cart_vel","hinge1_pos","hinge1_vel",
                      "hinge2_pos","hinge2_vel","tip_pos")
        }

    def _sensor(self, name):
        sid = self._sid[name]
        adr = self.model.sensor_adr[sid]
        dim = self.model.sensor_dim[sid]
        return self.data.sensordata[adr: adr+dim].copy()

    def _get_obs(self):
        cp  = self._sensor("cart_pos")[0]
        cv  = self._sensor("cart_vel")[0]
        h1  = self._sensor("hinge1_pos")[0]
        h1v = self._sensor("hinge1_vel")[0]
        h2  = self._sensor("hinge2_pos")[0]
        h2v = self._sensor("hinge2_vel")[0]
        tip = self._sensor("tip_pos")

        # FIX 1: Compute the world-frame angle of pole 2 and expose
        # sin/cos of it directly. The network can now distinguish between
        # "h2 is large because pole1 is tilted" vs "h2 is large in the world".
        h2_world = h1 + h2

        return np.array([
            cp, cv,
            math.sin(h1), math.cos(h1), h1v,
            math.sin(h2), math.cos(h2), h2v,
            math.sin(h2_world), math.cos(h2_world),  # FIX 1: world angle
            tip[0], tip[2],                           # tip x and z only
        ], dtype=np.float32)

    def _reward(self, action):
        h1    = self._sensor("hinge1_pos")[0]
        h1v   = self._sensor("hinge1_vel")[0]
        h2    = self._sensor("hinge2_pos")[0]
        h2v   = self._sensor("hinge2_vel")[0]
        cx    = self._sensor("cart_pos")[0]
        tip_z = self._sensor("tip_pos")[2]

        # Pole 1 upright reward
        h1_wrap = (h1 + math.pi) % (2 * math.pi) - math.pi
        r_up1   = W_UPRIGHT1 * (1.0 - abs(h1_wrap) / math.pi)

        # Pole 2 upright reward (world frame)
        h2_world_wrap = (h1 + h2 + math.pi) % (2 * math.pi) - math.pi
        w2 = self.w2_s1 if self.stage == 1 else W_UPRIGHT2
        r_up2 = w2 * (1.0 - abs(h2_world_wrap) / math.pi)

        # Upright bonus (FIX 6: tighter angle, larger bonus)
        near_upright1 = abs(h1_wrap) < UPRIGHT_BONUS_ANGLE
        near_upright2 = (True if self.stage == 1
                         else abs(h2_world_wrap) < UPRIGHT_BONUS_ANGLE)
        r_bonus = UPRIGHT_BONUS if (near_upright1 and near_upright2) else 0.0

        # Penalty terms
        r_cart = -W_CART_PEN * cx**2
        r_ctrl = -W_CTRL_PEN * action**2

        # FIX 4: Angular velocity penalties — essential for stable upright
        r_vel  = -W_VEL1 * h1v**2 - W_VEL2 * h2v**2

        # FIX 5: W_LAST_PEN removed — rapid corrective control is needed

        self.last     = action
        self.last_tip = tip_z

        return r_up1 + r_up2 + r_bonus + r_cart + r_ctrl + r_vel

    def _done(self):
        return (abs(self._sensor("cart_pos")[0]) > 1.9 or
                self.step_count >= MAX_EP_STEPS)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        kid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "upright")
        mujoco.mj_resetDataKeyframe(self.model, self.data, kid)
        self.data.qpos[:] += np.random.uniform(-0.05, 0.05, self.data.qpos.shape)
        self.data.qvel[:] += np.random.uniform(-0.05, 0.05, self.data.qvel.shape)
        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0
        self.last       = 0.0
        self.last_tip   = float(self._sensor("tip_pos")[2])
        return self._get_obs()

    def step(self, action):
        self.data.ctrl[0] = float(action[0])
        for _ in range(SIM_STEPS):
            mujoco.mj_step(self.model, self.data)
        self.step_count += 1
        return self._get_obs(), self._reward(float(action[0])), self._done(), {}

    @property
    def state(self):
        return {
            "cart_x": float(self._sensor("cart_pos")[0]),
            "h1":     float(self._sensor("hinge1_pos")[0]),
            "h2":     float(self._sensor("hinge2_pos")[0]),
            "tip":    self._sensor("tip_pos"),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Visualizer (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────
class Visualizer:
    WIN_W, WIN_H = 1280, 720
    SIM_W        = 720
    DASH_X       = 724
    DASH_W       = 556
    SCALE        = 160
    OX           = 360
    OY           = 430
    RAIL_HALF    = 2.0
    L1 = L2      = 0.6
    TRAIL_LEN    = 90

    def __init__(self):
        pygame.init()
        self.screen  = pygame.display.set_mode((self.WIN_W, self.WIN_H))
        pygame.display.set_caption("PPO · Cart Double Inverted Pendulum")
        self.clock   = pygame.time.Clock()
        self.font_lg = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_md = pygame.font.SysFont("monospace", 15)
        self.font_sm = pygame.font.SysFont("monospace", 13)
        self.ep_rewards  = []
        self.mean20_hist = []
        self.tip_trail   = deque(maxlen=self.TRAIL_LEN)
        self.info        = {}

    def _w2s(self, wx, wz):
        return (int(self.OX + wx * self.SCALE),
                int(self.OY - wz * self.SCALE))

    def _rect(self, x, y, w, h, col, radius=4, border=None, border_col=None):
        pygame.draw.rect(self.screen, col, (x, y, w, h), border_radius=radius)
        if border:
            pygame.draw.rect(self.screen, border_col or C["border"],
                             (x, y, w, h), border, border_radius=radius)

    def _text(self, txt, x, y, font=None, col=None):
        font = font or self.font_sm
        col  = col  or C["text"]
        s = font.render(txt, True, col)
        self.screen.blit(s, (x, y))
        return s.get_width(), s.get_height()

    def _hline(self, x0, x1, y, col=None):
        pygame.draw.line(self.screen, col or C["border"], (x0, y), (x1, y))

    def _draw_sim(self, state):
        scr = self.screen
        cx, h1, h2 = state["cart_x"], state["h1"], state["h2"]
        self._rect(0, 0, self.SIM_W, self.WIN_H, C["panel"], radius=0)
        pygame.draw.rect(scr, C["border"], (0, 0, self.SIM_W, self.WIN_H), 1)
        for gz in (-0.6, 0.0, 0.6, 1.2):
            sy = int(self.OY - gz * self.SCALE)
            pygame.draw.line(scr, C["grid"], (0, sy), (self.SIM_W, sy))
            lbl = self.font_sm.render(f"{gz:.1f}m", True, C["border"])
            scr.blit(lbl, (4, sy - 14))
        g0 = self._w2s(0, 0)
        g1 = self._w2s(0, self.L1)
        g2 = self._w2s(0, self.L1 + self.L2)
        pygame.draw.line(scr, C["ghost"], g0, g1, 2)
        pygame.draw.line(scr, C["ghost"], g1, g2, 2)
        pygame.draw.circle(scr, C["ghost"], g2, 5, 1)
        rl = self._w2s(-self.RAIL_HALF, 0)
        rr = self._w2s( self.RAIL_HALF, 0)
        pygame.draw.line(scr, C["rail"], rl, rr, 4)
        for rx in (rl[0], rr[0]):
            pygame.draw.line(scr, C["rail"], (rx, self.OY-10), (rx, self.OY+10), 2)
        pts = list(self.tip_trail)
        for i in range(1, len(pts)):
            a   = i / len(pts)
            col = tuple(int(c * a) for c in C["trail"])
            pygame.draw.line(scr, col, self._w2s(*pts[i-1]), self._w2s(*pts[i]), 2)
        cs = self._w2s(cx, 0)
        cw, ch = int(0.24 * self.SCALE), int(0.10 * self.SCALE)
        self._rect(cs[0]-cw//2, cs[1]-ch//2, cw, ch, C["cart"], border=2)
        for wo in (-cw//3, cw//3):
            pygame.draw.circle(scr, C["border"], (cs[0]+wo, cs[1]+ch//2+6), 7)
            pygame.draw.circle(scr, C["rail"],   (cs[0]+wo, cs[1]+ch//2+6), 4)
        p1x = cx + self.L1 * math.sin(h1)
        p1z =      self.L1 * math.cos(h1)
        p1s = self._w2s(p1x, p1z)
        pygame.draw.line(scr, C["pole1"], cs, p1s, 7)
        pygame.draw.circle(scr, C["pole1"], p1s, 7)
        ah2 = h1 + h2
        p2x = p1x + self.L2 * math.sin(ah2)
        p2z = p1z + self.L2 * math.cos(ah2)
        p2s = self._w2s(p2x, p2z)
        pygame.draw.line(scr, C["pole2"], p1s, p2s, 5)
        pygame.draw.circle(scr, C["pole2"], p2s, 5)
        self.tip_trail.append((p2x, p2z))
        pygame.draw.circle(scr, C["tip"], p2s, 9)
        pygame.draw.circle(scr, (255,255,255), p2s, 4)
        stage   = self.info.get("stage", 1)
        stg_col = C["s1_col"] if stage == 1 else C["s2_col"]
        badge   = f"  STAGE {stage}/2  {'| POLE 1' if stage==1 else '| BOTH POLES'}  "
        self._rect(6, 6, self.font_lg.size(badge)[0]+4, 28, stg_col, radius=5)
        self._text(badge, 8, 10, self.font_lg, C["bg"])
        by  = self.WIN_H - 28
        bx0 = 20; bx1 = self.SIM_W - 20
        mid = (bx0 + bx1) // 2
        self._text("cart", bx0, by-16, col=C["text_dim"])
        pygame.draw.line(scr, C["border"], (bx0, by), (bx1, by), 2)
        pygame.draw.line(scr, C["border"], (mid, by-6), (mid, by+6))
        norm  = np.clip(cx / self.RAIL_HALF, -1, 1)
        cmx   = int(mid + norm * (bx1 - mid - 8))
        self._rect(cmx-7, by-6, 14, 12, C["cart"], radius=3)

    def _gauge(self, cx, cy, r, angle, label, col):
        scr = self.screen
        pygame.draw.circle(scr, C["bar_bg"], (cx, cy), r)
        pygame.draw.circle(scr, C["border"], (cx, cy), r, 1)
        for tick in (0, math.pi/2, -math.pi/2, math.pi):
            tx = cx + int((r-3)*math.sin(tick))
            ty = cy - int((r-3)*math.cos(tick))
            pygame.draw.line(scr, C["border"], (cx, cy), (tx, ty), 1)
        gx = cx + int((r-2)*math.sin(0))
        gy = cy - int((r-2)*math.cos(0))
        pygame.draw.line(scr, C["green"], (cx, cy), (gx, gy), 3)
        nx = cx + int(r * math.sin(angle))
        ny = cy - int(r * math.cos(angle))
        pygame.draw.line(scr, col, (cx, cy), (nx, ny), 3)
        pygame.draw.circle(scr, col, (nx, ny), 5)
        pygame.draw.circle(scr, C["bg"], (cx, cy), 4)
        deg = math.degrees(angle) % 360
        if deg > 180: deg -= 360
        self._text(f"{deg:+.1f}°", cx - 22, cy - r - 18, col=col)
        self._text(label, cx - self.font_sm.size(label)[0]//2, cy + r + 4, col=C["text_dim"])

    def _plot(self, x, y, w, h):
        scr = self.screen
        self._rect(x, y, w, h, C["bar_bg"], radius=4, border=1)
        self._text("Episode Reward", x+6, y+4, col=C["text_dim"])
        data = self.ep_rewards[-200:]
        m20  = self.mean20_hist[-200:]
        if len(data) < 2:
            return
        mn, mx = min(data), max(data)
        span   = (mx - mn) or 1.0
        pad = 22
        def tp(i, v, arr):
            pw = w - 2*pad
            ph = h - 2*pad - 14
            px = x + pad + int(i / (len(arr)-1) * pw)
            py = y + pad + 14 + int((1 - (v - mn)/span) * ph)
            return px, py
        zy = y + pad + 14 + int((1 - (0-mn)/span) * (h-2*pad-14))
        pygame.draw.line(scr, C["border"], (x+pad, zy), (x+pad+w-2*pad, zy))
        pts = [tp(i, v, data) for i,v in enumerate(data)]
        if len(pts) >= 2:
            pygame.draw.lines(scr, C["accent"], False, pts, 1)
        if len(m20) >= 2:
            mp = [tp(i, v, m20) for i,v in enumerate(m20)]
            pygame.draw.lines(scr, C["green"], False, mp, 2)
        self._text(f"last {data[-1]:.1f}", x+w-90, y+4, col=C["accent"])
        if m20:
            self._text(f"M20 {m20[-1]:.1f}", x+w-80, y+18, col=C["green"])

    def _draw_dash(self, state):
        scr  = self.screen
        dx   = self.DASH_X
        dw   = self.DASH_W
        cx   = dx + 8
        cy   = 12
        self._rect(dx, 0, dw, self.WIN_H, C["panel"], radius=0)
        pygame.draw.rect(scr, C["border"], (dx, 0, dw, self.WIN_H), 1)
        info      = self.info
        stage     = info.get("stage", 1)
        total     = info.get("total_steps", 0)
        ep        = info.get("episode", 0)
        rew       = info.get("ep_reward", 0.0)
        m20       = info.get("mean20", 0.0)
        act       = info.get("action", 0.0)
        threshold = STAGE1_THRESHOLD if stage == 1 else STAGE2_THRESHOLD
        max_s     = STAGE1_MAX_STEPS  if stage == 1 else STAGE2_MAX_STEPS
        stg_col   = C["s1_col"] if stage == 1 else C["s2_col"]
        self._text("PPO  TRAINING DASHBOARD", cx, cy, self.font_lg, C["text"])
        cy += 28
        self._hline(dx+4, dx+dw-4, cy); cy += 10
        self._text(
            f"Stage {stage}/2  —  {'pole 1 only' if stage==1 else 'both poles upright'}",
            cx, cy, self.font_md, stg_col)
        cy += 22
        bw = dw - 20; bh = 14
        prog = min(m20 / threshold, 1.0)
        self._rect(cx, cy, bw, bh, C["bar_bg"], radius=6, border=1)
        if prog > 0:
            self._rect(cx, cy, int(bw*prog), bh, stg_col, radius=6)
        pygame.draw.rect(scr, C["border"], (cx, cy, bw, bh), 1, border_radius=6)
        self._text(f"mean20: {m20:.1f} / {threshold:.0f}  ({prog*100:.0f}%)",
                   cx, cy+bh+2, col=C["text_dim"])
        cy += bh + 22
        self._hline(dx+4, dx+dw-4, cy); cy += 10
        stats = [
            ("Total Steps", f"{total:,}"),
            ("Episodes",    f"{ep:,}"),
            ("Ep Reward",   f"{rew:+.2f}"),
            ("Mean-20 Rew", f"{m20:.2f}"),
            ("Step Budget", f"{total/max_s*100:.1f}%"),
            ("Threshold",   f"{threshold:.0f}"),
        ]
        col_w = (dw - 24) // 2
        for i, (k, v) in enumerate(stats):
            rx = cx + (i%2) * (col_w + 8)
            ry = cy + (i//2) * 38
            self._rect(rx, ry, col_w, 34, C["bar_bg"], radius=4)
            self._text(k, rx+5, ry+3,  col=C["text_dim"])
            self._text(v, rx+5, ry+16, self.font_md, C["text"])
        cy += (math.ceil(len(stats)/2)) * 38 + 10
        self._hline(dx+4, dx+dw-4, cy); cy += 8
        ph = 128
        self._plot(dx+6, cy, dw-12, ph)
        cy += ph + 12
        self._hline(dx+4, dx+dw-4, cy); cy += 8
        self._text("Pole Angles  (green tick = upright goal)", cx, cy, col=C["text_dim"])
        cy += 18
        rg  = 44
        g1x = dx + dw//3
        g2x = dx + 2*dw//3
        gcy = cy + rg + 18
        self._gauge(g1x, gcy, rg, state["h1"], "hinge1", C["pole1"])
        self._gauge(g2x, gcy, rg, state["h1"]+state["h2"], "pole2 abs", C["pole2"])
        cy += rg*2 + 38
        self._hline(dx+4, dx+dw-4, cy); cy += 8
        self._text("Cart Velocity Command", cx, cy, col=C["text_dim"]); cy += 18
        bw = dw - 20; bh = 20
        self._rect(cx, cy, bw, bh, C["bar_bg"], radius=4, border=1)
        mid   = cx + bw//2
        norm  = float(np.clip(act/3.0, -1, 1))
        apx   = int(bw/2 * norm)
        acol  = C["green"] if abs(norm)<0.5 else (C["yellow"] if abs(norm)<0.8 else C["red"])
        if apx >= 0:
            self._rect(mid, cy+2, apx, bh-4, acol, radius=2)
        else:
            self._rect(mid+apx, cy+2, -apx, bh-4, acol, radius=2)
        pygame.draw.line(scr, C["text_dim"], (mid, cy), (mid, cy+bh))
        self._text(f"{act:+.3f} m/s", cx + bw//2 - 30, cy+3, col=C["text"])
        cy += bh + 12
        tip_z = float(state["tip"][2])
        frac  = float(np.clip(tip_z/1.2, 0, 1))
        self._text(f"Tip height: {tip_z:.3f} m  (max 1.2 m)", cx, cy, col=C["text_dim"])
        cy += 16
        bw = dw - 20; bh = 14
        self._rect(cx, cy, bw, bh, C["bar_bg"], radius=4, border=1)
        hcol = C["green"] if frac>0.8 else (C["yellow"] if frac>0.4 else C["red"])
        if frac > 0:
            self._rect(cx, cy, int(bw*frac), bh, hcol, radius=4)
        cy += bh + 8
        fps = self.clock.get_fps()
        self._text(f"render {fps:.0f} fps", dx + dw - 90, self.WIN_H - 20, col=C["text_dim"])

    def update(self, state, info):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
        self.info = info
        if info.get("ep_just_done"):
            self.ep_rewards.append(info["ep_reward_done"])
            self.mean20_hist.append(info["mean20"])
        self.screen.fill(C["bg"])
        self._draw_sim(state)
        self._draw_dash(state)
        pygame.display.flip()
        self.clock.tick(0)

    def close(self):
        pygame.quit()


# ─────────────────────────────────────────────────────────────────────────────
# GAE + PPO
# ─────────────────────────────────────────────────────────────────────────────
def compute_gae(rewards, values, dones, last_val):
    n   = len(rewards)
    adv = np.zeros(n, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(n)):
        nv   = last_val if t == n-1 else values[t+1]
        mask = 1.0 - float(dones[t])
        d    = rewards[t] + GAMMA * nv * mask - values[t]
        gae  = d + GAMMA * GAE_LAMBDA * mask * gae
        adv[t] = gae
    return adv, adv + values


def ppo_update(policy, optimizer, rollout, lr_now=None):
    if lr_now is not None:
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

    obs_t    = torch.FloatTensor(rollout["obs"])
    act_t    = torch.FloatTensor(rollout["actions"])
    logp_old = torch.FloatTensor(rollout["log_probs"])
    adv_t    = torch.FloatTensor(rollout["advantages"])
    ret_t    = torch.FloatTensor(rollout["returns"])

    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    n = len(obs_t)

    for _ in range(UPDATE_EPOCHS):
        idx = torch.randperm(n)
        for s in range(0, n, MINI_BATCH_SIZE):
            mb              = idx[s: s+MINI_BATCH_SIZE]
            lp, v, entropy  = policy.evaluate(obs_t[mb], act_t[mb])
            ratio           = (lp - logp_old[mb]).exp()
            surr            = torch.min(ratio * adv_t[mb],
                                        ratio.clamp(1-CLIP_EPS, 1+CLIP_EPS) * adv_t[mb])
            loss = (-surr.mean()
                    + VALUE_COEF * (ret_t[mb] - v).pow(2).mean()
                    - ENTROPY_COEF * entropy.mean())
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
            optimizer.step()


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────
def train_stage(stage, policy, optimizer, xml_path,
                max_steps, threshold, save_path, viz):

    env         = CartDoublePendulumEnv(xml_path, stage=stage)
    obs         = env.reset()
    ep_reward   = 0.0
    episode     = 0
    total_steps = 0
    recent      = deque(maxlen=20)
    last_action = 0.0

    rms = RunningMeanStd()
    bufs = {"obs":[], "act":[], "logp":[], "rew":[], "val":[], "done":[]}

    print(f"\n{'='*60}")
    print(f"  STAGE {stage}  —  {'Pole 1 only (ramping pole-2)' if stage==1 else 'Both poles upright'}")
    print(f"{'='*60}")

    while total_steps < max_steps:

        # FIX 3: Ramp W_UPRIGHT2_S1 linearly over Stage 1 training
        if stage == 1:
            ramp_frac = min(total_steps / max_steps, 1.0)
            env.w2_s1 = (W_UPRIGHT2_S1_START
                         + (W_UPRIGHT2_S1_END - W_UPRIGHT2_S1_START) * ramp_frac)

        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            action, logp, value = policy.get_action(obs_t)

        action_np   = action.numpy()[0]
        last_action = float(action_np[0])

        next_obs, reward, done, _ = env.step(action_np)
        ep_reward   += reward
        total_steps += 1

        bufs["obs"].append(obs)
        bufs["act"].append(action_np)
        bufs["logp"].append(logp.item())
        bufs["rew"].append(reward)
        bufs["val"].append(value.item())
        bufs["done"].append(float(done))

        obs = next_obs

        ep_just_done    = False
        ep_reward_done  = 0.0

        if done:
            episode       += 1
            recent.append(ep_reward)
            ep_just_done   = True
            ep_reward_done = ep_reward
            mean20         = float(np.mean(recent))

            if episode % 10 == 0:
                w2_now = env.w2_s1 if stage == 1 else W_UPRIGHT2
                print(f"  S{stage} | Ep {episode:5d} | Steps {total_steps:8,} "
                      f"| EpRew {ep_reward:8.1f} | Mean20 {mean20:8.1f}"
                      f"  [w2={w2_now:.2f}]")

            ep_reward = 0.0
            obs       = env.reset()

        if len(bufs["obs"]) >= ROLLOUT_STEPS:
            raw_rew = np.array(bufs["rew"])
            rms.update(raw_rew)
            norm_rew = rms.normalise(raw_rew)

            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                _, _, lv = policy.get_action(obs_t)

            adv, ret = compute_gae(
                norm_rew, np.array(bufs["val"]),
                np.array(bufs["done"]), lv.item())

            lr_now = LR * (1.0 - total_steps / max_steps)
            lr_now = max(lr_now, LR * 0.05)

            ppo_update(policy, optimizer, {
                "obs":        np.array(bufs["obs"]),
                "actions":    np.array(bufs["act"]),
                "log_probs":  np.array(bufs["logp"]),
                "advantages": adv,
                "returns":    ret,
            }, lr_now=lr_now)

            for k in bufs: bufs[k].clear()

        if viz is not None:
            mean20 = float(np.mean(recent)) if recent else 0.0
            viz.update(env.state, {
                "stage":          stage,
                "total_steps":    total_steps,
                "episode":        episode,
                "ep_reward":      ep_reward,
                "ep_reward_done": ep_reward_done,
                "mean20":         mean20,
                "action":         last_action,
                "ep_just_done":   ep_just_done,
            })

        if len(recent) == 20 and np.mean(recent) >= threshold:
            print(f"\n  ✓ Stage {stage} GRADUATED!  "
                  f"Mean20={np.mean(recent):.1f} >= {threshold}")
            break

    torch.save(policy.state_dict(), save_path)
    print(f"  Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-render",   action="store_true")
    ap.add_argument("--skip-stage1", action="store_true")
    ap.add_argument("--xml",         default=XML_PATH)
    args = ap.parse_args()

    viz = None if args.no_render else Visualizer()

    policy    = ActorCritic()
    optimizer = optim.Adam(policy.parameters(), lr=LR)

    if args.skip_stage1 and os.path.exists(STAGE1_MODEL):
        policy.load_state_dict(torch.load(STAGE1_MODEL, weights_only=True))
        print(f"Loaded Stage-1 weights from {STAGE1_MODEL}")
    else:
        train_stage(1, policy, optimizer, args.xml,
                    STAGE1_MAX_STEPS, STAGE1_THRESHOLD, STAGE1_MODEL, viz)

    for pg in optimizer.param_groups:
        pg["lr"] = LR * 0.5
    if viz:
        viz.tip_trail.clear()

    train_stage(2, policy, optimizer, args.xml,
                STAGE2_MAX_STEPS, STAGE2_THRESHOLD, STAGE2_MODEL, viz)

    if viz:
        viz.close()

    print("\nAll stages complete.")


if __name__ == "__main__":
    main()