"""TSLOB original VFX textures (white/grey + alpha so particles take any colour).
Flipbooks are square sheets (Grid4x4 or Grid8x8) up to 1024 px, frames left->right, top->bottom.
"""
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates, gaussian_filter
import os, math

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(1337)


class Noise3:
    """Tileable-in-time value noise (cubic interpolated), fbm helper."""

    def __init__(self, seed, size=(64, 64, 32)):
        r = np.random.default_rng(seed)
        self.g = r.random(size).astype(np.float32)
        self.size = size

    def sample(self, x, y, t):
        sx, sy, st = self.size
        x, y, t = np.broadcast_arrays(np.asarray(x, np.float32), np.asarray(y, np.float32), np.asarray(t, np.float32))
        coords = np.array([np.mod(x, sx), np.mod(y, sy), np.mod(t, st)])
        return map_coordinates(self.g, coords.reshape(3, -1), order=3, mode="grid-wrap").reshape(x.shape)

    def fbm(self, x, y, t, octaves=4, lac=2.0, gain=0.5):
        a, f, s, n = 1.0, 1.0, 0.0, 0.0
        for o in range(octaves):
            s += a * self.sample(x * f + o * 7.1, y * f + o * 3.3, t * f + o * 1.7)
            n += a
            a *= gain
            f *= lac
        return s / n


def smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def to_rgba(alpha, lum=None):
    a = np.clip(alpha, 0, 1)
    l = np.ones_like(a) if lum is None else np.broadcast_to(np.clip(lum, 0, 1), a.shape)
    rgb = (l * 255).astype(np.uint8)
    return np.dstack([rgb, rgb, rgb, (a * 255).astype(np.uint8)])


def sheet(frames, grid):
    fh, fw = frames[0].shape[:2]
    out = np.zeros((fh * grid, fw * grid, 4), np.uint8)
    for i, f in enumerate(frames[: grid * grid]):
        r, c = divmod(i, grid)
        out[r * fh:(r + 1) * fh, c * fw:(c + 1) * fw] = f
    return out


def save(name, arr):
    Image.fromarray(arr, "RGBA").save(os.path.join(OUT, name + ".png"), optimize=True)
    print(name, arr.shape)


def grid_uv(w, h):
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    u = (x + 0.5) / w * 2 - 1  # -1..1
    v = 1 - (y + 0.5) / h  # 0 bottom .. 1 top
    return u, v


# ---------------------------------------------------------------------------------------------
# 1. anime flame (8x8, 128 px frames): 2-4 licking tongues from one base, sharp curling tips, holes,
#    flickering heights, tears apart and burns out. soft=True gives a softer energy wisp.
def flame_frames(seed, n=64, S=128, tongues=3, turb=1.0, holes=0.5, soft=False, spread=0.35):
    r = np.random.default_rng(seed)
    N = Noise3(seed)
    N2 = Noise3(seed + 1)
    u, v = grid_uv(S, S)
    px = 2.0 / S
    T = [dict(x0=r.uniform(-spread, spread), L=r.uniform(0.6, 0.98), W=r.uniform(0.22, 0.34), ph=r.uniform(0, 6.3),
              fr=r.uniform(1.5, 3.0), A=r.uniform(0.08, 0.2), w=r.uniform(5, 9)) for _ in range(tongues)]
    T[0].update(x0=0.0, L=0.97, W=0.36)
    frames = []
    for i in range(n):
        life = i / (n - 1)
        t = life * 6.0
        grow = smooth(0.0, 0.16, life)
        die = smooth(0.6, 1.0, life)
        n1 = N.fbm(u * 2.0 + 3, v * 2.6 - t * 2.2, t * 0.5, 4)
        n2 = N2.fbm(u * 4.0, v * 5.0 - t * 3.6, t * 0.8, 3)
        uu = u + (n1 - 0.5) * 0.45 * turb * v
        d = np.full_like(u, -1.0)
        for k in T:
            L = k["L"] * (0.8 + 0.2 * np.sin(t * k["w"] + k["ph"])) * (0.25 + 0.75 * grow) * (1 - 0.55 * die)
            vv = (v - 0.07) / max(L, 1e-3)
            xc = k["x0"] * (1 - 0.5 * vv) + k["A"] * np.sin(vv * k["fr"] * np.pi + k["ph"] - t * 3) * vv ** 1.3
            body = np.clip(1 - vv, 0, 1) ** 1.25 * k["W"] * (1 + 0.6 * np.exp(-((vv - 0.12) / 0.12) ** 2))
            cap = k["W"] * 1.6 * np.sqrt(np.clip(1 - ((v - 0.07) / 0.06) ** 2, 0, 1))
            dk = np.where((vv <= 1) & (vv >= 0), body, np.where(v < 0.07, cap, -1.0)) - np.abs(uu - xc)
            d = np.maximum(d, dk)
        # common rounded base
        base = 0.3 * grow * (1 - 0.4 * die) - np.sqrt((uu / 1.4) ** 2 + ((v - 0.2) / 0.5) ** 2) * 0.75
        d = np.maximum(d, base)
        d += (n2 - 0.5) * 0.12 * turb * np.clip(v * 2, 0, 1)
        # holes + burn out from the top
        hole = N2.fbm(u * 3 + 9, v * 3.2 - t * 2.6, t, 3)
        d = np.minimum(d, (holes * 0.62 + 0.35 - hole) * 0.6 + (1 - v) * 0.25 - die * (0.25 + 0.4 * v))
        if soft:
            a = smooth(-0.06, 0.08, d)
        else:
            a = smooth(-px, px, d - px * 0.5)
        lum = 0.62 + 0.38 * smooth(0.0, 0.1, d)
        a *= (1 - smooth(0.9, 1.0, life)) * (0.25 + 0.75 * smooth(0.0, 0.16, v))
        frames.append(to_rgba(a, lum))
    return frames


# 2. smoke puff (4x4 x 256 = 1024): soft noisy puff, billows, dissipates
def smoke_frames(seed, n=16, S=256, hard=0.0):
    N = Noise3(seed)
    u, v = grid_uv(S, S)
    r = np.sqrt(u * u + (v * 2 - 1) ** 2) * 1.3
    frames = []
    for i in range(n):
        life = i / (n - 1)
        t = life * 4
        f = N.fbm(u * 2.5 + t * 0.3, v * 2.5 - t * 0.6, t * 0.5, 5)
        thr = 0.32 + life * 0.42
        d = (1 - r * (0.85 + 0.25 * life)) * 0.9 + (f - 0.5) * 1.2 - thr * 0.6
        if hard > 0:
            a = smooth(0.0, 0.03 + 0.25 * (1 - hard), d)
        else:
            a = smooth(-0.05, 0.35, d)
        a *= smooth(1.0, 0.75, r)
        lum = 0.65 + 0.35 * f
        frames.append(to_rgba(a, lum))
    return frames


# 3. streak (64 x 512): thin bright line, tapered both ends, soft glow around
def streak():
    W, H = 64, 512
    u, v = grid_uv(W, H)
    taper = np.sin(np.pi * np.clip(v, 0, 1)) ** 0.7
    core = smooth(0.09, 0.0, np.abs(u) / np.maximum(taper, 1e-3) * 0.6)
    glow = np.exp(-(u / (0.32 * np.maximum(taper, 1e-3))) ** 2) * taper * 0.45
    a = np.clip(np.maximum(core * taper, glow), 0, 1)
    return to_rgba(a)


# 4. spark star (4 point) with glow
def star(S=256):
    u, v = grid_uv(S, S)
    v = v * 2 - 1
    ax, ay = np.abs(u), np.abs(v)
    arm = np.maximum(np.exp(-ay * 26 / (1 + ax * 3)) * (1 - ax), np.exp(-ax * 26 / (1 + ay * 3)) * (1 - ay))
    r = np.sqrt(u * u + v * v)
    glow = np.exp(-(r / 0.22) ** 2)
    a = np.clip(np.maximum(arm ** 1.3, glow), 0, 1)
    return to_rgba(a)


def glowdot(S=128, p=2.0):
    u, v = grid_uv(S, S)
    r = np.sqrt(u * u + (v * 2 - 1) ** 2)
    return to_rgba(np.clip(1 - r, 0, 1) ** p)


# 5. electric arcs (4x4 x 256): jagged branching bolt across the frame, white core + glow
def lightning_frames(seed, n=16, S=256):
    r = np.random.default_rng(seed)
    frames = []
    for i in range(n):
        acc = np.zeros((S, S), np.float32)

        def bolt(p0, p1, w, depth):
            pts = [np.array(p0, float), np.array(p1, float)]
            disp = np.linalg.norm(pts[1] - pts[0]) * 0.28
            for _ in range(6):
                new = [pts[0]]
                for a, b in zip(pts[:-1], pts[1:]):
                    m = (a + b) / 2
                    d = b - a
                    nrm = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-6)
                    new += [m + nrm * r.normal(0, disp), b]
                pts = new
                disp *= 0.55
            for a, b in zip(pts[:-1], pts[1:]):
                steps = int(np.linalg.norm(b - a)) + 1
                for s in range(steps):
                    p = a + (b - a) * s / steps
                    x, y = int(p[0]), int(p[1])
                    if 0 <= x < S and 0 <= y < S:
                        acc[y, x] = max(acc[y, x], w)
            if depth < 2:
                for _ in range(r.integers(1, 3)):
                    k = r.integers(len(pts) // 4, len(pts) * 3 // 4)
                    st = pts[k]
                    ang = r.uniform(0, 2 * np.pi)
                    ln = r.uniform(0.15, 0.35) * S
                    bolt(st, st + np.array([np.cos(ang), np.sin(ang)]) * ln, w * 0.6, depth + 1)

        y0 = r.uniform(0.2, 0.8) * S
        y1 = r.uniform(0.2, 0.8) * S
        bolt((4, y0), (S - 4, y1), 1.0, 0)
        core = gaussian_filter(acc, 0.8) * 2.2
        glow = gaussian_filter(acc, 5.0) * 9
        a = np.clip(np.maximum(core, glow * 0.6), 0, 1)
        lum = np.clip(0.75 + core, 0, 1)
        frames.append(to_rgba(a, lum))
    return frames


# 6. ring (thin, bright) and magic circle
def ring(S=512, rad=0.86, w=0.012):
    u, v = grid_uv(S, S)
    v = v * 2 - 1
    r = np.sqrt(u * u + v * v)
    d = np.abs(r - rad)
    a = np.clip(smooth(w * 1.6, w * 0.6, d) + np.exp(-(d / (w * 6)) ** 2) * 0.4, 0, 1)
    return to_rgba(a)


def magic_circle(S=1024):
    u, v = grid_uv(S, S)
    v = v * 2 - 1
    r = np.sqrt(u * u + v * v)
    th = np.arctan2(v, u)
    px = 2.0 / S
    a = np.zeros_like(r)

    def line(d, w):
        return smooth(w + px, w - px, np.abs(d))

    for rad, w in [(0.96, 0.008), (0.9, 0.004), (0.7, 0.006), (0.66, 0.003), (0.34, 0.005)]:
        a = np.maximum(a, line(r - rad, w))
    # tick marks between the outer rings
    ticks = np.abs(((th / (2 * np.pi) * 72) % 1) - 0.5)
    a = np.maximum(a, (r > 0.905) * (r < 0.955) * smooth(0.12, 0.05, ticks))
    # block glyph band (original geometric "runes")
    seg = np.floor((th / (2 * np.pi) + 0.5) * 36)
    g = np.random.default_rng(7).random(40)
    band = (r > 0.74) * (r < 0.86)
    glyph = g[seg.astype(int) % 36]
    sub = ((r - 0.74) / 0.12 * 3).astype(int)
    pat = ((glyph * 8).astype(int) >> np.clip(sub, 0, 2)) & 1
    frac = np.abs(((th / (2 * np.pi) + 0.5) * 36) % 1 - 0.5)
    a = np.maximum(a, band * pat * smooth(0.42, 0.32, frac) * smooth(0.0, 0.004, np.minimum(r - 0.74, 0.86 - r)))
    # inner pattern: radial spokes between the inner rings + a dotted ring (no stars / polygons)
    spokes = np.abs(((th / (2 * np.pi) * 24) % 1) - 0.5)
    a = np.maximum(a, (r > 0.36) * (r < 0.64) * smooth(0.06, 0.02, spokes) * 0.85)
    dots = np.abs(((th / (2 * np.pi) * 48) % 1) - 0.5)
    a = np.maximum(a, smooth(0.012, 0.004, np.abs(r - 0.5)) * smooth(0.25, 0.15, dots))
    a = np.maximum(a, line(r - 0.16, 0.004))
    a = np.maximum(a, np.exp(-((r - 0.96) / 0.05) ** 2) * 0.25)
    return to_rgba(np.clip(a, 0, 1))


# 7. shard / spike: long sharp crystal spike, bright edge, slightly dimmer inside
def shard(W=128, H=512):
    u, v = grid_uv(W, H)
    hw = 0.55 * (1 - v) ** 1.1 * smooth(0.0, 0.06, v)
    d = hw - np.abs(u)
    px = 2.0 / W
    a = smooth(-px, px, d)
    lum = 0.75 + 0.25 * smooth(0.25 * hw.max(), 0, d)
    return to_rgba(a, lum)


# 8. spiky flare flipbook (4x4 x 256): cluster of sharp spikes bursting up from the bottom
def spikeflare_frames(seed, n=16, S=256):
    r = np.random.default_rng(seed)
    u, v = grid_uv(S, S)
    spikes = [(r.uniform(-0.7, 0.7), r.uniform(-0.45, 0.45), r.uniform(0.55, 0.98), r.uniform(0.1, 0.2)) for _ in range(6)]
    frames = []
    px = 2.0 / S
    for i in range(n):
        life = i / (n - 1)
        g = smooth(0, 0.25, life)
        die = smooth(0.55, 1, life)
        a = np.zeros_like(u)
        for x0, lean, ln, w in spikes:
            L = ln * (0.3 + 0.7 * g) * (1 - 0.6 * die)
            vv = v / max(L, 1e-3)
            xc = x0 * 0.75 + lean * v
            hw = w * (1 - np.clip(vv, 0, 1)) ** 1.2 * (vv >= 0)
            hw *= (1 - die * 0.7)
            d = hw - np.abs(u - xc)
            a = np.maximum(a, smooth(-px, px, d) * (vv <= 1))
        a *= 1 - smooth(0.85, 1, life)
        frames.append(to_rgba(a, 0.8 + 0.2 * a))
    return frames


# 9. curling tendril (4x4 x 256) for cursed energy
def tendril_frames(seed, n=16, S=256):
    r = np.random.default_rng(seed)
    frames = []
    curl = r.uniform(3, 5) * (1 if r.random() < 0.5 else -1)
    for i in range(n):
        life = i / (n - 1)
        acc = np.zeros((S, S), np.float32)
        x, y = S * 0.5, S * 0.97
        ang = -np.pi / 2
        steps = int(120 * smooth(0, 0.4, life) + 10)
        for s in range(steps):
            tt = s / 120
            rad = (1 - tt) ** 0.8 * S * 0.15 * (1 - 0.6 * smooth(0.6, 1, life)) + 1
            yy, xx = np.ogrid[0:S, 0:S]
            m = np.clip(rad - np.sqrt((xx - x) ** 2 + (yy - y) ** 2) + 0.5, 0, 1)
            acc = np.maximum(acc, m)
            ang += curl * tt ** 2 * 0.05 + np.sin(tt * 11 + life * 6) * 0.02
            x += np.cos(ang) * S * 0.0072
            y += np.sin(ang) * S * 0.0072
        acc *= 1 - smooth(0.85, 1, life)
        frames.append(to_rgba(acc, 0.85))
    return frames


# 10. water: wavy flowing blob (4x4) + bubble + droplet
def water_frames(seed, n=16, S=256):
    N = Noise3(seed)
    u, v = grid_uv(S, S)
    frames = []
    px = 2.0 / S
    for i in range(n):
        life = i / (n - 1)
        t = life * 5
        f = N.fbm(u * 1.8, v * 1.8 - t * 0.5, t * 0.4, 3)
        vv = v
        prof = 0.45 * np.clip(np.sin(np.pi * np.clip(vv, 0, 1)), 0, 1) ** 0.6 * (1 - 0.5 * life)
        uu = u + np.sin(vv * 6 + t * 2) * 0.12
        d = prof + (f - 0.5) * 0.35 - np.abs(uu)
        a = smooth(-px * 1.5, px * 1.5, d)
        lum = 0.7 + 0.3 * smooth(0, 0.08, d) * (1 - smooth(0.05, 0.2, d)) + 0.2 * smooth(0.18, 0.3, d)
        a *= 1 - smooth(0.8, 1, life)
        frames.append(to_rgba(a, np.clip(lum, 0, 1)))
    return frames


def bubble(S=256):
    u, v = grid_uv(S, S)
    v = v * 2 - 1
    r = np.sqrt(u * u + v * v)
    rim = smooth(0.92, 0.86, r) * smooth(0.6, 0.86, r)
    hl = np.exp(-(((u + 0.35) / 0.18) ** 2 + ((v - 0.4) / 0.12) ** 2)) * (r < 0.9)
    a = np.clip(rim * 0.9 + hl + (r < 0.9) * 0.08, 0, 1)
    return to_rgba(a)


def main():
    save("tslob_flame_8x8", sheet(flame_frames(11, tongues=3, turb=1.0, holes=0.45), 8))
    save("tslob_fire_8x8", sheet(flame_frames(23, tongues=4, turb=1.6, holes=0.8, spread=0.5), 8))
    save("tslob_wisp_8x8", sheet(flame_frames(29, tongues=3, turb=1.3, holes=0.6, soft=True), 8))
    save("tslob_smoke_4x4", sheet(smoke_frames(31), 4))
    save("tslob_inksmoke_4x4", sheet(smoke_frames(37, hard=0.9), 4))
    save("tslob_streak", streak())
    save("tslob_star", star())
    save("tslob_glow", glowdot())
    save("tslob_lightning_4x4", sheet(lightning_frames(41), 4))
    save("tslob_ring", ring())
    save("tslob_circle", magic_circle())
    save("tslob_shard", shard())
    save("tslob_spikes_4x4", sheet(spikeflare_frames(53), 4))
    save("tslob_tendril_4x4", sheet(tendril_frames(61), 4))
    save("tslob_water_4x4", sheet(water_frames(71), 4))
    save("tslob_bubble", bubble())


if __name__ == "__main__":
    main()
