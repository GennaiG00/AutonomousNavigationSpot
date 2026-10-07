#!/usr/bin/env python3
"""
Visualizzatore dei dati di missione (2026-10-06). Gira su qualunque computer: servono solo
numpy e matplotlib, NON l'SDK di Spot.

Durante la missione il robot non disegna piu' figure (SAVE_FIGURES_DURING_MISSION = False in
easy_walk.py): salva i dati grezzi, e le figure si fanno qui, dopo.

    python3 mission_viewer.py                                  # (o tasto Run) l'ultima missione in MissionMap/
    python3 mission_viewer.py <cartella_missione>              # interattivo, tutte le scansioni
    python3 mission_viewer.py <file.npz>                       # una scansione o una vista PRM
    python3 mission_viewer.py <cartella_missione> --png <dir>  # esporta tutte le figure in PNG
    python3 mission_viewer.py <cartella_missione> --overview   # solo il riepilogo della missione

Tasti (modalita' interattiva):
    A sinistra la scansione (layer a scelta), a destra la MAPPA GLOBALE in quel momento:
    mosaico delle scansioni fatte fin li' (rosso ostacolo, verde chiaro libero, bianco mai visto),
    celle di missione (verde visitata, rossa bloccata), grafo PRM colorato per pendenza
    (verde in avanti, rosso di traverso), percorso reale, percorso pianificato, finestra
    attuale (azzurro), blocchi (quadrati rossi).
    freccia destra / sinistra   scansione successiva / precedente
    freccia su / giu'            10 scansioni avanti / indietro
    1..9, 0                      layer del pannello sinistro (vedi LAYERS)
    o                            pannello destro: fine missione <-> momento della scansione
    g                            grafo PRM sopra la scansione: mostra / nascondi
    il valore sotto il mouse compare in basso a destra, con le coordinate VISION

File letti:
    scans/scan_*.npz    un pacchetto per scansione del ciclo di navigazione
    *_vis.npz           al posto delle vecchie figure: dati + grafo PRM in quel momento
    *_BLOCKED_*.npz     i dati che hanno deciso un blocco
"""
import glob
import os
import sys

import numpy as np

PNG_MODE = "--png" in sys.argv
if PNG_MODE:
    import matplotlib
    matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.collections import LineCollection    # noqa: E402
from matplotlib.colors import ListedColormap          # noqa: E402
import matplotlib.patches as patches                  # noqa: E402
import csv                                            # noqa: E402

SLOPE_THRESHOLD = 0.6   # come spotGrid.SLOPE_THRESHOLD: serve solo a colorare gli archi

REASON_COLORS = {'fine_percorso': 'tab:green', 'fine_dati': 'tab:blue', 'ostacolo': 'tab:red',
                 'rugosita': 'tab:orange', 'pendenza': 'tab:purple'}

# nome, funzione che estrae la matrice, mappa colori, vmin, vmax, etichetta
LAYERS = [
    ("quota rispetto al suolo", lambda d: d['terrain_raw'] - _ground(d), 'terrain', -0.5, 0.5,
     "quota grezza - suolo stimato (m)"),
    ("terrain_valid grezzo", lambda d: d['terrain_valid_raw'], 'gray_r', 0, 1, "1 = valido"),
    ("obstacle_distance", lambda d: d['obstacle_distance'], 'plasma', 0, 1.0, "distanza dall'ostacolo (m)"),
    ("rugosita'", lambda d: d['roughness'], 'coolwarm', 0, 0.3, "deviazione standard (m)"),
    ("gradiente", lambda d: d['gradient'], 'viridis', 0, 1.0, "tan(pendenza)"),
    ("maschera fusa", lambda d: d['obstacle_mask'], 'RdYlGn', -1, 1, "-1 = ostacolo"),
    ("celle usate (is_valid)", lambda d: d['is_valid'].astype(float), 'gray_r', 0, 1,
     "0 = riempita/scartata"),
    ("quota corretta", lambda d: d['terrain_corrected'] - _ground(d), 'terrain', -0.5, 0.5,
     "quota corretta - suolo (m)"),
    ("quota rispetto all'avvio missione", lambda d: d['terrain_corrected'] - _mission_z0(d), 'terrain',
     -2.0, 2.0, "quota corretta - suolo all'avvio missione (m)"),
    ("celle mai scritte", lambda d: d['unwritten'].astype(float) if 'unwritten' in d
     else np.zeros_like(d['terrain_raw']), 'gray_r', 0, 1, "1 = mai scritta: trattata come non vista"),
]


def _ground(d):
    g = float(d['ground_z']) if 'ground_z' in d else np.nan
    if not np.isfinite(g):
        t = d['terrain_raw']
        g = float(np.median(t[np.isfinite(t)]))
    return g


def _mission_z0(d):
    """Suolo all'avvio missione; per i file vecchi (senza il campo) il suolo della scansione."""
    z0 = float(d['mission_z0']) if 'mission_z0' in d else np.nan
    return z0 if np.isfinite(z0) else _ground(d)


def _load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _extent(d):
    ny, nx = d['terrain_raw'].shape
    ox, oy = d['grid_origin']
    cs = float(d['cell_size'])
    return [ox, ox + nx * cs, oy, oy + ny * cs]


def _hover(ax, mat, extent):
    ny, nx = mat.shape
    x0, x1, y0, y1 = extent
    cs = (x1 - x0) / nx

    def fmt(x, y):
        c, r = int((x - x0) / cs), int((y - y0) / cs)
        if 0 <= r < ny and 0 <= c < nx:
            return f"x={x:.2f} y={y:.2f} | riga {r} col {c} | valore {mat[r, c]:.4f}"
        return f"x={x:.2f} y={y:.2f}"
    ax.format_coord = fmt


def draw_scan(ax, d, layer_idx, title_prefix=""):
    ax.clear()
    name, fn, cmap, vmin, vmax, label = LAYERS[layer_idx]
    mat = np.asarray(fn(d), dtype=float)
    ext = _extent(d)
    im = ax.imshow(mat, origin='lower', extent=ext, cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation='nearest')
    _hover(ax, mat, ext)

    # contorno della distanza minima dal centro del percorso
    od = d['obstacle_distance']
    clear = float(d.get('frontier_clearance', 0.30))
    xs = np.linspace(ext[0], ext[1], od.shape[1]); ys = np.linspace(ext[2], ext[3], od.shape[0])
    ax.contour(xs, ys, od, levels=[clear], colors='white', linewidths=0.8, linestyles='--')

    pl = d['polyline']
    ax.plot(pl[:, 0], pl[:, 1], '-', color='magenta', lw=2, label='percorso')
    rx, ry = d['robot_xyz'][:2]
    yaw = float(d['robot_yaw'])
    ax.plot(rx, ry, 'o', color='blue', ms=9)
    ax.arrow(rx, ry, 0.35 * np.cos(yaw), 0.35 * np.sin(yaw), color='blue', width=0.02)
    reason = str(d['frontier_reason'])
    sx, sy = d['frontier_stop_xy']
    ax.plot(sx, sy, 'X', color=REASON_COLORS.get(reason, 'k'), ms=13, mec='white',
            label=f"fronte: {reason}")
    tx, ty = d['decision_target']
    if np.isfinite(tx):
        ax.plot(tx, ty, '*', color='lime', ms=14, mec='k', label='punto comandato')
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect('equal')
    ax.set_xlabel("X [m] (VISION)"); ax.set_ylabel("Y [m] (VISION)")
    ax.set_title(f"{title_prefix}{name}\n"
                 f"fronte {float(d['frontier_dist']):.2f} m ({reason}), avanzamento "
                 f"{float(d['allowed_advance']):.2f} m, decisione: {str(d['decision_kind'])}"
                 + (" [senza ruotare]" if bool(d.get('keep_heading', False)) else ""), fontsize=9)
    ax.legend(loc='upper left', fontsize=7, framealpha=0.8)
    return im, label


def draw_overview(ax, scans):
    ax.clear()
    track = np.array([d['robot_xyz'][:2] for d in scans])
    ax.plot(track[:, 0], track[:, 1], '-o', color='blue', ms=3, lw=1, label='robot')
    for reason, col in REASON_COLORS.items():
        pts = np.array([d['frontier_stop_xy'] for d in scans if str(d['frontier_reason']) == reason])
        if len(pts):
            ax.plot(pts[:, 0], pts[:, 1], 'x', color=col, ms=6, label=f"fronte: {reason} ({len(pts)})")
    blocks = [d for d in scans if str(d['decision_kind']) == 'blocked']
    for d in blocks:
        ax.plot(*d['frontier_stop_xy'], 's', color='red', ms=12, mfc='none', mew=2)
    for d in scans:
        x0, x1, y0, y1 = _extent(d)
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color='gray', lw=0.3, alpha=0.4)
    ax.set_aspect('equal'); ax.grid(alpha=0.3)
    ax.set_title(f"Missione: {len(scans)} scansioni, {len(blocks)} blocchi confermati "
                 f"(quadrati rossi)", fontsize=10)
    ax.set_xlabel("X [m] (VISION)"); ax.set_ylabel("Y [m] (VISION)")
    ax.legend(fontsize=7)


def draw_vis_bundle(ax, d):
    """Le vecchie figure 'Path Visualization': scansione, grafo PRM, percorso, celle."""
    ax.clear()
    pts = d['pts']
    if pts.size:
        mask = d['obstacle_mask']
        ax.scatter(pts[:, 0], pts[:, 1], c=np.where(mask < 0, 1.0, 0.0), cmap='RdYlGn_r', s=1,
                   vmin=0, vmax=1, alpha=0.5)
    xy = {int(i): tuple(p) for i, p in zip(d['prm_node_ids'], d['prm_node_xy'])}
    segs = [(xy[a], xy[b]) for a, b in d['prm_edges'] if a in xy and b in xy]
    if segs:
        ax.add_collection(LineCollection(segs, colors='gray', linewidths=0.3, alpha=0.4))
    trav = [(xy[a], xy[b]) for a, b in d['prm_traversed_edges'] if a in xy and b in xy]
    if trav:
        ax.add_collection(LineCollection(trav, colors='tab:blue', linewidths=2))
    stops = np.array([xy[int(i)] for i in d['prm_stop_nodes'] if int(i) in xy]).reshape(-1, 2)
    if len(stops):
        ax.plot(stops[:, 0], stops[:, 1], 'o', color='tab:orange', ms=5, label='nodi di sosta')
    cp = d['chosen_path']
    if len(cp):
        ax.plot(cp[:, 0], cp[:, 1], '-', color='magenta', lw=2.5, label='percorso scelto')
    ax.plot(*d['robot_xy'], 'o', color='blue', ms=10, label='robot')
    if np.all(np.isfinite(d['chosen_point'])):
        ax.plot(*d['chosen_point'], '*', color='green', ms=16, label='obiettivo')
    for (r, c, x, y), st in zip(d['env_cells'], d['env_cell_status']):
        ax.text(x, y, f"{int(r)},{int(c)}", fontsize=7, ha='center')
    ax.set_aspect('equal'); ax.grid(alpha=0.3); ax.autoscale_view()
    ax.set_title(f"Iterazione {int(d['iteration'])}: {len(segs)} archi PRM, {len(stops)} nodi di sosta",
                 fontsize=10)
    ax.legend(fontsize=7)



# =============================================================================
# VISTA GLOBALE (2026-10-07): l'equivalente della vecchia figura "Global Map View".
# Tutto e' ricostruito "com'era in quel momento": la mappa e' il mosaico delle scansioni
# fatte FINO alla scansione mostrata, il grafo e' l'ultima vista PRM salvata prima.
# =============================================================================
MOSAIC_CMAP = ListedColormap(['#ffffff', '#dfe8d8', '#d62728'])   # ignoto, libero, ostacolo


class MissionContext:
    """Dati di tutta la missione, caricati una volta: scansioni, viste PRM, traiettoria."""

    def __init__(self, folder, scans, scan_files, vis_files):
        self.scans, self.scan_files = scans, scan_files
        self.vis = []
        for f in vis_files:
            try:
                self.vis.append((os.path.getmtime(f), os.path.basename(f), _load(f)))
            except Exception as e:
                print(f"[viewer] vista PRM non leggibile {os.path.basename(f)}: {e}")
        self.vis.sort(key=lambda t: t[0])
        self.traj = self._load_trajectory(folder)
        # griglia del mosaico: risoluzione e estensione comuni a tutte le scansioni
        if scans:
            self.res = float(scans[0]['cell_size'])
            ext = np.array([_extent(d) for d in scans])
            pad = 1.0
            self.x0, self.y0 = ext[:, 0].min() - pad, ext[:, 2].min() - pad
            x1, y1 = ext[:, 1].max() + pad, ext[:, 3].max() + pad
            self.nx = int(np.ceil((x1 - self.x0) / self.res))
            self.ny = int(np.ceil((y1 - self.y0) / self.res))
        self._mosaic, self._mosaic_upto = None, -1

    @staticmethod
    def _load_trajectory(folder):
        """trajectory.csv in MissionLogs/<stessa missione>, se c'e' (missioni dal 2026-10-07)."""
        name = os.path.basename(os.path.normpath(folder))
        for cand in (os.path.join(os.path.dirname(os.path.dirname(os.path.normpath(folder))), "MissionLogs", name,
                                  "trajectory.csv"),
                     os.path.join(folder, "trajectory.csv")):
            if os.path.exists(cand):
                try:
                    with open(cand, newline='', encoding='utf-8') as f:
                        rows = list(csv.DictReader(f))
                    return np.array([(float(r['time']), float(r['x']), float(r['y'])) for r in rows])
                except Exception as e:
                    print(f"[viewer] trajectory.csv non leggibile: {e}")
        return None

    def mosaic(self, upto):
        """Mosaico 0/1/2 (ignoto/libero/ostacolo) delle scansioni 0..upto: l'ultima vince."""
        if self._mosaic is None or upto < self._mosaic_upto:
            self._mosaic = np.zeros((self.ny, self.nx), dtype=np.uint8)
            self._mosaic_upto = -1
        for i in range(self._mosaic_upto + 1, upto + 1):
            d = self.scans[i]
            om = np.asarray(d['obstacle_mask']).reshape(d['terrain_raw'].shape)
            known = np.asarray(d['is_valid'], bool) | (om < 0)
            if 'unwritten' in d:
                known &= ~np.asarray(d['unwritten'], bool) | (om < 0)
            val = np.where(om < 0, 2, 1).astype(np.uint8)
            gox, goy = d['grid_origin']
            r0 = int(np.floor((goy - self.y0) / self.res))
            c0 = int(np.floor((gox - self.x0) / self.res))
            h, w = om.shape
            rs, cs_ = slice(max(r0, 0), min(r0 + h, self.ny)), slice(max(c0, 0), min(c0 + w, self.nx))
            sub_r, sub_c = slice(rs.start - r0, rs.stop - r0), slice(cs_.start - c0, cs_.stop - c0)
            tgt = self._mosaic[rs, cs_]
            k = known[sub_r, sub_c]
            tgt[k] = val[sub_r, sub_c][k]
        self._mosaic_upto = upto
        return self._mosaic

    def vis_before(self, t):
        """Ultima vista PRM salvata prima dell'istante t (None se nessuna)."""
        best = None
        for mt, name, d in self.vis:
            if t is None or mt <= t + 2.0:
                best = (name, d)
        return best if best is not None else ((self.vis[0][1], self.vis[0][2]) if self.vis else None)


def _scalar(d, key, default):
    """Valore scalare da un npz, robusto a campi mancanti, vuoti o NaN (file vecchi o di prova)."""
    try:
        v = np.asarray(d[key], dtype=float).ravel()
        return float(v[0]) if v.size and np.isfinite(v[0]) else default
    except Exception:
        return default


def _arr(d, key, shape_tail=None):
    """Array da un npz, vuoto se manca (file di versioni precedenti)."""
    if key in d:
        return np.asarray(d[key])
    return np.empty((0,) + tuple(shape_tail or ()))


def _slope_color(long_s, lat_s):
    """Come easy_walk._slope_edge_color: verde = in avanti, rosso = di traverso; piu' marcato = piu' ripido."""
    total = long_s + lat_s
    if total < 1e-6:
        return (0.6, 0.6, 0.6, 0.25)
    r, g, b, _ = plt.get_cmap('RdYlGn_r')(lat_s / total)
    return (r, g, b, 0.3 + 0.5 * min(1.0, max(long_s, lat_s) / SLOPE_THRESHOLD))


def _draw_cells(ax, vd):
    """Griglia di missione: verde = visitata, rosso = bloccata, tratteggio = da esplorare."""
    if vd is None or 'env_cells' not in vd or not len(vd['env_cells']):
        return
    size = _scalar(vd, 'env_cell_size', 5.0)
    yaw = _scalar(vd, 'env_origin_yaw', 0.0)
    h = size / 2.0
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    base = np.array([[-h, -h], [h, -h], [h, h], [-h, h]]) @ R.T
    for (r, c, x, y), st in zip(vd['env_cells'], vd['env_cell_status']):
        st = str(st)
        if st == '1':
            kw = dict(edgecolor='darkgreen', facecolor='lightgreen', alpha=0.25, lw=1.2)
        elif st == '-1':
            kw = dict(edgecolor='darkred', facecolor='lightcoral', alpha=0.35, lw=1.2)
        else:
            kw = dict(edgecolor='gray', facecolor='none', alpha=0.6, lw=0.8, ls='--')
        ax.add_patch(patches.Polygon(base + [x, y], zorder=1, **kw))
        ax.text(x, y, f"{int(r)},{int(c)}", ha='center', va='center', fontsize=7, color='0.3', zorder=1)


def overlay_prm_local(ax, ctx, i):
    """Archi e nodi del grafo (ultima vista PRM prima della scansione) sopra la finestra locale."""
    try:
        _overlay_prm_local(ax, ctx, i)
    except Exception as e:
        print(f"[viewer] grafo locale non disegnato: {e}")


def _overlay_prm_local(ax, ctx, i):
    d = ctx.scans[i]
    vis = ctx.vis_before(float(d['time']) if 'time' in d else None)
    if not vis:
        return
    vd = vis[1]
    x0, x1, y0, y1 = _extent(d)
    xy = {int(k): tuple(p) for k, p in zip(_arr(vd, 'prm_node_ids', ()), _arr(vd, 'prm_node_xy', (2,)))}
    inside = lambda p: x0 - 1 <= p[0] <= x1 + 1 and y0 - 1 <= p[1] <= y1 + 1
    slope = {(int(a), int(b)): tuple(v) for (a, b), v in zip(_arr(vd, 'prm_slope_keys', (2,)), _arr(vd, 'prm_slope_values', (2,)))}
    segs, cols = [], []
    for a, b in _arr(vd, 'prm_edges', (2,)):
        a, b = int(a), int(b)
        if a in xy and b in xy and inside(xy[a]) and inside(xy[b]):
            segs.append((xy[a], xy[b]))
            sv = slope.get((min(a, b), max(a, b)))
            c = _slope_color(*sv) if sv else (0.2, 0.2, 0.2, 0.3)
            cols.append((c[0], c[1], c[2], 0.35))
    if segs:
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=0.5, zorder=3))
    pts = np.array([p for p in xy.values() if inside(p)]).reshape(-1, 2)
    if len(pts):
        ax.plot(pts[:, 0], pts[:, 1], '.', color='k', ms=3, alpha=0.6, zorder=3)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)


def draw_global(ax, ctx, i, final=False):
    """Vista globale alla scansione i (o a fine missione con final=True)."""
    ax.clear()
    upto = len(ctx.scans) - 1 if final else i
    d = ctx.scans[upto]
    t_now = float(d['time']) if 'time' in d else None
    mos = ctx.mosaic(upto)
    ax.imshow(mos, origin='lower', cmap=MOSAIC_CMAP, vmin=0, vmax=2, interpolation='nearest',
              extent=[ctx.x0, ctx.x0 + ctx.nx * ctx.res, ctx.y0, ctx.y0 + ctx.ny * ctx.res], zorder=0)

    vis = ctx.vis_before(None if final else t_now)
    vd = vis[1] if vis else None
    try:
        _draw_cells(ax, vd)
    except Exception as e:
        print(f"[viewer] celle non disegnate: {e}")
    if vd is not None:
        xy = {int(k): tuple(p) for k, p in zip(_arr(vd, 'prm_node_ids', ()), _arr(vd, 'prm_node_xy', (2,)))}
        slope = {(int(a), int(b)): tuple(v) for (a, b), v in zip(_arr(vd, 'prm_slope_keys', (2,)), _arr(vd, 'prm_slope_values', (2,)))}
        segs, cols = [], []
        for a, b in _arr(vd, 'prm_edges', (2,)):
            a, b = int(a), int(b)
            if a in xy and b in xy:
                segs.append((xy[a], xy[b]))
                sv = slope.get((min(a, b), max(a, b)))
                cols.append(_slope_color(*sv) if sv else (0.5, 0.5, 0.5, 0.25))
        if segs:
            ax.add_collection(LineCollection(segs, colors=cols, linewidths=0.5, zorder=2))
        stops = np.array([xy[int(k)] for k in _arr(vd, 'prm_stop_nodes', ()) if int(k) in xy]).reshape(-1, 2)
        if len(stops):
            ax.plot(stops[:, 0], stops[:, 1], '.', color='tab:orange', ms=5, zorder=4, label='nodi di sosta')
        cp = _arr(vd, 'chosen_point').astype(float).ravel()
        if cp.size == 2 and np.all(np.isfinite(cp)):
            ax.plot(*cp, '*', color='green', ms=15, mec='k', zorder=6, label='obiettivo cella')

    # traiettoria reale (trajectory.csv) o, se manca, le posizioni alle scansioni
    if ctx.traj is not None and len(ctx.traj):
        tr = ctx.traj if (final or t_now is None) else ctx.traj[ctx.traj[:, 0] <= t_now + 0.5]
        if len(tr):
            ax.plot(tr[:, 1], tr[:, 2], '-', color='tab:blue', lw=1.5, alpha=0.8, zorder=5, label='percorso reale')
    else:
        track = np.array([s['robot_xyz'][:2] for s in ctx.scans[:upto + 1]])
        ax.plot(track[:, 0], track[:, 1], '-', color='tab:blue', lw=1.5, alpha=0.8, zorder=5,
                label='robot (alle scansioni)')

    for s in ctx.scans[:upto + 1]:
        if str(s['decision_kind']) == 'blocked':
            ax.plot(*s['frontier_stop_xy'], 's', color='red', ms=11, mfc='none', mew=2, zorder=7)
    if not final:
        pl = d['polyline']
        ax.plot(pl[:, 0], pl[:, 1], '-', color='magenta', lw=2.5, zorder=6, label='percorso pianificato')
        x0, x1, y0, y1 = _extent(d)
        ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor='cyan', lw=2, zorder=6))
        reason = str(d['frontier_reason'])
        ax.plot(*d['frontier_stop_xy'], 'X', color=REASON_COLORS.get(reason, 'k'), ms=11, mec='white', zorder=8)
    rx, ry = d['robot_xyz'][:2]
    ax.plot(rx, ry, 'o', color='blue', ms=9, mec='white', zorder=9)

    ax.set_aspect('equal')
    ax.grid(alpha=0.2)
    n_blk = sum(str(s['decision_kind']) == 'blocked' for s in ctx.scans[:upto + 1])
    title = ("Fine missione" if final else f"Mappa globale alla scansione {upto + 1}/{len(ctx.scans)}")
    ax.set_title(f"{title}\nrosso = ostacolo visto, verde chiaro = libero, bianco = mai visto; "
                 f"quadrati rossi = blocchi ({n_blk})" + (f"\ngrafo: {vis[0]}" if vis else ""), fontsize=8)
    ax.set_xlabel("X [m] (VISION)")
    ax.legend(loc='upper left', fontsize=7, framealpha=0.8)


def _latest_mission_folder():
    """
    Senza argomenti (es. tasto Run dell'editor): l'ultima missione in MissionMap/, cercata
    accanto a questo file e nella cartella corrente. Se non c'e', finestra per sceglierla.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    cands = []
    for base in (here, os.getcwd(), os.path.dirname(here)):
        cands += [d for d in glob.glob(os.path.join(base, "MissionMap", "Mission_*")) if os.path.isdir(d)]
    if cands:
        return max(set(cands), key=os.path.getmtime)
    try:
        import tkinter
        from tkinter import filedialog
        root = tkinter.Tk()
        root.withdraw()
        return filedialog.askdirectory(title="Cartella della missione (dentro MissionMap)") or None
    except Exception:
        return None


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        found = _latest_mission_folder()
        if not found:
            print(__doc__)
            print("Nessuna cartella MissionMap/Mission_* trovata accanto a questo file.")
            return 1
        print(f"Nessuna cartella indicata: apro l'ultima missione, {found}")
        args = [found]
    target = args[0]
    out_dir = sys.argv[sys.argv.index("--png") + 1] if PNG_MODE else None

    if os.path.isdir(target):
        scan_files = sorted(glob.glob(os.path.join(target, "scans", "scan_*.npz")))
        vis_files = sorted(glob.glob(os.path.join(target, "*_vis.npz")))
    else:
        scan_files = [target] if os.path.basename(target).startswith("scan_") else []
        vis_files = [target] if target.endswith("_vis.npz") else []
    scans = [_load(f) for f in scan_files]
    print(f"{len(scans)} scansioni, {len(vis_files)} viste PRM")
    folder = target if os.path.isdir(target) else os.path.dirname(os.path.dirname(os.path.abspath(target)))
    ctx = MissionContext(folder, scans, scan_files, vis_files) if scans else None

    if PNG_MODE:
        os.makedirs(out_dir, exist_ok=True)
        for k, (f, d) in enumerate(zip(scan_files, scans)):
            fig, (axl, axr) = plt.subplots(1, 2, figsize=(20, 9.5))
            im, label = draw_scan(axl, d, 5)          # maschera degli ostacoli
            overlay_prm_local(axl, ctx, k)
            fig.colorbar(im, ax=axl, label=label, shrink=0.8)
            draw_global(axr, ctx, k)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, os.path.basename(f).replace(".npz", ".png")), dpi=100)
            plt.close(fig)
        for f in vis_files:
            fig, ax = plt.subplots(figsize=(11, 11))
            draw_vis_bundle(ax, _load(f))
            fig.savefig(os.path.join(out_dir, os.path.basename(f).replace(".npz", ".png")), dpi=110)
            plt.close(fig)
        if scans:
            fig, ax = plt.subplots(figsize=(12, 12))
            draw_global(ax, ctx, len(scans) - 1, final=True)
            fig.savefig(os.path.join(out_dir, "overview.png"), dpi=110)
            plt.close(fig)
        print(f"Figure salvate in {out_dir}")
        return 0

    if "--overview" in sys.argv or not scans:
        fig, ax = plt.subplots(figsize=(11, 11))
        if scans:
            draw_global(ax, ctx, len(scans) - 1, final=True)
        elif vis_files:
            draw_vis_bundle(ax, _load(vis_files[0]))
        plt.show()
        return 0

    state = {'i': 0, 'layer': 0, 'overview': False, 'cbar': None, 'prm': True}
    fig, (ax, axg) = plt.subplots(1, 2, figsize=(18, 8.5))

    def redraw():
        if state['cbar'] is not None:
            state['cbar'].remove(); state['cbar'] = None
        d = scans[state['i']]
        im, label = draw_scan(ax, d, state['layer'],
                              title_prefix=f"[{state['i'] + 1}/{len(scans)}] "
                                           f"{os.path.basename(scan_files[state['i']])}\n")
        state['cbar'] = fig.colorbar(im, ax=ax, label=label, shrink=0.8)
        if state['prm']:
            overlay_prm_local(ax, ctx, state['i'])
        draw_global(axg, ctx, state['i'], final=state['overview'])
        fig.canvas.draw_idle()

    def on_key(ev):
        if ev.key == 'right':
            state['i'] = min(len(scans) - 1, state['i'] + 1); state['overview'] = False
        elif ev.key == 'left':
            state['i'] = max(0, state['i'] - 1); state['overview'] = False
        elif ev.key == 'up':
            state['i'] = min(len(scans) - 1, state['i'] + 10); state['overview'] = False
        elif ev.key == 'down':
            state['i'] = max(0, state['i'] - 10); state['overview'] = False
        elif ev.key in [str((k + 1) % 10) for k in range(min(len(LAYERS), 10))]:
            state['layer'] = (int(ev.key) - 1) % 10   # '0' = decimo layer
        elif ev.key == 'o':
            state['overview'] = not state['overview']
        elif ev.key == 'g':
            state['prm'] = not state['prm']
        else:
            return
        redraw()

    fig.canvas.mpl_connect('key_press_event', on_key)
    print("Frecce sinistra/destra: scansione | su/giu': 10 scansioni | 1-9, 0: layer a sinistra | "
          "o: a destra fine missione / momento della scansione | g: grafo a sinistra si'/no | "
          "valore sotto il mouse in basso a destra")
    for k, layer in enumerate(LAYERS):
        print(f"  {k + 1}: {layer[0]}")
    redraw()
    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())