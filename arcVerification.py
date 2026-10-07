"""
Verifica del percorso in tempo reale: FRONTE SICURO (2026-10-06).

Fino al 2026-10-05 questo modulo verificava il percorso ARCO PER ARCO e ricordava
per sempre i verdetti ("verificato" / "bloccato"). Nella prima missione in esterno
(2026-10-05 16:05) questo ha fatto ritirare il robot cinque volte da terreno libero:
gli archi venivano giudicati mentre il robot camminava, da lontano, troncati sul
bordo della finestra -- e il verdetto non veniva piu' riesaminato. Vedi CONTEXT.md,
punto 19. Il difetto di fondo era di impostazione: l'arco e' un oggetto del grafo,
senza significato fisico, mentre la visibilita' e' continua.

Ora il dato e' uno solo: FIN DOVE, lungo il percorso pianificato e partendo dalla
posizione attuale del robot, il terreno e' confermato libero ADESSO. Si ricalcola da
zero a ogni scansione, quindi non esiste un verdetto che possa sopravvivere alle
condizioni in cui e' stato preso.

Il fronte si ferma al primo di:
  'ostacolo'      obstacle_distance sotto la distanza minima dalla linea centrale
  'rugosita'      cella vetata per rugosita' (stesso criterio di fuse_obstacle_mask)
  'pendenza'      tratto con pendenza longitudinale o laterale oltre SLOPE_THRESHOLD
  'fine_dati'     fuori dalla finestra, o cella mai vista dal sensore
  'fine_percorso' tutto il percorso e' libero

La distinzione fra "ostacolo" e "non ancora visto" e' quella che mancava: un fronte
che si ferma sul bordo dei dati dice "avvicinati e guarda", non "e' bloccato".

ATTENZIONE AL NOME "FOV" nelle funzioni di supporto in fondo al file: non e' il campo
visivo delle telecamere (360 gradi, ben oltre), e' l'estensione della griglia locale
(~3.8 m di lato attorno al robot).
"""
import threading
import time
import logging

import numpy as np

from bosdyn.client.async_tasks import AsyncPeriodicQuery
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.frame_helpers import get_a_tform_b, VISION_FRAME_NAME, BODY_FRAME_NAME

import spotUtils
import spotGrid


LOGGER = logging.getLogger(__name__)

FRONTIER_REASONS_STOP = ('ostacolo', 'rugosita', 'pendenza')   # c'e' qualcosa
FRONTIER_REASONS_OPEN = ('fine_dati', 'fine_percorso')          # non si vede oltre / finito


# =============================================================================
# Fotografia della griglia locale: tutto cio' che serve al fronte, gia' in 2D.
# =============================================================================
# =========================================================================
# FILTRO TEMPORALE SU obstacle_distance (2026-10-07)
#
# Misurato sulla missione del 2026-10-07 14:0x, 19 scansioni dell'iterazione 0: con il
# robot FERMO e il percorso IDENTICO (stessi nodi [187, 140, 3152], stessa origine di
# griglia, stesso yaw), fra la scansione 003 e la 004 la cella in (-1.38, 1.46) ha
# letto 0.296 e poi 0.336 su una soglia secca di 0.300. Verdetto del fronte: "ostacolo,
# avanzamento 0.04 m" e poi "libero, 1.35 m". Niente si era mosso.
#
# Rumore misurato su obstacle_distance fra due scansioni consecutive, lungo il percorso:
# 1.5 cm in media, 7.3 cm di punta -- cioe' dello stesso ordine del margine d'aria di
# 5 cm con cui lo confrontiamo. Un percorso che sfiora il bordo gonfiato di un ostacolo
# attraversa quindi la soglia avanti e indietro a ogni scansione, e il robot alterna
# "vai" e "fermati" sulla stessa scena. E' la causa dei cicli attesa/blocco/arretramento
# osservati, non una decisione sbagliata del fronte.
#
# Due rimedi, scelti il 2026-10-07:
#   - MEDIANA sulle ultime OBSTACLE_MEDIAN_FRAMES letture della stessa cella di mondo.
#     Una punta isolata di rumore non decide piu' da sola. Costa al massimo una
#     scansione di ritardo nel vedere un ostacolo nuovo, che il fronte assorbe perche'
#     guarda avanti 1.3 m, non 3 cm.
#   - ISTERESI: una cella diventa bloccante sotto FRONTIER_CLEARANCE_M e smette di
#     esserlo solo sopra FRONTIER_CLEARANCE_M + OBSTACLE_HYSTERESIS_M. Dentro la banda
#     il verdetto precedente resta, quindi non puo' oscillare.
#
# L'allineamento fra scansioni e' esatto e gratuito: verificato sulle 19 scansioni, le
# origini di griglia sono sempre multipli interi della cella (0.03 m) e gli scorrimenti
# fra scansioni consecutive sono interi esatti (es. -40, -1 celle). Il riallineamento e'
# quindi uno scorrimento di indici, senza interpolazione e senza perdita. Se per qualche
# ragione lo scorrimento non fosse intero, o cambiasse la dimensione della cella, la
# cronologia viene azzerata e si torna al comportamento senza filtro: mai dati inventati.
# =========================================================================
OBSTACLE_MEDIAN_FRAMES = 3
OBSTACLE_HYSTERESIS_M = 0.04


class _ObstacleHistory:
    """
    Cronologia di obstacle_distance allineata al mondo, condivisa fra il ciclo principale
    e il thread del fronte (le scansioni vengono dallo stesso robot a frazioni di secondo
    di distanza: piu' campioni rendono la mediana migliore, non peggiore).

    update() restituisce (mediana, latch): la mediana da usare al posto della lettura
    grezza, e la maschera booleana delle celle che l'isteresi considera bloccanti.
    """

    def __init__(self, frames=OBSTACLE_MEDIAN_FRAMES, hysteresis_m=OBSTACLE_HYSTERESIS_M):
        self._lock = threading.Lock()
        self._frames = max(1, int(frames))
        self._hyst = float(hysteresis_m)
        self._hist = []          # [(od 2D, ix, iy)] -- ix/iy = origine in CELLE di mondo
        self._latch = None       # 2D bool, allineata all'ultimo elemento di _hist
        self._latch_ij = None
        self._cell_size = None

    def reset(self):
        with self._lock:
            self._hist = []
            self._latch = None
            self._latch_ij = None
            self._cell_size = None

    @staticmethod
    def _align(src, dix, diy, shape):
        """
        src riletta negli indici di una griglia spostata di (dix, diy) celle: il valore
        che la nuova griglia vede in (r, c) sta in src[r + diy, c + dix]. Fuori da src:
        NaN per i float, False per i booleani.
        """
        rows, cols = shape
        fill = False if src.dtype == bool else np.nan
        out = np.full(shape, fill, dtype=src.dtype if src.dtype == bool else np.float64)
        r0, c0 = max(0, diy), max(0, dix)                       # primo indice valido in src
        rr0, cc0 = max(0, -diy), max(0, -dix)                   # dove finisce in out
        nr = min(src.shape[0] - r0, rows - rr0)
        nc = min(src.shape[1] - c0, cols - cc0)
        if nr > 0 and nc > 0:
            out[rr0:rr0 + nr, cc0:cc0 + nc] = src[r0:r0 + nr, c0:c0 + nc]
        return out

    # Tolleranza sul resto dello scorrimento, in frazione di cella. Lo scorrimento si misura
    # SEMPRE fra due griglie consecutive, mai da un'origine assoluta: cell_size arriva dal
    # protobuf come float32 (0.029999999329447746, non 0.03), e dividendo per quel valore
    # un'origine a -88 celle l'errore relativo diventa 2.5e-06 -- sopra qualunque tolleranza
    # ragionevole, e la cronologia si azzererebbe a ogni scansione (succedeva: scoperto il
    # 2026-10-07 perche' il filtro non cambiava nessun verdetto). La differenza fra due
    # origini vicine, invece, resta esatta a meno di 1e-7 celle.
    _SHIFT_TOL_CELLS = 0.05

    def update(self, od, origin_x, origin_y, cell_size):
        od = np.asarray(od, dtype=np.float64)
        cs = float(cell_size)
        ox, oy = float(origin_x), float(origin_y)

        def shift_from(old_ox, old_oy):
            """(dix, diy) in celle, oppure None se lo scorrimento non e' intero."""
            fx, fy = (ox - old_ox) / cs, (oy - old_oy) / cs
            dix, diy = int(round(fx)), int(round(fy))
            if abs(fx - dix) > self._SHIFT_TOL_CELLS or abs(fy - diy) > self._SHIFT_TOL_CELLS:
                return None
            return dix, diy

        with self._lock:
            if self._cell_size is not None and abs(self._cell_size - cs) > 1e-9:
                self._hist = []
                self._latch = None
                self._latch_ij = None
            self._cell_size = cs

            stack = [od]
            for (old, old_ox, old_oy) in self._hist:
                if old.shape != od.shape:
                    continue
                sh = shift_from(old_ox, old_oy)
                if sh is None:
                    continue
                stack.append(self._align(old, sh[0], sh[1], od.shape))
            if len(stack) > 1:
                with np.errstate(invalid='ignore'):
                    med = np.nanmedian(np.stack(stack, axis=0), axis=0)
                med = np.where(np.isfinite(med), med, od)
            else:
                med = od.copy()

            thr = spotGrid.FRONTIER_CLEARANCE_M
            latch = med < thr
            if self._latch is not None and self._latch_ij is not None:
                sh = shift_from(self._latch_ij[0], self._latch_ij[1])
                if sh is not None:
                    prev = self._align(self._latch, sh[0], sh[1], od.shape)
                    latch |= prev & (med < thr + self._hyst)

            self._hist.append((od, ox, oy))
            if len(self._hist) > self._frames:
                self._hist.pop(0)
            self._latch = latch
            self._latch_ij = (ox, oy)

        return med, latch


OBSTACLE_HISTORY = _ObstacleHistory()


class GridSnapshot:
    """
    Una scansione pronta per il fronte. Tutti gli array sono 2D (num_y, num_x), con lo
    stesso ordinamento usato ovunque nel progetto: riga = y, colonna = x, origine nel
    vertice in basso a sinistra (vedi spotGrid._gather_heights).

    blocked_latch (2026-10-07): maschera dell'isteresi, vedi _ObstacleHistory. None =
    nessun filtro, comportamento identico a prima.
    """
    __slots__ = ('obstacle_dist', 'rough_veto', 'seen', 'terrain', 'origin_x', 'origin_y',
                 'cell_size', 'robot_x', 'robot_y', 'time', 'valid', 'blocked_latch')

    def __init__(self, obstacle_dist, rough_veto, seen, terrain, origin_x, origin_y, cell_size,
                 robot_x, robot_y, timestamp=None, valid=None, blocked_latch=None):
        self.blocked_latch = blocked_latch
        self.obstacle_dist = obstacle_dist
        self.rough_veto = rough_veto
        self.seen = seen
        self.terrain = terrain
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)
        self.cell_size = float(cell_size)
        self.robot_x = float(robot_x)
        self.robot_y = float(robot_y)
        self.time = time.time() if timestamp is None else timestamp
        self.valid = valid      # is_valid 2D (None = tutte attendibili): per il profilo di pendenza


def make_grid_snapshot(cells_obstacle_dist, rough_values, is_valid, terrain_valid_raw,
                       terrain_corrected, num_x, num_y, origin_x, origin_y, cell_size,
                       robot_x, robot_y, unwritten=None):
    """
    Costruisce la fotografia dai vettori piatti che il codice calcola gia' a ogni scansione.

    - rough_veto: stesso criterio di fuse_obstacle_mask (rugosita' > soglia, su cella valida).
    - seen: il sensore ha dato una quota per quella cella (terrain_valid grezzo). NON e'
      is_valid: is_valid esclude anche le celle scartate dal filtro altezza (muri, celle mai
      scritte), che invece sono state viste -- l'eventuale ostacolo lo segnala
      obstacle_distance.
      2026-10-07: il confronto e' con spotGrid.TERRAIN_VALID_MIN, non con 0.0. Il layer
      contiene 0 e 1, ma unpack_grid gli applica scale e offset e lo 0 diventa +/-5.5e-17,
      con il SEGNO che cambia da una scansione all'altra: il vecchio `> 0.0` dichiarava
      "vista" ogni cella in 10 scansioni su 16. Vedi la costante in spotGrid.
    - unwritten (2026-10-06): celle mai scritte dai sensori (correct_terrain con
      return_unwritten=True). Spot ne dichiara non valide il 98-100% (misurato il
      2026-10-07), ma non tutte: qui contano comunque come NON viste, quindi il fronte si
      ferma li' con 'fine_dati'.
    """
    shape = (num_y, num_x)
    rough = np.asarray(rough_values, dtype=np.float64).reshape(shape)
    valid = np.asarray(is_valid, dtype=bool).reshape(shape)
    if terrain_valid_raw is not None and np.size(terrain_valid_raw) == num_x * num_y:
        seen = np.asarray(terrain_valid_raw).reshape(shape) >= spotGrid.TERRAIN_VALID_MIN
    else:
        seen = np.ones(shape, dtype=bool)
    if unwritten is not None and np.size(unwritten) == num_x * num_y:
        seen = seen & ~np.asarray(unwritten, dtype=bool).reshape(shape)
    # Filtro temporale + isteresi su obstacle_distance (vedi _ObstacleHistory). La lettura
    # grezza non viene piu' usata direttamente da nessuno: ne' dal fronte, ne' dal test di
    # rotazione, che leggono entrambi questa fotografia.
    od_raw = np.asarray(cells_obstacle_dist, dtype=np.float64).reshape(shape)
    od_filtered, latch = OBSTACLE_HISTORY.update(od_raw, origin_x, origin_y, cell_size)

    return GridSnapshot(
        obstacle_dist=od_filtered,
        rough_veto=(rough > spotGrid.ROUGH_THRESHOLD) & valid,
        seen=seen,
        terrain=np.asarray(terrain_corrected, dtype=np.float64).reshape(shape),
        origin_x=origin_x, origin_y=origin_y, cell_size=cell_size,
        robot_x=robot_x, robot_y=robot_y, valid=valid, blocked_latch=latch)


def _cells(snap, xs, ys):
    """Indici di cella per indicizzazione diretta (niente ricerca del vicino) + maschera 'dentro'."""
    ny, nx = snap.obstacle_dist.shape
    cols = np.floor((xs - snap.origin_x) / snap.cell_size).astype(np.int64)
    rows = np.floor((ys - snap.origin_y) / snap.cell_size).astype(np.int64)
    inside = (rows >= 0) & (rows < ny) & (cols >= 0) & (cols < nx)
    return np.clip(rows, 0, ny - 1), np.clip(cols, 0, nx - 1), inside


def _polyline_samples(polyline, step):
    """Campioni equispaziati lungo la spezzata: (xs, ys, s, indice del segmento)."""
    xs, ys, ss, seg = [], [], [], []
    s0 = 0.0
    for k in range(len(polyline) - 1):
        (x1, y1), (x2, y2) = polyline[k], polyline[k + 1]
        L = float(np.hypot(x2 - x1, y2 - y1))
        if L < 1e-9:
            continue
        n = max(1, int(np.ceil(L / step)))
        t = np.arange(n) / n                      # l'ultimo punto e' il primo del segmento dopo
        xs.append(x1 + t * (x2 - x1)); ys.append(y1 + t * (y2 - y1))
        ss.append(s0 + t * L); seg.append(np.full(n, k))
        s0 += L
    if not xs:
        x, y = polyline[0]
        return np.array([x]), np.array([y]), np.array([0.0]), np.array([0]), 0.0
    x_last, y_last = polyline[-1]
    xs.append([x_last]); ys.append([y_last]); ss.append([s0]); seg.append([len(polyline) - 2])
    return (np.concatenate(xs), np.concatenate(ys), np.concatenate(ss),
            np.concatenate(seg).astype(int), s0)


def compute_safe_frontier(polyline, snap, extend_end_m=None, check_slope=True, body=None):
    """
    Fin dove il percorso e' confermato libero adesso.

    Args:
        polyline: [(x, y), ...] -- il PRIMO punto e' la posizione del robot, poi i waypoint.
        snap: GridSnapshot della scansione.
        extend_end_m: prolunga l'ultimo segmento di questa lunghezza oltre l'obiettivo, per
            verificare anche il terreno sotto il muso quando il robot ci arriva. Default
            FRONTIER_BODY_HALF_LENGTH_M.
        check_slope: controlla anche la pendenza dei tratti liberi.
        body: (meta' ingombro lungo la marcia, meta' ingombro di traverso), vedi
            spotGrid.body_extents. Default: robot che guarda dove va (0.55, 0.25). Quando il
            robot non puo' ruotare (passo 4) e cammina di lato, l'ingombro cambia: di
            traverso diventa 0.55 m.

    Returns: dict con
        dist        distanza lungo il percorso del primo punto NON libero (m)
        reason      vedi docstring del modulo
        stop_xy     dove si ferma il fronte
        value       il valore che ha fermato il fronte (obstacle_distance, pendenza, ...)
        path_len    lunghezza del percorso vero, senza prolungamento
        od_robot    obstacle_distance sotto il robot
        clearance   distanza minima applicata nel primo tratto
        robot_xy, time
    """
    half_along, half_across = body if body is not None else (spotGrid.FRONTIER_BODY_HALF_LENGTH_M,
                                                             spotGrid.ROBOT_HALF_WIDTH_M)
    clearance_nominal = half_across + spotGrid.ROBOT_CLEARANCE_AIR_M
    if extend_end_m is None:
        extend_end_m = half_along
    pl = [(float(x), float(y)) for x, y in polyline]

    # Lunghezza vera e prolungamento oltre l'obiettivo
    path_len = float(sum(np.hypot(pl[k + 1][0] - pl[k][0], pl[k + 1][1] - pl[k][1])
                         for k in range(len(pl) - 1)))
    if len(pl) >= 2 and extend_end_m > 0:
        (xa, ya), (xb, yb) = pl[-2], pl[-1]
        L = np.hypot(xb - xa, yb - ya)
        if L > 1e-6:
            pl.append((xb + extend_end_m * (xb - xa) / L, yb + extend_end_m * (yb - ya) / L))

    xs, ys, ss, seg, _ = _polyline_samples(pl, spotGrid.FRONTIER_SAMPLE_STEP_M)
    rows, cols, inside = _cells(snap, xs, ys)

    # obstacle_distance sotto il robot: decide se siamo gia' in uno spazio stretto
    r0, c0, in0 = _cells(snap, np.array([snap.robot_x]), np.array([snap.robot_y]))
    od_robot = float(snap.obstacle_dist[r0[0], c0[0]]) if in0[0] else np.inf
    clearance_escape = max(spotGrid.FRONTIER_MIN_CLEARANCE_M,
                           min(clearance_nominal, od_robot - 0.02))
    thr = np.where(ss < spotGrid.FRONTIER_ESCAPE_DIST_M, clearance_escape, clearance_nominal)

    od = snap.obstacle_dist[rows, cols]
    near_robot = np.hypot(xs - snap.robot_x, ys - snap.robot_y) <= spotGrid.FRONTIER_ROBOT_RADIUS_M

    no_data = ~inside | (~snap.seen[rows, cols] & ~near_robot)
    obstacle = inside & (od < thr)
    # Isteresi (2026-10-07): una cella gia' giudicata bloccante resta bloccante finche' non
    # risale sopra FRONTIER_CLEARANCE_M + OBSTACLE_HYSTERESIS_M (vedi _ObstacleHistory).
    # Si applica SOLO dove vale la soglia nominale: dove la regola di uscita l'ha abbassata
    # il robot sta cercando di uscire da uno spazio stretto, e lagarci dentro l'isteresi lo
    # inchioderebbe li'.
    if snap.blocked_latch is not None:
        nominal = thr >= clearance_nominal - 1e-9
        obstacle |= inside & nominal & snap.blocked_latch[rows, cols]
    # Regola di uscita (2026-10-06 sera, corretta dopo la revisione): nel primo metro la
    # soglia scende a od_robot - 2 cm per permettere di uscire da uno spazio stretto, ma
    # quella soglia valeva per QUALUNQUE ostacolo lungo il tratto. Un palo a 15 cm dal
    # fianco, 70 cm avanti, veniva accettato solo perche' il robot partiva vicino a un
    # altro oggetto (verificato: il corpo si sovrapponeva al palo di 10 cm). La promessa e'
    # "non avvicinarsi piu' di quanto gia' si sia": nel primo metro un campione sotto la
    # soglia nominale e' ammesso solo se non scende sotto il meglio raggiunto finora lungo
    # il percorso (meno la tolleranza qui sotto). Allontanarsi resta permesso, riavvicinarsi no.
    if clearance_escape < clearance_nominal:
        od_capped = np.minimum(np.where(inside, od, clearance_nominal), clearance_nominal)
        best_so_far = np.maximum.accumulate(od_capped)
        in_escape = inside & (ss < spotGrid.FRONTIER_ESCAPE_DIST_M) & (od < clearance_nominal)
        # Tolleranza: almeno 1.5 celle. obstacle_distance va a gradini di una cella (3 cm):
        # con 2 cm un percorso quasi parallelo a un muro, a distanza costante, scattava al
        # gradino successivo (verificato dalla seconda revisione: bloccato 257 volte su 360
        # a 5 gradi dal muro, prima 13). Con 4.5 cm il comportamento lungo un muro torna
        # quello di prima, e il palo sul fianco resta bloccato.
        tol = max(0.02, 1.5 * snap.cell_size)
        obstacle = obstacle | (in_escape & (od < best_so_far - tol))
    rough = inside & snap.rough_veto[rows, cols] & ~near_robot

    bad = no_data | obstacle | rough
    result = dict(path_len=path_len, od_robot=od_robot, clearance=clearance_escape,
                  clearance_nominal=clearance_nominal, half_along=half_along,
                  robot_xy=(snap.robot_x, snap.robot_y), time=snap.time)

    if bad.any():
        k = int(np.argmax(bad))
        if no_data[k]:
            reason, value = 'fine_dati', None
        elif obstacle[k]:
            reason, value = 'ostacolo', float(od[k])
        else:
            reason, value = 'rugosita', None
        dist, stop_xy = float(ss[k]), (float(xs[k]), float(ys[k]))
    else:
        reason, value = 'fine_percorso', None
        dist, stop_xy = float(ss[-1]), (float(xs[-1]), float(ys[-1]))

    # Pendenza sui tratti GIA' confermati liberi, segmento per segmento: se un tratto e'
    # troppo ripido il fronte arretra al suo inizio. Stessa funzione e stessa soglia del
    # PRM, quindi il fronte non puo' rifiutare per pendenza un arco che il PRM ammette
    # sugli stessi dati.
    if check_slope and spotGrid.FRONTIER_SLOPE_MIN_LENGTH_M > 0:
        s_start = 0.0
        for k in range(len(pl) - 1):
            (x1, y1), (x2, y2) = pl[k], pl[k + 1]
            L = float(np.hypot(x2 - x1, y2 - y1))
            if s_start >= dist:
                break
            free_len = min(L, dist - s_start)
            if free_len >= spotGrid.FRONTIER_SLOPE_MIN_LENGTH_M:
                f = free_len / L
                long_s, lat_s, has = spotGrid.compute_arc_slope_profile(
                    snap.terrain, snap.origin_x, snap.origin_y, snap.cell_size,
                    x1, y1, x1 + f * (x2 - x1), y1 + f * (y2 - y1), valid_2d=snap.valid)
                if has and max(long_s, lat_s) > spotGrid.SLOPE_THRESHOLD:
                    reason, value = 'pendenza', float(max(long_s, lat_s))
                    dist, stop_xy = s_start, (x1, y1)
                    break
            s_start += L

    result.update(dist=dist, reason=reason, stop_xy=stop_xy, value=value)
    return result


def allowed_advance(frontier):
    """
    Quanto puo' avanzare il CENTRO del robot lungo il percorso.

    Se tutto il percorso (prolungamento compreso) e' libero, fino all'obiettivo. Altrimenti
    il centro si ferma meta' lunghezza del corpo + margine prima del primo punto non libero,
    cosi' anche il muso resta su terreno confermato. Vale anche per 'fine_dati': il muso
    non deve finire su terreno mai visto.
    """
    if frontier['reason'] == 'fine_percorso':
        return frontier['path_len']
    half_along = frontier.get('half_along', spotGrid.FRONTIER_BODY_HALF_LENGTH_M)
    return max(0.0, frontier['dist'] - half_along - spotGrid.FRONTIER_MARGIN_M)


def point_along(polyline, s):
    """Il punto a distanza s lungo la spezzata (troncato agli estremi) e l'indice del segmento."""
    s_acc = 0.0
    for k in range(len(polyline) - 1):
        (x1, y1), (x2, y2) = polyline[k], polyline[k + 1]
        L = float(np.hypot(x2 - x1, y2 - y1))
        if s <= s_acc + L or k == len(polyline) - 2:
            t = 0.0 if L < 1e-9 else min(1.0, max(0.0, (s - s_acc) / L))
            return (x1 + t * (x2 - x1), y1 + t * (y2 - y1)), k
        s_acc += L
    return polyline[-1], max(0, len(polyline) - 2)


def describe_frontier(frontier):
    """Una riga leggibile per il log."""
    r = frontier['reason']
    adv = allowed_advance(frontier)
    sx, sy = frontier['stop_xy']
    base = (f"{frontier['dist']:.2f} m confermati su {frontier['path_len']:.2f} m di percorso, "
            f"avanzamento consentito {adv:.2f} m")
    if r == 'fine_percorso':
        return f"[FRONTE] {base} -- percorso libero fino all'obiettivo"
    if r == 'fine_dati':
        return f"[FRONTE] {base} -- si ferma sul bordo dei dati in ({sx:.2f}, {sy:.2f}): avanzo e riguardo"
    if r == 'ostacolo':
        return (f"[FRONTE] {base} -- ostacolo a {frontier['value']:.2f} m dalla linea centrale in "
                f"({sx:.2f}, {sy:.2f}) (minimo {frontier.get('clearance_nominal', spotGrid.FRONTIER_CLEARANCE_M):.2f} m)")
    if r == 'rugosita':
        return f"[FRONTE] {base} -- terreno troppo irregolare in ({sx:.2f}, {sy:.2f})"
    return (f"[FRONTE] {base} -- tratto troppo ripido da ({sx:.2f}, {sy:.2f}) "
            f"(pendenza {frontier['value']:.3f} > {spotGrid.SLOPE_THRESHOLD:.3f})")


# =============================================================================
# Verificatore in background: tiene aggiornato il fronte mentre il robot cammina.
# =============================================================================
def _update_thread(async_task):
    while True:
        async_task.update()
        time.sleep(0.1)


class AsyncArcVerificationTracker(AsyncPeriodicQuery):
    """Non usata (chiama un servizio che non esiste). Lasciata com'era."""

    def __init__(self, robot_state_client):
        super(AsyncArcVerificationTracker, self).__init__('arc_verification', robot_state_client, LOGGER,
                                                          period_sec=0.2)

    def _start_query(self):
        return self._client.get_arc_verification(['arc_verification'])


class ArcVerificationTracker:
    """
    Thread in background che ricalcola il fronte sicuro sul percorso corrente a ogni
    scansione (~5 Hz), con dati propri e freschi.

    Serve DURANTE il movimento: il ciclo principale lo interroga per fermare il robot se il
    fronte si accorcia sotto il punto verso cui sta andando. Prima di partire il ciclo
    principale calcola il fronte da se', sulla propria scansione: non c'e' piu' nessuna
    attesa del thread.

    Il risultato porta la versione del percorso su cui e' stato calcolato: dopo una
    ripianificazione un fronte vecchio non puo' essere scambiato per uno nuovo.
    """

    def __init__(self, robot):
        self.robot = robot
        self._robot_state_client = self.robot.ensure_client(RobotStateClient.default_service_name)
        self._local_grid = spotGrid.LocalGrid(self.robot)
        self._lock = threading.Lock()
        self._waypoints = []          # [(x, y), ...] SENZA la posizione del robot
        self._extend_end_m = None
        self._body = None
        self._path_version = 0
        self._latest = None           # ultimo fronte calcolato
        self._running = False
        self._thread = None

    # --- interfaccia per il ciclo principale -------------------------------------------
    def update_path(self, waypoints_xy, extend_end_m=None, body=None):
        """
        Imposta il percorso da seguire (solo i waypoint). Restituisce la nuova versione.
        body: ingombro del corpo rispetto alla marcia (vedi compute_safe_frontier), da passare
        quando il robot cammina senza girarsi.
        """
        with self._lock:
            self._waypoints = [(float(x), float(y)) for x, y in waypoints_xy]
            self._extend_end_m = extend_end_m
            self._body = body
            self._path_version += 1
            self._latest = None
            return self._path_version

    def clear_path(self):
        self.update_path([])

    def get_frontier(self):
        """Copia dell'ultimo fronte (dict) con 'path_version', oppure None."""
        with self._lock:
            return None if self._latest is None else dict(self._latest)

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        print("ArcVerificationTracker (fronte sicuro) avviato in background.")

    def stop(self):
        self._running = False
        print('ArcVerificationTracker stopped.')

    # --- ciclo in background -------------------------------------------------------------
    def _process_loop(self):
        while self._running:
            with self._lock:
                waypoints = list(self._waypoints)
                extend = self._extend_end_m
                body = self._body
                version = self._path_version
            if not waypoints:
                time.sleep(0.2)
                continue
            try:
                snap = self._fresh_snapshot()
                if snap is None:
                    time.sleep(0.2)
                    continue
                frontier = compute_safe_frontier([(snap.robot_x, snap.robot_y)] + waypoints, snap,
                                                 extend_end_m=extend, body=body)
                frontier['path_version'] = version
                with self._lock:
                    if version == self._path_version:   # nel frattempo non e' cambiato il percorso
                        self._latest = frontier
            except Exception as e:
                LOGGER.error(f"Errore nel calcolo del fronte: {e}")
            time.sleep(0.2)

    def _fresh_snapshot(self):
        grids_data, main_proto, _ = self._local_grid.return_local_grid(
            ['obstacle_distance', 'terrain', 'terrain_valid'],
            robot_state_client=self._robot_state_client)
        if grids_data is None or 'obstacle_distance' not in grids_data:
            return None

        pts = grids_data['obstacle_distance']['pts']
        cells_obs = grids_data['obstacle_distance']['values']
        terrain_vals = grids_data['terrain']['values']
        valid_vals = grids_data['terrain_valid']['values']
        num_x = main_proto.local_grid.extent.num_cells_x
        num_y = main_proto.local_grid.extent.num_cells_y
        cell_size = main_proto.local_grid.extent.cell_size

        # Posa del robot NELLO STESSO istante della griglia (stessa fotografia di trasformate)
        transforms_snapshot = main_proto.local_grid.transforms_snapshot
        vision_tform_body = get_a_tform_b(transforms_snapshot, VISION_FRAME_NAME, BODY_FRAME_NAME)
        robot_x, robot_y = vision_tform_body.position.x, vision_tform_body.position.y
        q = vision_tform_body.rotation
        robot_yaw = np.arctan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y ** 2 + q.z ** 2))

        footprint_mask_2d = self._local_grid.compute_robot_footprint_mask(
            pts, robot_x, robot_y, robot_yaw, num_x, num_y)
        terrain_corrected, is_valid, unwritten = self._local_grid.correct_terrain(
            terrain_vals, valid_vals, num_x, num_y,
            robot_footprint_mask=footprint_mask_2d, cell_size=cell_size,
            unwritten_value=grids_data['terrain'].get('raw_zero'),
            unwritten_scale=grids_data['terrain'].get('scale'), return_unwritten=True)
        _, rough_vals = self._local_grid.compute_gradient_and_roughness(
            terrain_corrected, valid_vals, num_x, num_y, cell_size, is_valid=is_valid)

        grid_frame_name = main_proto.local_grid.frame_name_local_grid_data
        vision_tform_grid = get_a_tform_b(transforms_snapshot, VISION_FRAME_NAME, grid_frame_name)
        return make_grid_snapshot(cells_obs, rough_vals, is_valid, valid_vals, terrain_corrected,
                                  num_x, num_y, vision_tform_grid.position.x,
                                  vision_tform_grid.position.y, cell_size, robot_x, robot_y,
                                  unwritten=unwritten)


# =============================================================================
# Funzioni di supporto (usate dalla ritirata e dai controlli puntuali).
# =============================================================================
def arc_visible_fraction(x1, y1, x2, y2, pts, fov_margin=0.0, num_samples=40):
    """
    Frazione in [0, 1] dell'arco, misurata A PARTIRE da (x1, y1), che ricade dentro il
    rettangolo della griglia locale. Non piu' usata dal ciclo principale (sostituita dal
    fronte sicuro), lasciata per i controlli puntuali.
    """
    if len(pts) == 0:
        return 0.0

    x_min, x_max = pts[:, 0].min() - fov_margin, pts[:, 0].max() + fov_margin
    y_min, y_max = pts[:, 1].min() - fov_margin, pts[:, 1].max() + fov_margin

    def _inside(px, py):
        return (x_min <= px <= x_max) and (y_min <= py <= y_max)

    if not _inside(x1, y1):
        return 0.0

    last_ok = 0.0
    for i in range(1, num_samples + 1):
        t = i / num_samples
        if _inside(x1 + t * (x2 - x1), y1 + t * (y2 - y1)):
            last_ok = t
        else:
            break
    return last_ok


def is_arc_in_fov(x1, y1, x2, y2, pts, fov_margin=0.0):
    """
    True se l'arco sta TUTTO dentro la finestra della griglia locale (~3.8 m). Non e' il
    campo visivo delle telecamere. Usata dalla ritirata.
    """
    if len(pts) == 0:
        return False

    x_min, x_max = pts[:, 0].min() - fov_margin, pts[:, 0].max() + fov_margin
    y_min, y_max = pts[:, 1].min() - fov_margin, pts[:, 1].max() + fov_margin

    p1_in_fov = (x_min <= x1 <= x_max) and (y_min <= y1 <= y_max)
    p2_in_fov = (x_min <= x2 <= x_max) and (y_min <= y2 <= y_max)
    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
    mid_in_fov = (x_min <= mid_x <= x_max) and (y_min <= mid_y <= y_max)

    return p1_in_fov and p2_in_fov and mid_in_fov


def verify_arc_safety(x1, y1, x2, y2, pts, obstacle_mask):
    """Controllo puntuale di un segmento sulla maschera fusa. Usato dalla ritirata."""
    return spotUtils.check_line_of_sight(x1, y1, x2, y2, pts, obstacle_mask)