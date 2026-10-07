import os
import sys
import csv
import json
import time
from time import sleep
from datetime import datetime
import numpy as np

import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
import bosdyn.geometry
from bosdyn.client.frame_helpers import *
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient, blocking_stand)
from bosdyn.client.local_grid import LocalGridClient
from bosdyn.client.frame_helpers import get_a_tform_b
from types import SimpleNamespace
import navGraphUtils
from bosdyn.api.graph_nav import graph_nav_pb2 as _graph_nav_pb2
import movements
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
import spotGrid
import spotLogInUtils
import environmentMap
import spotUtils
import arcVerification

import global_sampler
import prm_graph

import threading
from concurrent.futures import ThreadPoolExecutor

import matplotlib
# CRUCIALE: Impostare il backend 'Agg' PRIMA di importare pyplot!
# Questo disabilita Tkinter ed evita qualsiasi crash multithread/SIGABRT.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection

# Thread safety and background execution setup
_VIS_LOCK = threading.Lock()
_VIS_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vis_worker")

# ==================================================================
# STATISTICHE DI MISSIONE -- solo diagnostica/log, mai lette per decidere il
# comportamento del robot. Resettato all'inizio di ogni chiamata a easy_walk().
# I contatori sul veto/fonte-pendenza vivono invece sull'istanza PRM stessa
# (vedi prm_graph.PRM.print_mission_summary), perché il PRM persiste per tutta
# la missione mentre qui teniamo solo cio' che non ha un posto naturale li'.
# ==================================================================
MISSION_STATS = {
    'gait_tier_counts': {'PLAIN': 0, 'MODERATE': 0, 'HARD': 0},
    'replan_count': 0,
    'near_threshold_saves': 0,
    'segments_logged': 0,
    'last_tier': None,   # tier dell'ultimo segmento, letto per la riga del CSV
    'time_s': {},        # secondi spesi per fase, vedi _timing_add / [TEMPO]
    'scan_bundles': 0,
    'frontier_blocks': 0,
    'frontier_aborts': 0,
    'rotation_refusals': 0,   # 2026-10-07: rotazioni rifiutate dal controllo sul settore
    'straight_retreats': 0,   # 2026-10-07: arretramenti dritti dopo un rifiuto di rotazione
    'goal_repicks': 0,        # 2026-10-07: punti obiettivo riscelti dentro il ciclo
    'cone_moves': 0,          # 2026-10-07: avanzamenti dentro il cono di rotazione possibile
    'retreats_suppressed': 0,  # 2026-10-07: arretramenti evitati perche' inutili
    'early_replans': 0,       # 2026-10-07: ripianificazioni alla prima cella bloccante
    'traps_detected': 0,      # 2026-10-07: trappole riconosciute dal ventaglio a 360 gradi
    'arrival_refusals': 0,    # 2026-10-07: passi rifiutati perche' la posa d'arrivo non ha uscite
    'graphnav_backtracks': 0,     # 2026-10-07: ritorni di un waypoint riusciti con GraphNav
    'graphnav_backtrack_fail': 0, # 2026-10-07: ritorni con GraphNav falliti (ripiego a zampe)
    'nodes_pruned': 0,            # 2026-10-07: nodi PRM scartati perche' dentro un ostacolo
    'rotation_maneuvers': 0,
    'shortcuts': 0,
    'falls': 0,
    'self_rights': 0,
    'graphnav_ok': 0,
    'graphnav_fail': 0,
}


def _check_and_recover_fall(robot_state_client, command_client, global_map, where):
    """
    Controlla se il robot e' caduto e, se si', lo rialza (movements.recover_from_fall) e segna
    il punto come pericoloso nella mappa globale, cosi' il pianificatore lo evita.

    Restituisce None se non c'e' stata nessuna caduta, altrimenti (x, y) del punto in cui il
    robot e' caduto, a robot di nuovo in piedi. Solleva movements.RobotFallenError se non si
    puo' rialzare da solo: la missione si ferma e serve l'operatore.
    """
    try:
        st = movements.check_fall(robot_state_client)
    except Exception as e:
        print(f"[CADUTA] Impossibile leggere lo stato del robot ({where}): {e}")
        return None
    if not st['fallen']:
        return None
    MISSION_STATS['falls'] += 1
    print(f"[CADUTA] Rilevata {where} (caduta n. {MISSION_STATS['falls']} della missione).")
    movements.recover_from_fall(command_client, robot_state_client, status=st)
    MISSION_STATS['self_rights'] += 1
    if MISSION_STATS['falls'] >= MAX_FALLS_PER_MISSION:
        raise movements.RobotFallenError(
            f"{MISSION_STATS['falls']} cadute in questa missione: il robot si e' rialzato ma la missione "
            f"si ferma qui. Controllare il terreno e il robot prima di ripartire.")
    if global_map is not None:
        n = global_map.mark_hazard(st['x'], st['y'], FALL_HAZARD_RADIUS_M)
        print(f"[CADUTA] Punto della caduta ({st['x']:.2f}, {st['y']:.2f}) segnato come pericoloso "
              f"nella mappa: disco di {FALL_HAZARD_RADIUS_M:.1f} m ({n} celle), il pianificatore lo evitera'.")
    return (st['x'], st['y'])


def _timing_add(phase, seconds):
    """Accumula il tempo speso in una fase (riepilogo [TEMPO] a fine missione)."""
    MISSION_STATS['time_s'][phase] = MISSION_STATS['time_s'].get(phase, 0.0) + float(seconds)


def _save_npz_async(path, **arrays):
    """Salvataggio compresso nel thread di servizio: il ciclo principale non aspetta il disco."""
    def _job():
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.savez_compressed(path, **arrays)
        except Exception as e:
            print(f"[DATA SAVE] Impossibile salvare {os.path.basename(path)}: {e}")
    _VIS_EXECUTOR.submit(_job)

# Contesto della missione in corso, per i log diagnostici che non hanno modo di ricevere
# questi oggetti come argomento (attempt_enter_cell_from_position e' chiamata da piu' punti
# e non vogliamo cambiarne la firma). Riempito in easy_walk(), letto da _log_segment() e
# _emit_mission_summary_once(). Solo diagnostica: nessuna decisione del robot lo legge.
_MISSION_CTX = {
    'prm': None,
    'log_folder': None,
    'segment_log': None,
    'summary_done': False,
    # Quota del suolo all'AVVIO DELLA MISSIONE nel frame VISION (2026-10-06). Spot misura le
    # quote rispetto a dove e' stato ACCESO: se lo si porta alla zona di missione facendo
    # delle scale, tutte le quote risultano spostate (-3.2 m il 2026-10-05). x, y e direzione
    # sono gia' riferiti all'avvio missione (env.set_origin con la posa di partenza); per la
    # quota si salva questo valore in tutti i dati e nelle mappe, e il viewer mostra le quote
    # rispetto ad esso. I CALCOLI non ne dipendono: usano solo differenze di quota e il
    # suolo stimato sotto il robot a ogni scansione (vedi spotGrid.GROUND_HEIGHT_TOLERANCE_M).
    'mission_z0': None,
}


def _mission_z0():
    """Quota di riferimento della missione (NaN finche' non e' fissata), per i file salvati."""
    z0 = _MISSION_CTX.get('mission_z0')
    return np.nan if z0 is None else float(z0)


def _set_mission_z0(robot_state_client, z_body):
    """
    Fissa la quota di riferimento all'avvio missione: il piano del suolo stimato da Spot
    sotto i piedi (frame 'gpe'); se non disponibile, la quota del corpo meno l'altezza
    nominale in piedi. Stampa di quanto la quota di missione differisce da quella di accensione.
    """
    z0, source = None, "piano del suolo di Spot"
    try:
        snap = robot_state_client.get_robot_state().kinematic_state.transforms_snapshot
        z0 = float(get_a_tform_b(snap, VISION_FRAME_NAME, GROUND_PLANE_FRAME_NAME).position.z)
    except Exception as e:
        print(f"[QUOTE] Piano del suolo non disponibile ({e}): uso la quota del corpo.")
    if z0 is None or not np.isfinite(z0):
        z0, source = float(z_body) - 0.5, "quota del corpo - 0.5 m"
    _MISSION_CTX['mission_z0'] = z0
    note = ""
    if abs(z0) > 1.0:
        note = (" -- il robot e' stato acceso a un'altra quota (scale?). La navigazione non ne "
                "risente; il valore e' salvato come mission_z0 in ogni file e mission_viewer "
                "mostra le quote rispetto all'avvio missione")
    print(f"[QUOTE] Riferimento quote = suolo all'avvio missione: z={z0:.3f} m nel frame VISION "
          f"({source}), cioe' {z0:+.2f} m rispetto all'accensione{note}.")
    return z0

# ==================================================================
# NAVIGAZIONE CON IL FRONTE SICURO (2026-10-06) -- vedi decide_next_move() e il ciclo
# in attempt_enter_cell_from_position(). Le distanze del fronte stanno in spotGrid
# (FRONTIER_*), qui solo la logica del ciclo.
#   MIN_PARTIAL_STEP_M           -- passo minimo: sotto questa lunghezza non vale la pena
#                                   muoversi (si pagherebbe una scansione per pochi cm).
#   FRONTIER_BLOCK_CONFIRM_SCANS -- quante scansioni consecutive, a robot FERMO, senza poter
#                                   avanzare nemmeno del passo minimo, prima di dichiarare
#                                   un blocco. E' la conferma "da vicino" che mancava: il
#                                   vecchio verificatore condannava al primo scorcio.
#   FRONTIER_RESCAN_WAIT_S       -- pausa fra due scansioni di conferma.
#   MAX_BLOCK_REPLANS            -- dopo un blocco si ripianifica dal punto in cui si e'; se
#                                   succede piu' di queste volte nello stesso tentativo, ci
#                                   si arrende (ritirata, cella rimandata). Dal 2026-10-07
#                                   lo stesso contatore limita anche le riscelte del punto
#                                   obiettivo, cosi' non se ne aggiunge uno nuovo.
#   FRONTIER_ABORT_CONFIRM       -- a movimento in corso, quanti fronti consecutivi (~0.2 s
#                                   l'uno) devono dire "non arrivi dove stai andando" prima
#                                   di fermare il robot. 2 evita di inchiodare per un singolo
#                                   fotogramma sporco; a 0.5 m/s sono ~20 cm.
#   GOAL_REACHED_TOLERANCE_M     -- se il fronte impedisce gli ultimi centimetri verso
#                                   l'obiettivo (es. un muro appena oltre), ci si accontenta.
#   MAX_LOOP_ITERATIONS          -- salvaguardia anti-stallo del ciclo.
# ==================================================================
MIN_PARTIAL_STEP_M = 0.40
FRONTIER_BLOCK_CONFIRM_SCANS = 3
FRONTIER_RESCAN_WAIT_S = 0.3
MAX_BLOCK_REPLANS = 3
FRONTIER_ABORT_CONFIRM = 2
FRONTIER_ABORT_TOLERANCE_M = 0.05

# 2026-10-07: legata allo scostamento che il fronte stesso impone, invece di 0.60 fisso.
# allowed_advance() ferma il CENTRO del robot a FRONTIER_BODY_HALF_LENGTH_M +
# FRONTIER_MARGIN_M = 0.65 m prima del primo punto non libero; con la tolleranza a 0.60
# restava una banda morta di 5 cm in cui il robot non poteva ne' avanzare ne' dichiarare di
# essere arrivato. Misurato sulla missione del 2026-10-07 (iterazione 1, cella (0,1)): il
# robot si e' fermato con path_len 0.63-0.64 m, cioe' 3 cm fuori tolleranza, ha aspettato
# tre scansioni e ha dichiarato un blocco. Le due costanti devono essere la stessa cosa.
GOAL_REACHED_TOLERANCE_M = spotGrid.FRONTIER_BODY_HALF_LENGTH_M + spotGrid.FRONTIER_MARGIN_M
MAX_LOOP_ITERATIONS = 150

# ==================================================================
# ROTAZIONE E ARRETRAMENTO (2026-10-07). Sostituiscono la marcia di traverso.
#
# Decisione dell'utente: il robot punta SEMPRE il muso verso il punto da raggiungere.
# Niente marcia obliqua. Di traverso Spot e' largo 1.10 m invece di 0.50, quindi il
# fronte pretende 0.59 m di obstacle_distance invece di 0.30: il vecchio ripiego
# "non riesco a ruotare, vado di lato" rendeva il robot PIU' LARGO proprio dove lo
# spazio e' poco. Misurato sulle missioni del 2026-10-07: sei blocchi su percorsi che
# col muso in avanti erano liberi per 1.3-1.4 m sono nati solo cosi'.
#
# Quando la rotazione non e' possibile il robot ARRETRA IN LINEA RETTA lungo l'asse
# del corpo -- che non e' marcia obliqua: l'ingombro resta 0.50 m e la soglia del
# fronte resta 0.30 -- e al giro dopo riprova da li'. Alla seconda volta sullo stesso
# tratto il tratto si scarta e si ripianifica.
#   ROTATION_RETREAT_M        -- quanto si tenta di arretrare (il fronte puo' concederne meno)
#   ROTATION_RETREAT_MIN_M    -- sotto questo arretramento non vale la pena muoversi
#   MAX_ROTATION_FAILS_SAME_EDGE -- tentativi di arretramento sullo stesso tratto prima di scartarlo
# ==================================================================
ROTATION_RETREAT_M = 0.60
ROTATION_RETREAT_MIN_M = 0.20
MAX_ROTATION_FAILS_SAME_EDGE = 2

# --- Ventaglio dentro il cono di rotazione possibile (2026-10-07) -- vedi plan_cone_move.
CONE_SWEEP_MAX_DEG = 90.0      # quanto in la' si guarda, a destra e a sinistra del muso
CONE_SWEEP_STEP_DEG = 10.0     # passo della scansione angolare (9 direzioni per lato)
CONE_MIN_GAIN_M = 0.20         # una direzione vale solo se avvicina all'obiettivo di tanto

# Arretramenti consecutivi ammessi, CONTATI PER SEQUENZA e non per arco (2026-10-07).
# Il limite precedente, MAX_ROTATION_FAILS_SAME_EDGE, era indicizzato su
# (nodo_robot, nodo_successivo): ma ogni arretramento crea un nodo di sosta nuovo, quindi la
# chiave cambiava e il contatore ripartiva da 1 all'infinito. Nella missione del 2026-10-07
# si vede nel log: "tentativo 1/2" sette volte di fila, e "2/2" solo nelle due occasioni in
# cui il robot non era riuscito a spostarsi e la chiave era rimasta la stessa. Il limite
# non e' mai scattato, ed e' cosi' che sono nati i 3.5 m indietro.
MAX_STRAIGHT_RETREATS_IN_A_ROW = 2

# --- Ventaglio a 360 gradi: riconoscere la trappola (2026-10-07) -- vedi scan_escape_directions.
ESCAPE_FAN_STEP_DEG = 15.0        # 24 direzioni; 24 valutazioni del fronte, ~frazione di scansione
ESCAPE_FAN_MIN_ADVANCE_M = 0.30   # sotto questo spazio una direzione non e' una via d'uscita
# Controllo preventivo sulla posa d'arrivo (vedi arrival_has_exit): si paga solo dove serve,
# cioe' sui passi corti e verso spazi stretti, che sono i casi in cui si entra negli angoli.
ARRIVAL_EXIT_CHECK_MAX_STEP_M = 0.40
ARRIVAL_EXIT_CHECK_OD_M = 0.50
MAX_ARRIVAL_REFUSALS = 3          # per tentativo di cella, per non avvitarsi sul rifiuto

# --- Ritorni con GraphNav invece che camminando all'indietro (2026-10-07) ----------------
# Tornare indietro a zampe, all'indietro, sui tratti registrati e' la manovra piu' rischiosa
# che facciamo: il robot non vede dove va e non si rilocalizza. GraphNav cammina in avanti,
# con la localizzazione e l'anti-ostacolo di Boston Dynamics. Si lascia una traccia di
# waypoint lungo la strada (vedi graphnav_drop_breadcrumb) e si torna su quella.
GRAPHNAV_WAYPOINT_MIN_SPACING_M = 0.75   # distanza minima fra due briciole
GRAPHNAV_MAX_BACKTRACK_HOPS = 3          # poi si rimanda la cella e si passa alla priorita' dopo
# Un nodo PRM con meno di questo spazio libero non e' raggiungibile dal fronte: si scarta.
PRM_NODE_MIN_CLEARANCE_M = spotGrid.FRONTIER_CLEARANCE_M   # 0.30
# Ripianificazioni "precoci" (alla prima cella bloccante, senza aspettare la conferma da
# ferma) ammesse per tentativo di cella. Vedi il ramo 'wait'.
MAX_EARLY_REPLANS = 4

# ==================================================================
# DATI DI MISSIONE (2026-10-06, passi 7-8).
#   SAVE_FIGURES_DURING_MISSION -- False: durante la missione NON si disegna nulla. Ogni
#       chiamata a visualize_grid_with_candidates salva invece gli stessi dati in un .npz
#       (grafo PRM compreso), e le figure si producono DOPO con mission_viewer.py. Le figure
#       erano "asincrone" ma in un thread Python: disegnare decine di migliaia di archi in
#       matplotlib si contende il GIL con il ciclo principale e lo rallenta comunque. Nella
#       missione del 2026-10-05 16:05 su 15 minuti solo ~2 erano di movimento.
#       True: comportamento di prima.
#   SAVE_SCAN_BUNDLES -- un .npz per ogni scansione del ciclo di navigazione, nella
#       sottocartella scans/: layer GREZZI come arrivano da Spot (terrain, terrain_valid,
#       obstacle_distance), i derivati usati per decidere, la georeferenziazione (origine,
#       cella, posa del robot, suolo stimato), percorso, fronte e decisione. E' cio' che usa
#       mission_replay.py per rifare i calcoli con il codice nuovo sui dati di una missione
#       vecchia. ~0.3 MB a scansione.
# ==================================================================
SAVE_FIGURES_DURING_MISSION = False
SAVE_SCAN_BUNDLES = True

# Raggio del disco segnato come pericoloso nella mappa globale attorno al punto di una
# caduta (vedi _check_and_recover_fall e GlobalGrid.mark_hazard).
FALL_HAZARD_RADIUS_M = 0.5

# ==================================================================
# RITIRATA (vedi retreat_along_traveled_path).
#   RETREAT_MAX_SEGMENTS  -- quanti tratti gia' percorsi ripercorrere a ritroso al
#                            massimo. Pochi: serve a uscire da una tasca, non a
#                            tornare al punto di partenza.
#   RETREAT_MIN_SEGMENT_M -- sotto questa lunghezza il tratto si salta (non vale un
#                            comando di movimento a se').
#   RETREAT_AFTER_ABORT_M -- quanto arretrare subito dopo un arresto a meta'
#                            movimento, per staccarsi da cio' che ha fatto fermare.
# ==================================================================
RETREAT_MAX_SEGMENTS = 3
RETREAT_MIN_SEGMENT_M = 0.20
RETREAT_AFTER_ABORT_M = 0.40
#   RETREAT_CONTINUITY_M  -- (2026-10-06 sera) si arretra lungo un tratto registrato solo se
#                            il robot sta davvero dove quel tratto finiva. Dopo uno spostamento
#                            con GraphNav fra celle (che non registra tratti) l'ultimo tratto
#                            registrato poteva stare a molti metri: il robot ci andava
#                            camminando all'indietro, alla cieca (rilievo della revisione).
#   RETREAT_MAX_SEGMENT_M -- tratto piu' lungo di cosi': non lo si ripercorre all'indietro.
RETREAT_CONTINUITY_M = 0.50
RETREAT_MAX_SEGMENT_M = 2.50


def _slope_edge_color(node_id, neighbor_id, edge_slope_info):
    """
    Colore di un arco del grafo PRM in base a quanto la sua pendenza sia LATERALE
    (sidehill/rollio, più rischioso -- vedi beta_lateral in prm_graph.PRM) rispetto a
    LONGITUDINALE (beccheggio). Verde = soprattutto longitudinale o poco ripido,
    arancione/rosso = pendenza soprattutto laterale e vicina alla soglia di veto.
    Grigio = arco senza dato di pendenza registrato (mai valutato, o scartato solo per
    occupazione) -- stesso colore neutro usato prima di questa modifica, nessun dato
    inventato. Riusa self.edge_slope_info già calcolato dal veto -- zero ricalcolo qui.
    """
    key = (min(node_id, neighbor_id), max(node_id, neighbor_id))
    info = edge_slope_info.get(key)
    if info is None:
        return (0.5, 0.5, 0.5, 0.3)  # grigio neutro, comportamento invariato

    long_s, lat_s = info
    total = long_s + lat_s
    if total < 1e-6:
        return (0.6, 0.6, 0.6, 0.25)  # terreno piatto, niente da evidenziare

    lateral_share = lat_s / total
    severity = min(1.0, max(long_s, lat_s) / spotGrid.SLOPE_THRESHOLD) if spotGrid.SLOPE_THRESHOLD > 0 else 0.0

    cmap = matplotlib.colormaps.get_cmap('RdYlGn_r')  # rosso = alto valore, verde = basso
    r, g, b, _ = cmap(lateral_share)
    alpha = 0.3 + 0.5 * severity  # più ripido -> più visibile, meno ripido -> discreto
    return (r, g, b, alpha)


# =========================================================================
# STRUMENTI DIAGNOSTICI -- tutto quello che segue serve SOLO a registrare cosa succede
# durante la missione. Nessuna decisione del robot dipende da questi oggetti, e ogni
# funzione e' scritta per non sollevare mai eccezioni verso il chiamante: un log che
# fallisce non deve mai fermare o alterare un movimento.
# =========================================================================

class _TeeStream:
    """
    Copia tutto quello che viene stampato su un file, mantenendo anche la stampa normale a
    schermo. Nel file ogni riga e' preceduta da un'ora (HH:MM:SS.mmm), utile per allineare i
    log con un eventuale video o con le note prese a mano durante la prova.
    """
    def __init__(self, original, file_path):
        self._orig = original
        self._file = open(file_path, 'a', buffering=1, encoding='utf-8')
        self._lock = threading.Lock()
        self._at_line_start = True

    def write(self, s):
        try:
            self._orig.write(s)
        except Exception:
            pass
        try:
            with self._lock:
                out = []
                for chunk in s.splitlines(keepends=True):
                    if self._at_line_start:
                        out.append(datetime.now().strftime('%H:%M:%S.%f')[:-3] + ' ')
                    out.append(chunk)
                    self._at_line_start = chunk.endswith('\n')
                if out:
                    self._file.write(''.join(out))
        except Exception:
            pass
        return len(s)

    def flush(self):
        try:
            self._orig.flush()
        except Exception:
            pass
        try:
            with self._lock:
                self._file.flush()
        except Exception:
            pass

    def close_file(self):
        try:
            with self._lock:
                self._file.close()
        except Exception:
            pass

    def __getattr__(self, name):
        # isatty, encoding, fileno... passano allo stream originale
        return getattr(self._orig, name)


def _install_stdout_tee(file_path):
    """Da qui in poi ogni print finisce anche in file_path. Idempotente."""
    try:
        if not isinstance(sys.stdout, _TeeStream):
            sys.stdout = _TeeStream(sys.stdout, file_path)
            print(f"[LOG] Copia completa dell'output su file: {file_path}")
    except Exception as e:
        print(f"[LOG] Impossibile attivare la copia dell'output su file: {e}")


def _remove_stdout_tee():
    try:
        if isinstance(sys.stdout, _TeeStream):
            tee = sys.stdout
            sys.stdout = tee._orig
            tee.close_file()
    except Exception:
        pass


def _quat_to_roll_pitch_deg(q):
    """Roll e pitch (gradi) di un quaternione con attributi w, x, y, z. Nel frame VISION,
    che e' allineato alla gravita', sono l'inclinazione reale del corpo rispetto
    all'orizzontale (stessa convenzione con cui il file calcola gia' lo yaw)."""
    roll = np.arctan2(2.0 * (q.w * q.x + q.y * q.z), 1.0 - 2.0 * (q.x ** 2 + q.y ** 2))
    pitch = np.arcsin(np.clip(2.0 * (q.w * q.y - q.z * q.x), -1.0, 1.0))
    return float(np.degrees(roll)), float(np.degrees(pitch))


class BodyTiltRecorder:
    """
    Registra l'inclinazione REALE del corpo di Spot (roll e pitch) mentre esegue un
    movimento, interrogando lo stato del robot circa 5 volte al secondo in un thread a parte.
    Serve a confrontare cio' che il pianificatore stimava (pendenza del terreno) con cio' che
    Spot ha effettivamente sentito: e' il dato che permette di ritarare soglie e pesi.

    Uso:
        rec = BodyTiltRecorder(robot_state_client); rec.start()
        ... movimento ...
        stats = rec.stop()   # dict con inizio, fine, picchi, numero di campioni

    Nota di lettura: Spot puo' livellare il corpo rispetto al terreno, quindi l'inclinazione
    del corpo e' un indicatore indiretto della pendenza, non la pendenza stessa. Ogni errore
    di lettura viene contato e ignorato.
    """
    def __init__(self, robot_state_client, period_s=0.2):
        self._client = robot_state_client
        self._period = period_s
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        self._start = None
        self._last = None
        self._peak_roll = 0.0
        self._peak_pitch = 0.0
        self._n = 0
        self._errors = 0
        # Traiettoria reale campionata (2026-10-07): (t, x, y, yaw, roll, pitch). Finisce in
        # trajectory.csv, per confrontare dove il robot e' andato davvero con cio' che era
        # stato comandato.
        self.samples = []

    def _sample(self):
        try:
            x, y, _, q = spotUtils.getPosition(self._client)
            roll, pitch = _quat_to_roll_pitch_deg(q)
            yaw = float(np.degrees(np.arctan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y ** 2 + q.z ** 2))))
        except Exception:
            with self._lock:
                self._errors += 1
            return None
        with self._lock:
            if self._start is None:
                self._start = (roll, pitch)
            self._last = (roll, pitch)
            self._peak_roll = max(self._peak_roll, abs(roll))
            self._peak_pitch = max(self._peak_pitch, abs(pitch))
            self._n += 1
            self.samples.append((time.time(), float(x), float(y), yaw, roll, pitch))
        return roll, pitch

    def _run(self):
        while not self._stop.wait(self._period):
            self._sample()

    def start(self):
        try:
            self._sample()
            self._thread = threading.Thread(target=self._run, name="body_tilt_recorder", daemon=True)
            self._thread.start()
        except Exception:
            pass

    def stop(self):
        try:
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=1.0)
            self._sample()
        except Exception:
            pass
        with self._lock:
            return {
                'roll_start': self._start[0] if self._start else None,
                'pitch_start': self._start[1] if self._start else None,
                'roll_end': self._last[0] if self._last else None,
                'pitch_end': self._last[1] if self._last else None,
                'roll_peak_abs': self._peak_roll if self._n else None,
                'pitch_peak_abs': self._peak_pitch if self._n else None,
                'samples': self._n,
                'errors': self._errors,
            }


_SEGMENT_COLUMNS = [
    'timestamp', 'iteration', 'cell_row', 'cell_col', 'node_from', 'node_to',
    'x_from', 'y_from', 'x_to', 'y_to', 'dist_m',
    'slope_source', 'long_slope', 'lat_slope', 'long_interp', 'lat_interp',
    'roughness', 'coverage_along', 'coverage_lateral',
    'cost_dist', 'cost_long', 'cost_lat', 'cost_rough', 'cost_total',
    'tier', 'gait_override', 'success', 'dist_traveled_m', 'dist_commanded_m', 'duration_s',
    'roll_start_deg', 'pitch_start_deg', 'roll_end_deg', 'pitch_end_deg',
    'roll_peak_abs_deg', 'pitch_peak_abs_deg', 'tilt_samples', 'tilt_errors',
]


class SegmentLog:
    """CSV con una riga per ogni segmento (da un nodo al successivo) effettivamente
    percorso o tentato. Si apre in un foglio di calcolo. Una riga viene scritta e scaricata su
    disco subito, cosi' anche una missione interrotta lascia i dati fino all'ultimo segmento."""
    def __init__(self, path):
        self._path = path
        self._lock = threading.Lock()
        with open(path, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(_SEGMENT_COLUMNS)

    def write_row(self, row_dict):
        try:
            def fmt(v):
                if v is None:
                    return ''
                if isinstance(v, (float, np.floating)):
                    return f"{float(v):.4f}"
                return v
            with self._lock:
                with open(self._path, 'a', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerow([fmt(row_dict.get(c)) for c in _SEGMENT_COLUMNS])
        except Exception as e:
            print(f"[LOG] Impossibile scrivere la riga del segmento: {e}")


_CSV_LOCK = threading.Lock()


def _append_csv(name, header, rows):
    """Aggiunge righe a un CSV nella cartella dei log (intestazione alla prima scrittura). Mai eccezioni."""
    folder = _MISSION_CTX.get('log_folder')
    if not folder or not rows:
        return
    try:
        path = os.path.join(folder, name)
        with _CSV_LOCK:
            new = not os.path.exists(path)
            with open(path, 'a', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                if new:
                    w.writerow(header)
                for r in rows:
                    w.writerow([f"{v:.4f}" if isinstance(v, (float, np.floating)) else v for v in r])
    except Exception as e:
        print(f"[LOG] Impossibile scrivere {name}: {e}")


def _log_segment(row_dict):
    seg_log = _MISSION_CTX.get('segment_log')
    if seg_log is not None:
        seg_log.write_row(row_dict)
        MISSION_STATS['segments_logged'] += 1


def _segment_diagnostics(terrain_2d, origin_x, origin_y, cell_size, x1, y1, x2, y2):
    """Stima alternativa (interpolata) e copertura dei campioni per un segmento. Solo log:
    la stima in uso per decidere resta quella di PRM.sample_terrain_between_points."""
    out = {'long_interp': None, 'lat_interp': None, 'coverage_along': None, 'coverage_lateral': None}
    try:
        li, lai, ok = spotGrid.compute_arc_slope_profile_interp(
            terrain_2d, origin_x, origin_y, cell_size, x1, y1, x2, y2)
        if ok:
            out['long_interp'], out['lat_interp'] = li, lai
        out['coverage_along'], out['coverage_lateral'] = spotGrid.compute_arc_sample_coverage(
            terrain_2d, origin_x, origin_y, cell_size, x1, y1, x2, y2)
    except Exception as e:
        print(f"[LOG] Diagnostica del segmento non disponibile: {e}")
    return out


def _raw_zero(grids_data):
    """Quota del valore grezzo 0 del layer terrain di questa scansione (None se assente)."""
    try:
        return grids_data['terrain'].get('raw_zero')
    except (KeyError, TypeError, AttributeError):
        return None


def _raw_scale(grids_data):
    try:
        return grids_data['terrain'].get('scale')
    except (KeyError, TypeError, AttributeError):
        return None


# ==================================================================
# SPOSTAMENTI CON GRAPHNAV (2026-10-06 sera, dopo la revisione) -- fra celle lontane e per il
# ritorno a wp_0. Sostituiscono, solo per le chiamate di questo file,
# RecordingInterface.navigate_to_waypoint / navigate_to_first_waypoint (navGraphUtils.py non
# e' modificato), che:
#   - non avevano tempo massimo (navigate_to_first_waypoint ripeteva all'infinito anche con il
#     robot caduto o senza percorso, e la missione non arrivava mai a sedersi);
#   - usavano TravelParams vuoti: nessun limite di velocita', cioe' gli spostamenti piu' lunghi
#     della missione alla velocita' di default di Spot;
#   - su STATUS_LOST forzavano la localizzazione SUL WAYPOINT DI DESTINAZIONE e dichiaravano
#     "arrivato": il robot poteva trovarsi altrove. Ora STATUS_LOST = fallimento, robot fermo.
# ==================================================================
MAX_FALLS_PER_MISSION = 3      # alla terza caduta il robot si rialza e la missione si ferma
GRAPHNAV_TIMEOUT_S = 240.0
GRAPHNAV_MAX_LINEAR_VEL_MPS = 0.5
GRAPHNAV_MAX_ANGULAR_VEL_RPS = 0.6
_GN = _graph_nav_pb2.NavigationFeedbackResponse
_GRAPHNAV_FAIL_STATUSES = {
    _GN.STATUS_STUCK: "bloccato (STUCK)", _GN.STATUS_NO_ROUTE: "nessun percorso nel grafo",
    _GN.STATUS_NO_LOCALIZATION: "non localizzato", _GN.STATUS_ROBOT_IMPAIRED: "robot in difficolta' (caduto?)",
    _GN.STATUS_CONSTRAINT_FAULT: "vincolo violato", _GN.STATUS_COMMAND_OVERRIDDEN: "comando sostituito",
    _GN.STATUS_NOT_LOCALIZED_TO_ROUTE: "non localizzato sul percorso", _GN.STATUS_LEASE_ERROR: "errore di lease",
    _GN.STATUS_AREA_CALLBACK_ERROR: "errore di area callback",
}


def _graphnav_navigate_to(recordingInterface, waypoint_id, robot_state_client, command_client, label,
                          timeout_s=GRAPHNAV_TIMEOUT_S):
    """Porta il robot a un waypoint con GraphNav. True solo se ci arriva davvero."""
    client = recordingInterface._graph_nav_client
    tp = _graph_nav_pb2.TravelParams()
    tp.velocity_limit.CopyFrom(movements._velocity_limit(GRAPHNAV_MAX_LINEAR_VEL_MPS, GRAPHNAV_MAX_ANGULAR_VEL_RPS))
    print(f"[GRAPHNAV] {label}: avvio (massimo {timeout_s:.0f} s, {GRAPHNAV_MAX_LINEAR_VEL_MPS:.1f} m/s).")
    t0, cmd_id, fb_errors, nav_errors = time.time(), None, 0, 0
    while time.time() - t0 < timeout_s:
        try:
            cmd_id = client.navigate_to(waypoint_id, 1.0, command_id=cmd_id, travel_params=tp)
            nav_errors = 0
        except Exception as e:
            nav_errors += 1
            print(f"[GRAPHNAV] {label}: comando rifiutato ({nav_errors}/3): {e}")
            # 2026-10-07: qui il `return False` stava FUORI dall'if, quindi un solo errore
            # RPC transitorio abbandonava lo spostamento mentre il messaggio diceva "(1/3)",
            # e le tre righe sotto erano codice morto. I contatori non arrivavano mai a 2.
            if nav_errors >= 3:
                movements.stop_robot(command_client, reason="GraphNav: comando rifiutato")
                MISSION_STATS['graphnav_fail'] += 1
                return False
            cmd_id = None
            time.sleep(0.5)
            continue
        time.sleep(0.5)
        try:
            status = client.navigation_feedback(cmd_id).status
            fb_errors = 0
        except Exception as e:
            fb_errors += 1
            print(f"[GRAPHNAV] {label}: errore nel feedback ({fb_errors}/3): {e}")
            # 2026-10-07: stesso difetto del ramo sopra -- `return False` fuori dall'if.
            if fb_errors >= 3:
                movements.stop_robot(command_client, reason="GraphNav: feedback non disponibile")
                MISSION_STATS['graphnav_fail'] += 1
                return False
            cmd_id = None      # il comando potrebbe non essere piu' valido: se ne invia uno nuovo
            time.sleep(0.5)
            continue
        if status == _GN.STATUS_REACHED_GOAL:
            print(f"[GRAPHNAV] {label}: arrivato in {time.time() - t0:.0f} s.")
            MISSION_STATS['graphnav_ok'] += 1
            return True
        if status == _GN.STATUS_LOST:
            # Non si forza la localizzazione sulla destinazione (vedi sopra): fallimento.
            print(f"[GRAPHNAV] {label}: GraphNav ha perso la localizzazione. Fermo il robot.")
            movements.stop_robot(command_client, reason="GraphNav: robot perso")
            MISSION_STATS['graphnav_fail'] += 1
            return False
        if status == _GN.STATUS_COMMAND_TIMED_OUT:
            cmd_id = None          # comando scaduto: se ne invia uno nuovo
            continue
        if status in _GRAPHNAV_FAIL_STATUSES:
            print(f"[GRAPHNAV] {label}: interrotto -- {_GRAPHNAV_FAIL_STATUSES[status]}.")
            movements.stop_robot(command_client, reason=f"GraphNav: {_GRAPHNAV_FAIL_STATUSES[status]}")
            MISSION_STATS['graphnav_fail'] += 1
            return False
    print(f"[GRAPHNAV] {label}: tempo massimo di {timeout_s:.0f} s superato. Fermo il robot.")
    movements.stop_robot(command_client, reason="GraphNav: tempo massimo superato")
    MISSION_STATS['graphnav_fail'] += 1
    return False


def _graphnav_return_to_start(recordingInterface, robot_state_client, command_client):
    """Ritorno a wp_0 a fine missione, con tempo massimo. False se non ci riesce."""
    try:
        graph = recordingInterface._get_graph(force_refresh=True)
        wp0 = next((w for w in graph.waypoints if w.annotations.name == "wp_0"), None)
    except Exception as e:
        print(f"[GRAPHNAV] Ritorno a wp_0: grafo non disponibile ({e}).")
        return False
    if wp0 is None:
        print("[GRAPHNAV] Ritorno a wp_0: waypoint wp_0 non trovato nel grafo.")
        return False
    return _graphnav_navigate_to(recordingInterface, wp0.id, robot_state_client, command_client,
                                 "ritorno a wp_0", timeout_s=2 * GRAPHNAV_TIMEOUT_S)


def _mask_for_global_map(obstacle_mask, is_valid):
    """
    Maschera da passare a GlobalGrid.update: -1 ostacolo, +1 vista libera, 0 nessuna
    informazione (cella non attendibile in questa scansione). Prima le celle non attendibili
    arrivavano come +1, cioe' "viste libere" (vedi GlobalGrid.update).
    """
    m = np.asarray(obstacle_mask, dtype=np.float64).ravel()
    v = np.asarray(is_valid, dtype=bool).ravel()
    return np.where(m < 0, -1.0, np.where(v, 1.0, 0.0))


def _log_ground_filter(local_grid, label):
    """
    Una riga solo quando il filtro sull'altezza del suolo ha scartato qualcosa o non ha
    potuto stimare il suolo -- nel caso normale tace. Vedi spotGrid.GROUND_HEIGHT_TOLERANCE_M.
    """
    info = getattr(local_grid, 'last_ground_filter', None) or {}
    if info.get('ground_z') is None:
        print(f"[GROUND-FILTER] {label}: suolo non stimabile ({info.get('source')}, "
              f"{info.get('n_ref', 0)} celle di riferimento) -- filtro non applicato")
    elif info.get('n_above', 0) > 0:
        print(f"[GROUND-FILTER] {label}: {info['n_above']} celle scartate, fino a "
              f"{info['max_above']:.2f} m sopra il suolo (suolo z={info['ground_z']:.3f}, "
              f"stimato {info['source']} su {info['n_ref']} celle)")
    if info.get('n_unwritten', 0) > 0:
        print(f"[GROUND-FILTER] {label}: {info['n_unwritten']} celle mai scritte dai sensori "
              f"({info.get('n_unwritten_below', 0)} sotto il suolo) -> trattate come non viste"
              + (f", quota {info['quota_unwritten']:+.4f} m" if info.get('quota_unwritten') is not None else ""))


def _log_footprint_diagnostic(local_grid, pts, robot_x, robot_y, robot_yaw,
                              num_x, num_y, obstacle_mask, label,
                              cells_obstacle_dist=None, rough_values=None, is_valid=None):
    """
    Misura quante celle sotto l'impronta del robot risultano ostacolo, e PERCHE'.
    PURAMENTE OSSERVATIVO: usa diagnostic_only=True, quindi non consuma il flag
    one-shot di compute_robot_footprint_mask e non cambia nessun comportamento.

    AGGIORNAMENTO 2026-10-07 -- la "questione A" del CONTEXT era chiusa sulla base
    sbagliata. Diceva "era prossimita', non auto-visione, e zero celle nel nucleo".
    Sui dati del 2026-10-07 si misurano invece 190-191 celle NEL NUCLEO, tutte per
    RUGOSITA'. La causa non e' l'auto-visione: sotto il corpo i sensori non possono
    scrivere, quelle celle restano alla quota del plateau delle celle mai scritte, e il
    salto di ~0.93 m rispetto al pavimento fa esplodere la deviazione standard della
    finestra 5x5. Con _cells_never_written corretta (vedi spotGrid, 2026-10-07) scendono
    a ZERO. Quindi la maschera one-shot resta com'e': applicarla a ogni scansione
    nasconderebbe muri veri in un corridoio stretto, e il problema che sembrava
    richiederla non esisteva.

    PERCHE' -- i due veti di fuse_obstacle_mask, ricalcolati identici:
      - "prossimita'": cells_obstacle_dist <= OBSTACLE_THRESHOLD. NON vuol dire
        "qui c'e' un ostacolo", vuol dire "qui sei a meno di 15 cm da un ostacolo".
        Se domina questo, il footprint NON c'entra: il robot sta davvero passando
        vicino a qualcosa, e mascherare cancellerebbe proprio quell'avviso.
      - "rugosita'": rough_values > ROUGH_THRESHOLD su celle che il sensore dichiara
        valide.

    DOVE -- rettangolo intero (1.10 x 0.50 + margine) contro il solo NUCLEO
    (0.90 x 0.35), cioe' la fascia entro la carreggiata delle zampe. Un ostacolo
    dentro il nucleo significherebbe che il robot ci sta gia' sopra, quindi mascherare li'
    non puo' nascondere nulla di raggiungibile, mentre mascherare l'intero rettangolo in un
    corridoio stretto puo' cancellare un muro vero.

    Nota: anche arcVerification.py chiama compute_robot_footprint_mask, quindi in
    condizioni di corsa puo' essere il thread di background a consumare l'unica
    maschera prevista, lasciando senza il ciclo principale.
    """
    try:
        fp = local_grid.compute_robot_footprint_mask(
            pts, robot_x, robot_y, robot_yaw, num_x, num_y, diagnostic_only=True)
        if fp is None:
            return
        fp_flat = fp.ravel()
        obs_flat = np.asarray(obstacle_mask).ravel()
        n_fp = int(fp_flat.sum())
        if n_fp == 0:
            return
        blocked = (obs_flat < 0) & fp_flat
        n_obs = int(blocked.sum())

        line = (f"[FOOTPRINT-DIAG] {label}: celle sotto il corpo marcate ostacolo: "
                f"{n_obs}/{n_fp} ({100.0 * n_obs / n_fp:.1f}%)")

        # --- PERCHE': i due veti separati, stessa identica logica di fuse_obstacle_mask ---
        if n_obs > 0 and cells_obstacle_dist is not None and rough_values is not None \
                and is_valid is not None:
            d = np.asarray(cells_obstacle_dist).ravel()
            r = np.asarray(rough_values).ravel()
            v = np.asarray(is_valid).ravel()
            if d.shape == fp_flat.shape and r.shape == fp_flat.shape and v.shape == fp_flat.shape:
                v1 = (d <= spotGrid.OBSTACLE_THRESHOLD) & v   # prossimita' a un ostacolo
                v3 = (r > spotGrid.ROUGH_THRESHOLD) & v       # terreno irregolare
                n_prox = int(np.sum(v1 & fp_flat & ~v3))
                n_rough = int(np.sum(v3 & fp_flat & ~v1))
                n_both = int(np.sum(v1 & v3 & fp_flat))
                n_invalid = int(np.sum(~v & fp_flat))
                line += (f"\n                 perche': prossimita'={n_prox}  rugosita'={n_rough}  "
                         f"entrambi={n_both}   (celle non valide sotto il corpo: {n_invalid})")

        # --- DOVE: nucleo (dentro la carreggiata delle zampe) contro corona esterna ---
        if n_obs > 0:
            core = local_grid.compute_robot_footprint_mask(
                pts, robot_x, robot_y, robot_yaw, num_x, num_y,
                diagnostic_only=True, body_length=0.90, body_width=0.35)
            if core is not None:
                core_flat = core.ravel()
                n_core_fp = int(core_flat.sum())
                n_core_obs = int(np.sum(blocked & core_flat))
                n_ring_obs = n_obs - n_core_obs
                line += (f"\n                 dove: nucleo={n_core_obs}/{n_core_fp}  "
                         f"corona esterna={n_ring_obs}/{n_fp - n_core_fp}")

        print(line)
    except Exception as e:
        print(f"[FOOTPRINT-DIAG] diagnostica non disponibile: {e}")


def _emit_mission_summary_once():
    """Stampa e salva il riepilogo di missione una sola volta. Viene chiamata sia a fine
    missione normale sia da main() in un blocco finally, cosi' esce anche se la missione
    termina per un'eccezione o per una ripresa del controllo."""
    if _MISSION_CTX.get('summary_done') or _MISSION_CTX.get('prm') is None:
        return
    _MISSION_CTX['summary_done'] = True
    try:
        prm = _MISSION_CTX['prm']
        prm.print_mission_summary()
        total_gait = sum(MISSION_STATS['gait_tier_counts'].values())
        print("================ [MISSION STATS] ================")
        print(f"Distribuzione tier di gait (totale {total_gait} passi):")
        for tier, count in MISSION_STATS['gait_tier_counts'].items():
            pct = (100.0 * count / total_gait) if total_gait > 0 else 0.0
            print(f"  {tier:10s}: {count:6d}  ({pct:5.1f}%)")
        print(f"Ripianificazioni totali: {MISSION_STATS['replan_count']}")
        print(f"Segmenti registrati nel CSV: {MISSION_STATS['segments_logged']}")
        print(f"Salvataggi .npz per archi vicino soglia: {MISSION_STATS['near_threshold_saves']}")
        print(f"Blocchi confermati dal fronte: {MISSION_STATS['frontier_blocks']}, "
              f"arresti a meta' movimento: {MISSION_STATS['frontier_aborts']}, "
              f"rotazioni rifiutate: {MISSION_STATS['rotation_refusals']}, "
              f"arretramenti dritti: {MISSION_STATS['straight_retreats']}, "
              f"obiettivi riscelti: {MISSION_STATS['goal_repicks']}, "
              f"scorciatoie: {MISSION_STATS['shortcuts']}")
        print(f"Avanzamenti nel cono di rotazione: {MISSION_STATS['cone_moves']}, "
              f"arretramenti evitati perche' inutili: {MISSION_STATS['retreats_suppressed']}, "
              f"ripianificazioni immediate: {MISSION_STATS['early_replans']}")
        print(f"Ritorni con GraphNav: {MISSION_STATS['graphnav_backtracks']} riusciti, "
              f"{MISSION_STATS['graphnav_backtrack_fail']} falliti (ripiego a zampe); "
              f"nodi PRM scartati perche' dentro ostacoli: {MISSION_STATS['nodes_pruned']}")
        print(f"Trappole riconosciute dal ventaglio: {MISSION_STATS['traps_detected']}, "
              f"passi rifiutati per mancanza di uscite all'arrivo: {MISSION_STATS['arrival_refusals']}")
        print(f"Cadute: {MISSION_STATS['falls']}, rialzate da solo: {MISSION_STATS['self_rights']}")
        print(f"Spostamenti GraphNav riusciti: {MISSION_STATS['graphnav_ok']}, falliti: {MISSION_STATS['graphnav_fail']}")
        lg_main = _MISSION_CTX.get('local_grid')
        lg_trk = getattr(_MISSION_CTX.get('tracker'), '_local_grid', None)
        print(f"Scansioni scartate (layer mancanti o non allineati): ciclo principale "
              f"{getattr(lg_main, 'rejected_scans', 0)}, controllo in movimento {getattr(lg_trk, 'rejected_scans', 0)}")
        print(f"Pacchetti di scansione salvati: {MISSION_STATS['scan_bundles']}")
        times = MISSION_STATS['time_s']
        if times:
            tot = sum(times.values())
            print(f"[TEMPO] Ripartizione del tempo misurato ({tot:.0f} s):")
            for phase, sec in sorted(times.items(), key=lambda kv: -kv[1]):
                print(f"  {phase:22s} {sec:8.1f} s  ({100.0 * sec / tot:5.1f}%)")
        print("===================================================\n")

        folder = _MISSION_CTX.get('log_folder')
        if folder:
            summary = {
                'prm': prm.get_mission_summary_dict(),
                'mission_stats': {k: v for k, v in MISSION_STATS.items()},
                'lateral_slope_cost_multiplier': spotGrid.LATERAL_SLOPE_COST_MULTIPLIER,
                'slope_threshold': spotGrid.SLOPE_THRESHOLD,
                'near_threshold_margin_fraction': spotGrid.NEAR_THRESHOLD_MARGIN_FRACTION,
                'slope_baseline_m': spotGrid.SLOPE_BASELINE_M,
                'lateral_slice_half_width_m': spotGrid.LATERAL_SLICE_HALF_WIDTH_M,
                'min_arc_sample_fraction': spotGrid.MIN_ARC_SAMPLE_FRACTION,
                'terrain_valid_min': spotGrid.TERRAIN_VALID_MIN,
                'goal_reached_tolerance_m': GOAL_REACHED_TOLERANCE_M,
            }
            with open(os.path.join(folder, 'mission_summary.json'), 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, default=str)
    except Exception as e:
        print(f"[LOG] Impossibile produrre il riepilogo di missione: {e}")

# TODO: check if we can avoid to set a sleep after each movement command
# TODO: change the folder destination of the name download of graph

def sample_cell_points(env, cell_row, cell_col, num_samples=200):
    """Sample random points within a cell."""
    world_pos = env.get_world_position_from_cell(cell_row, cell_col)
    if world_pos is None:
        return []

    cell_center_x, cell_center_y = world_pos
    half_size = env.cell_size / 2.0

    samples = []
    for _ in range(num_samples):
        offset_x = np.random.uniform(-half_size * 0.8, half_size * 0.8)
        offset_y = np.random.uniform(-half_size * 0.8, half_size * 0.8)

        cos_yaw = np.cos(env.origin_yaw)
        sin_yaw = np.sin(env.origin_yaw)

        world_offset_x = offset_x * cos_yaw - offset_y * sin_yaw
        world_offset_y = offset_x * sin_yaw + offset_y * cos_yaw

        sample_x = cell_center_x + world_offset_x
        sample_y = cell_center_y + world_offset_y

        samples.append((sample_x, sample_y))

    return samples

def find_best_point_in_cell(robot_x, robot_y, env, cell_row, cell_col, pts, cells_obstacle_dist,
                            global_sampler, global_map=None, safety_margin=None, exclude=None):
    """
    Sceglie il punto obiettivo dentro la cella: il piu' CENTRALE fra quelli non occupati.

    Prima questa funzione restituiva SEMPRE il centro geometrico esatto della cella,
    ignorando del tutto `pts` e `cells_obstacle_dist` che pure riceveva. Con celle da
    5x5 m in una stanza arredata, il centro geometrico cade spesso dentro o a ridosso di
    un mobile: il robot veniva mandato verso un punto irraggiungibile, tutti gli archi
    che ci arrivavano erano vetati, e l'unico percorso superstite diventava un giro
    largo. E' il meccanismo dietro il percorso 176 -> 211 -> 3127 della missione del
    2026-10-05 13:12, che superava il bersaglio di oltre un metro per poi tornare
    indietro, finendo in una tasca stretta.

    Ordine di preferenza, a parita' di vicinanza al centro:
      1. punti OSSERVATI e liberi      -- sappiamo che si puo' stare li'
      2. punti MAI OSSERVATI           -- non sappiamo, ma non abbiamo nulla contro
      3. (mai scelti) punti occupati
    Se tutti i candidati risultano occupati la cella non ha un obiettivo valido e la
    funzione restituisce (None, None, ...): il chiamante lo tratta gia' come fallimento
    di ingresso e passa alla logica dei lati non ancora provati.

    Quando la cella non e' mai stata osservata (il caso normale: e' la cella in cui si
    sta entrando) tutti i candidati sono "mai osservati" e viene scelto il centro, cioe'
    esattamente il comportamento precedente. La differenza emerge solo quando abbiamo
    davvero informazione che dice che il centro non va bene.

    exclude (2026-10-07): lista di punti gia' provati e risultati irraggiungibili, da non
    riproporre. Serve alla riscelta dentro il ciclo di navigazione (vedi
    attempt_enter_cell_from_position): il punto scelto all'inizio del tentativo viene
    deciso da lontano, quando la cella e' ancora tutta ignota, e puo' rivelarsi a ridosso
    di un muro quando ci si arriva. Senza escluderlo, la riscelta ripescherebbe lo stesso.

    global_map: spotGrid.GlobalGrid accumulata. Viene aggiornata con la scansione
    corrente appena prima di questa chiamata, quindi e' la fonte piu' completa
    disponibile -- piu' della sola griglia locale, che copre ~3.8 m e in genere non
    arriva al centro della cella bersaglio. Se None, comportamento identico a prima.

    Returns:
        (target_x, target_y, valid_samples, rejected_samples)
        Le due liste alimentano la visualizzazione (gialli = ammessi, rossi = scartati).
    """
    # Stesso margine del PRM e del fronte sicuro (2026-10-06): un punto obiettivo piu' vicino
    # agli ostacoli di quanto il fronte accetti non verrebbe mai raggiunto. Era 0.05.
    if safety_margin is None:
        safety_margin = spotGrid.PRM_EDGE_SAFETY_MARGIN_M
    cell_center = env.get_world_position_from_cell(cell_row, cell_col)
    if cell_center is None:
        return None, None, [], []

    cell_center_x, cell_center_y = cell_center
    sampled = list(global_sampler.get_point_in_cell(cell_row, cell_col))

    if global_map is None:
        # Nessuna informazione di occupazione: comportamento storico invariato.
        return cell_center_x, cell_center_y, sampled, []

    # Il centro geometrico resta il candidato preferito, ma ora deve passare lo stesso
    # vaglio di tutti gli altri.
    candidates = [(cell_center_x, cell_center_y)] + sampled

    # Punti gia' provati e falliti: fuori dai giochi (vedi `exclude`).
    if exclude:
        def _too_close(p):
            return any(np.hypot(p[0] - ex, p[1] - ey) < 0.30 for ex, ey in exclude)
        candidates = [p for p in candidates if not _too_close(p)]
        if not candidates:
            print(f"[TARGET] Cella ({cell_row},{cell_col}): tutti i punti candidati sono gia' "
                  f"stati provati senza riuscirci.")
            return None, None, [], []

    free, unknown, blocked = [], [], []
    for (px, py) in candidates:
        if global_map.is_occupied(px, py, safety_margin=safety_margin):
            blocked.append((px, py))
        elif global_map.is_known(px, py):
            free.append((px, py))
        else:
            unknown.append((px, py))

    pool = free if free else unknown
    if not pool:
        print(f"[TARGET] Cella ({cell_row},{cell_col}): tutti i {len(candidates)} punti "
              f"candidati risultano occupati. Nessun obiettivo valido in questa cella.")
        return None, None, [], blocked

    def _d2_from_center(p):
        return (p[0] - cell_center_x) ** 2 + (p[1] - cell_center_y) ** 2

    best_x, best_y = min(pool, key=_d2_from_center)
    dist_from_center = float(np.sqrt(_d2_from_center((best_x, best_y))))

    if dist_from_center < 1e-6:
        print(f"[TARGET] Cella ({cell_row},{cell_col}): uso il centro geometrico "
              f"({best_x:.2f}, {best_y:.2f})  [liberi={len(free)} ignoti={len(unknown)} "
              f"occupati={len(blocked)}]")
    else:
        print(f"[TARGET] Cella ({cell_row},{cell_col}): centro geometrico non utilizzabile, "
              f"ripiego sul punto libero piu' vicino al centro ({best_x:.2f}, {best_y:.2f}), "
              f"a {dist_from_center:.2f} m da esso  [liberi={len(free)} ignoti={len(unknown)} "
              f"occupati={len(blocked)}]")

    return best_x, best_y, free + unknown, blocked


def goal_still_reachable(snap, goal_x, goal_y):
    """
    2026-10-07. Il punto obiettivo e' ancora raggiungibile sui dati FRESCHI?

    Il fronte sicuro pretende che la linea centrale del percorso stia ad almeno
    FRONTIER_CLEARANCE_M da un ostacolo. Un punto obiettivo con obstacle_distance sotto
    quella soglia non potra' MAI essere raggiunto, per quante scansioni si facciano: il
    robot aspetta, aspetta, e dichiara un blocco.

    Successo misurato sulla missione del 2026-10-07 (iterazione 1, cella (0,1)): il punto
    obiettivo aveva obstacle_distance 0.27 contro i 0.30 richiesti -- 3 cm -- ed era stato
    scelto da 3 m di distanza quando l'area era ancora tutta ignota. Il robot ha fatto tre
    scansioni ferme e si e' bloccato. Entro 80 cm da quel punto c'erano 808 celle valide,
    la piu' vicina a 2 cm dal robot.

    Returns:
        True se il punto e' raggiungibile o se non lo si puo' giudicare (fuori dalla
        finestra scansionata: non sappiamo, e non si tocca niente sulla base di nulla).
    """
    if snap is None:
        return True
    rows, cols, inside = arcVerification._cells(snap, np.array([float(goal_x)]), np.array([float(goal_y)]))
    if not bool(inside[0]):
        return True
    return float(snap.obstacle_dist[rows[0], cols[0]]) >= spotGrid.FRONTIER_CLEARANCE_M


def visualize_grid_with_candidates(pts, terrain_real, obstacle_mask, robot_x, robot_y,
                                   candidates, chosen_point, iteration, env, save_path,
                                   prm_graph=None, chosen_path=None,
                                   cells_obstacle_dist=None, intensity_values=None,
                                   valid_values=None, grad_values=None, rough_values=None,
                                   include_diagnostics=True):
    """
    Non-blocking async wrapper. Snapshots current data and offloads heavy Matplotlib
    rendering to a background thread executor so robot navigation is not delayed.

    Con SAVE_FIGURES_DURING_MISSION = False (default dal 2026-10-06) non disegna: salva gli
    stessi dati in <nome>_vis.npz, da cui mission_viewer.py rigenera le figure dopo.
    """
    if save_path is None:
        return
    if not SAVE_FIGURES_DURING_MISSION:
        _save_visualization_bundle(save_path, pts, terrain_real, obstacle_mask, robot_x, robot_y,
                                   candidates, chosen_point, iteration, env, prm_graph, chosen_path,
                                   cells_obstacle_dist, valid_values, grad_values, rough_values)
        return

    # 1. Fast snapshot of NumPy arrays to prevent thread race conditions
    pts_snap = pts.copy() if pts is not None else None
    terrain_snap = terrain_real.copy() if terrain_real is not None else None
    obs_snap = obstacle_mask.copy() if obstacle_mask is not None else None
    dist_snap = cells_obstacle_dist.copy() if cells_obstacle_dist is not None else None
    valid_snap = valid_values.copy() if valid_values is not None else None
    grad_snap = grad_values.copy() if grad_values is not None else None
    rough_snap = rough_values.copy() if rough_values is not None else None

    # 2. Extract graph snapshots safely
    prm_nodes = dict(prm_graph.nodes) if (prm_graph and hasattr(prm_graph, 'nodes')) else {}
    prm_edges = dict(prm_graph.edges) if (prm_graph and hasattr(prm_graph, 'edges')) else {}
    # Longitudinale/laterale per arco (vedi PRM.edge_slope_info) -- solo per colorare il
    # grafo nella visualizzazione, nessun ricalcolo: riusa quanto già calcolato dal veto.
    edge_slope_snap = dict(prm_graph.edge_slope_info) if (prm_graph and hasattr(prm_graph, 'edge_slope_info')) else {}

    # 3. Snapshot environment structures safely
    env_info = None
    if env is not None:
        # Pre-compute vectorized global accumulation on main thread (takes < 1ms)
        ACCUM_RES = 0.05
        if not hasattr(env, '_accumulated_pts'):
            env._accumulated_pts = {}

        # Compute colors for accumulation
        fused_colors_temp = np.zeros((len(obs_snap), 3), dtype=np.float32) if obs_snap is not None else np.zeros((0, 3))
        if terrain_snap is not None:
            z_terrain = terrain_snap.ravel()
            z_min, z_max = z_terrain.min(), z_terrain.max()
            z_norm = (z_terrain - z_min) / (z_max - z_min) if (z_max - z_min) > 0.001 else np.zeros_like(z_terrain)
            cmap_walkable = matplotlib.colormaps.get_cmap('YlGn')
            fused_colors_temp[:] = cmap_walkable(z_norm)[:, :3]
        else:
            fused_colors_temp[:] = [0.9, 0.9, 0.9]

        if obs_snap is not None:
            fused_colors_temp[obs_snap.ravel() == -1] = [1.0, 0.0, 0.0]

        # Vectorized key computation
        keys = (np.round(pts_snap[:, :2] / ACCUM_RES)).astype(np.int32)
        colors_uint8 = (fused_colors_temp * 255).astype(np.uint8)
        for k, col in zip(keys, colors_uint8):
            env._accumulated_pts[tuple(k)] = col

        # Extract accumulated points array snapshot
        if env._accumulated_pts:
            accum_keys = np.array(list(env._accumulated_pts.keys()), dtype=np.float32)
            accum_wx = accum_keys[:, 0] * ACCUM_RES
            accum_wy = accum_keys[:, 1] * ACCUM_RES
            accum_colors = np.array(list(env._accumulated_pts.values()), dtype=np.float32) / 255.0
        else:
            accum_wx = np.array([robot_x], dtype=np.float32)
            accum_wy = np.array([robot_y], dtype=np.float32)
            accum_colors = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)

        traveled_arcs = list(getattr(env, '_traveled_arcs', []))

        # Snapshot grid cell parameters
        grid_cells = []
        cos_yaw, sin_yaw = np.cos(env.origin_yaw), np.sin(env.origin_yaw)
        half_size = env.cell_size / 2.0
        base_corners = np.array([[-half_size, -half_size], [half_size, -half_size],
                                 [half_size, half_size], [-half_size, half_size]])
        rot_matrix = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])

        for row in range(env.rows):
            for col in range(env.cols):
                world_pos = env.get_world_position_from_cell(row, col)
                if world_pos is None:
                    continue
                cell_x, cell_y = world_pos
                world_corners = np.dot(base_corners, rot_matrix.T) + [cell_x, cell_y]
                status_res = env.get_cell_status(row, col)
                cell_status = status_res[0] if isinstance(status_res, (tuple, list)) else status_res
                grid_cells.append((cell_x, cell_y, row, col, world_corners, cell_status))

        env_info = {
            'cell_size': env.cell_size,
            'accum_wx': accum_wx,
            'accum_wy': accum_wy,
            'accum_colors': accum_colors,
            'traveled_arcs': traveled_arcs,
            'grid_cells': grid_cells
        }

    # 4. Dispatch rendering asynchronously to worker queue
    _VIS_EXECUTOR.submit(
        _async_render_worker,
        pts_snap, terrain_snap, obs_snap, robot_x, robot_y, candidates,
        chosen_point, iteration, env_info, save_path, prm_nodes, prm_edges,
        chosen_path, dist_snap, valid_snap, grad_snap, rough_snap, include_diagnostics,
        edge_slope_snap
    )


def _save_visualization_bundle(save_path, pts, terrain_real, obstacle_mask, robot_x, robot_y,
                               candidates, chosen_point, iteration, env, prm_graph, chosen_path,
                               cells_obstacle_dist, valid_values, grad_values, rough_values):
    """
    Al posto di una figura: tutti i dati che la figura avrebbe mostrato, compreso il grafo
    PRM in quel momento. Viene chiamata quando il grafo cambia (inizio tentativo,
    ripianificazione, blocco), quindi il grafo si salva qui e non a ogni scansione.
    """
    t0 = time.perf_counter()
    try:
        def _arr(a, dtype=np.float64):
            return np.empty(0, dtype=dtype) if a is None else np.asarray(a, dtype=dtype)

        node_ids = np.array(list(prm_graph.nodes.keys()), dtype=np.int64) if prm_graph else np.empty(0, np.int64)
        node_xy = (np.array([prm_graph.nodes[i] for i in node_ids], dtype=np.float64)
                   if prm_graph and len(node_ids) else np.empty((0, 2)))
        pairs, weights = [], []
        if prm_graph:
            for a, nbrs in prm_graph.edges.items():
                for b, w in nbrs:
                    if a < b:
                        pairs.append((a, b)); weights.append(w)
        slope_keys = list(prm_graph.edge_slope_info.keys()) if prm_graph else []
        slope_vals = [prm_graph.edge_slope_info[k] for k in slope_keys] if prm_graph else []
        stop_ids = np.array(list(getattr(prm_graph, 'stop_nodes', {}).keys()), dtype=np.int64)
        traversed = np.array(sorted(getattr(prm_graph, 'traversed_edges', set())), dtype=np.int64)

        cells = []
        if env is not None:
            for row in range(env.rows):
                for col in range(env.cols):
                    wp = env.get_world_position_from_cell(row, col)
                    if wp is None:
                        continue
                    st = env.get_cell_status(row, col)
                    st = st[0] if isinstance(st, (tuple, list)) else st
                    cells.append((row, col, wp[0], wp[1], str(st)))

        def _pts(lst):
            try:
                a = np.asarray([(float(p[0]), float(p[1])) for p in (lst or [])], dtype=np.float64)
                return a.reshape(-1, 2)
            except Exception:
                return np.empty((0, 2))

        out = os.path.splitext(save_path)[0] + "_vis.npz"
        _save_npz_async(
            out,
            pts=_arr(pts), terrain=_arr(terrain_real), obstacle_mask=_arr(obstacle_mask),
            obstacle_distance=_arr(cells_obstacle_dist), terrain_valid=_arr(valid_values),
            gradient=_arr(grad_values), roughness=_arr(rough_values),
            robot_xy=np.array([robot_x, robot_y], dtype=np.float64), mission_z0=_mission_z0(),
            chosen_point=np.array([np.nan if v is None else v for v in (chosen_point or (None, None))],
                                  dtype=np.float64),
            chosen_path=_pts(chosen_path), iteration=iteration,
            candidates_valid=_pts((candidates or {}).get('valid')),
            candidates_rejected=_pts((candidates or {}).get('rejected')),
            prm_node_ids=node_ids, prm_node_xy=node_xy,
            prm_edges=np.array(pairs, dtype=np.int64).reshape(-1, 2),
            prm_edge_weights=np.array(weights, dtype=np.float64),
            prm_slope_keys=np.array(slope_keys, dtype=np.int64).reshape(-1, 2),
            prm_slope_values=np.array(slope_vals, dtype=np.float64).reshape(-1, 2),
            prm_stop_nodes=stop_ids, prm_traversed_edges=traversed.reshape(-1, 2),
            env_cells=np.array([c[:4] for c in cells], dtype=np.float64).reshape(-1, 4),
            env_cell_status=np.array([c[4] for c in cells]),
            env_cell_size=np.nan if env is None else float(env.cell_size),
            env_origin_yaw=np.nan if env is None else float(env.origin_yaw),
            traveled_arcs=np.array(getattr(env, '_traveled_arcs', []), dtype=np.float64).reshape(-1, 4),
        )
    except Exception as e:
        print(f"[DATA SAVE] Impossibile preparare i dati della figura: {e}")
    _timing_add('dati figure', time.perf_counter() - t0)


def _async_render_worker(pts, terrain_real, obstacle_mask, robot_x, robot_y,
                         candidates, chosen_point, iteration, env_info, save_path,
                         prm_nodes, prm_edges, chosen_path, cells_obstacle_dist,
                         valid_values, grad_values, rough_values, include_diagnostics,
                         edge_slope_info=None):
    """Background thread worker handling figure creation and file I/O."""
    edge_slope_info = edge_slope_info or {}
    with _VIS_LOCK:
        try:
            x = pts[:, 0]
            y = pts[:, 1]

            ZOOM_RADIUS = 3.0
            local_xmin_zoom, local_xmax_zoom = robot_x - ZOOM_RADIUS, robot_x + ZOOM_RADIUS
            local_ymin_zoom, local_ymax_zoom = robot_y - ZOOM_RADIUS, robot_y + ZOOM_RADIUS
            local_x_min, local_x_max = x.min(), x.max()
            local_y_min, local_y_max = y.min(), y.max()

            def _apply_common_axis_settings(target_ax, title_text):
                target_ax.set_xlim(local_xmin_zoom, local_xmax_zoom)
                target_ax.set_ylim(local_ymin_zoom, local_ymax_zoom)
                target_ax.set_aspect('equal', adjustable='box')
                target_ax.set_xlabel('X [m] (VISION)', fontsize=11, fontweight='bold')
                target_ax.set_ylabel('Y [m] (VISION)', fontsize=11, fontweight='bold')
                target_ax.set_title(title_text, fontsize=12, fontweight='bold')
                target_ax.grid(True, alpha=0.3)

            def _draw_prm_and_robot(target_ax):
                # High-speed LineCollection rendering for PRM graph
                if prm_nodes and prm_edges:
                    edge_segments = []
                    edge_colors = []
                    for node_id, edges in prm_edges.items():
                        if node_id in prm_nodes:
                            nx1, ny1 = prm_nodes[node_id]
                            if local_xmin_zoom <= nx1 <= local_xmax_zoom and local_ymin_zoom <= ny1 <= local_ymax_zoom:
                                for neighbor_id, _ in edges:
                                    if neighbor_id in prm_nodes:
                                        nx2, ny2 = prm_nodes[neighbor_id]
                                        edge_segments.append([(nx1, ny1), (nx2, ny2)])
                                        edge_colors.append(_slope_edge_color(node_id, neighbor_id, edge_slope_info))

                    if edge_segments:
                        lc = LineCollection(edge_segments, colors=edge_colors, linewidths=0.7, alpha=0.5, zorder=2)
                        target_ax.add_collection(lc)
                        target_ax.text(0.01, 0.01,
                                       "Archi PRM: verde = pendenza in avanti, rosso = traverso (laterale),\n"
                                       "piu' marcato = piu' vicino alla soglia di veto, grigio = nessun dato",
                                       transform=target_ax.transAxes, fontsize=8, va='bottom', ha='left',
                                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.75),
                                       zorder=10, clip_on=True)

                    # Plot visible PRM nodes
                    visible_nodes = np.array([pos for pos in prm_nodes.values()
                                              if local_xmin_zoom <= pos[0] <= local_xmax_zoom and
                                              local_ymin_zoom <= pos[1] <= local_ymax_zoom])
                    if visible_nodes.size > 0:
                        target_ax.plot(visible_nodes[:, 0], visible_nodes[:, 1], 'k.', markersize=3, alpha=0.5, zorder=3)

                # Path
                if chosen_path and len(chosen_path) > 1:
                    path_x = [p[0] for p in chosen_path if p is not None]
                    path_y = [p[1] for p in chosen_path if p is not None]
                    target_ax.plot(path_x, path_y, color='magenta', linewidth=3.0, linestyle='-', zorder=4)
                    target_ax.plot(path_x, path_y, 'mo', markersize=6, markeredgecolor='white', zorder=5)

                # Target
                if chosen_point is not None:
                    target_ax.plot(chosen_point[0], chosen_point[1], 'g*', markersize=18, markeredgewidth=1.5, zorder=6)
                    target_ax.plot([robot_x, chosen_point[0]], [robot_y, chosen_point[1]], 'g--', linewidth=1.8, alpha=0.6, zorder=3)

                # Robot
                target_ax.plot(robot_x, robot_y, 'bo', markersize=12, zorder=7)
                for r in [1.0, 2.0]:
                    circle = patches.Circle((robot_x, robot_y), r, fill=False, linestyle=':', linewidth=1, edgecolor='blue', alpha=0.3, zorder=2)
                    target_ax.add_patch(circle)

            base_save_path, ext = os.path.splitext(save_path)

            # Colors matrix
            fused_colors = np.zeros((len(obstacle_mask), 3), dtype=np.float32)
            if terrain_real is not None:
                z_terrain = terrain_real.ravel()
                z_min, z_max = z_terrain.min(), z_terrain.max()
                z_norm = (z_terrain - z_min) / (z_max - z_min) if (z_max - z_min) > 0.001 else np.zeros_like(z_terrain)
                cmap_walkable = matplotlib.colormaps.get_cmap('YlGn')
                fused_colors[:] = cmap_walkable(z_norm)[:, :3]
            else:
                fused_colors[:] = [0.9, 0.9, 0.9]

            fused_colors[obstacle_mask.ravel() == -1] = [1.0, 0.0, 0.0]

            # ------------------------------------------------------------------ #
            # FIGURE 1: MAIN LOCAL VIEW
            # ------------------------------------------------------------------ #
            fig, ax = plt.subplots(figsize=(8, 8))
            try:
                ax.scatter(x, y, c=fused_colors, s=8, alpha=0.7, zorder=1, label='Terreno / Ostacoli')

                if env_info:
                    margin = env_info['cell_size']
                    for cell_x, cell_y, row, col, world_corners, cell_status in env_info['grid_cells']:
                        # Filter against the ZOOM window (what's actually visible), not the full
                        # local-scan extent -- otherwise cells far outside the zoomed view still
                        # get drawn/labeled off-canvas, and since Text defaults to clip_on=False,
                        # bbox_inches='tight' expands the saved figure to include them, crushing
                        # the real (visible) plot into a tiny corner.
                        if not (local_xmin_zoom - margin <= cell_x <= local_xmax_zoom + margin and
                                local_ymin_zoom - margin <= cell_y <= local_ymax_zoom + margin):
                            continue

                        if cell_status == 1:
                            rect = patches.Polygon(world_corners, linewidth=1.5, edgecolor='darkgreen', facecolor='lightgreen', alpha=0.3, zorder=2, clip_on=True)
                        elif cell_status == -1:
                            rect = patches.Polygon(world_corners, linewidth=1.5, edgecolor='darkred', facecolor='lightcoral', alpha=0.4, zorder=2, clip_on=True)
                        else:
                            rect = patches.Polygon(world_corners, linewidth=1.0, edgecolor='gray', facecolor='none', alpha=0.5, linestyle='--', zorder=2, clip_on=True)

                        ax.add_patch(rect)
                        ax.text(cell_x, cell_y, f'{row},{col}', ha='center', va='center', fontsize=7, color='black', weight='bold', zorder=3, clip_on=True)

                if candidates:
                    if 'rejected' in candidates:
                        for point in candidates['rejected']:
                            ax.plot(point[0], point[1], 'rx', markersize=8, markeredgewidth=2, zorder=5)
                    if 'valid' in candidates:
                        for point in candidates['valid']:
                            ax.plot(point[0], point[1], 'yo', markersize=8, markerfacecolor='yellow', markeredgewidth=1.5, markeredgecolor='orange', zorder=5)

                _draw_prm_and_robot(ax)
                _apply_common_axis_settings(ax, f'Iterazione {iteration}: Path Visualization & Local Scan')
                ax.legend(loc='upper right', fontsize=8)
                plt.tight_layout()
                fig.savefig(save_path, dpi=120, bbox_inches='tight')
            finally:
                plt.close(fig)

            # ------------------------------------------------------------------ #
            # DIAGNOSTICS (Off by default during navigation)
            # ------------------------------------------------------------------ #
            if include_diagnostics:
                # Fixed, physically-meaningful color ranges instead of per-frame min/max.
                # Per-frame auto-scaling lets a single tall/rough/steep tree pixel stretch the
                # whole colorbar, flattening all the ground-level detail into one color band.
                # Values beyond vmax simply clip to the top color (still reads as "extreme")
                # instead of dragging the scale.
                robot_z_ref = float(np.nanmedian(terrain_real)) if terrain_real is not None and terrain_real.size > 0 else 0.0
                terrain_relative = (terrain_real - robot_z_ref) if terrain_real is not None else None

                diagnostic_layers = {
                    'terrain': (terrain_relative, 'YlGn', 'Quota relativa al suolo (m)', 'Layer: TERRAIN', -0.3, 0.3),
                    'obstacle_dist': (cells_obstacle_dist, 'plasma', 'Distanza (m)', 'Layer: OBSTACLE DISTANCE', 0.0, 2.0),
                    'valid': (valid_values, 'binary', '1=Valido, 0=Cieco', 'Layer: VALID MAP', 0.0, 1.0),
                    'gradient': (grad_values, 'YlOrRd', 'Gradiente / Pendenza', 'Diagnostica: PENDENZA', 0.0, 1.2),
                    'roughness': (rough_values, 'coolwarm', 'Indice di Rugosità', 'Diagnostica: RUGOSITÀ', 0.0, 0.3)
                }

                for layer_key, (data_matrix, cmap, label_cb, title_suffix, vmin, vmax) in diagnostic_layers.items():
                    if data_matrix is not None and data_matrix.size > 0:
                        fig_diag, ax_diag = plt.subplots(figsize=(8, 8))
                        try:
                            z_data = data_matrix.flatten() if data_matrix.shape != x.shape else data_matrix
                            sc = ax_diag.scatter(x, y, c=z_data, cmap=cmap, s=6, alpha=0.6, zorder=1,
                                                 vmin=vmin, vmax=vmax)
                            cbar_extend = 'neither' if layer_key == 'valid' else 'both'
                            cbar = fig_diag.colorbar(sc, ax=ax_diag, label=label_cb, shrink=0.8, extend=cbar_extend)
                            _draw_prm_and_robot(ax_diag)
                            _apply_common_axis_settings(ax_diag, f'Iterazione {iteration} - {title_suffix}')
                            plt.tight_layout()
                            diag_save_path = f"{base_save_path}_{layer_key}{ext}"
                            fig_diag.savefig(diag_save_path, dpi=100, bbox_inches='tight')
                        finally:
                            plt.close(fig_diag)

            # ------------------------------------------------------------------ #
            # FIGURE 2: GLOBAL MAP
            # ------------------------------------------------------------------ #
            if env_info:
                fig2, ax2 = plt.subplots(figsize=(12, 10))
                try:
                    ax2.scatter(env_info['accum_wx'], env_info['accum_wy'], c=env_info['accum_colors'], s=2, alpha=0.6, label='Accumulated Local Scan')

                    # Vectorized global PRM edges
                    if prm_nodes and prm_edges:
                        global_segments = []
                        global_colors = []
                        for node_id, edges in prm_edges.items():
                            if node_id in prm_nodes:
                                nx1, ny1 = prm_nodes[node_id]
                                for neighbor_id, _ in edges:
                                    if neighbor_id in prm_nodes:
                                        nx2, ny2 = prm_nodes[neighbor_id]
                                        global_segments.append([(nx1, ny1), (nx2, ny2)])
                                        global_colors.append(_slope_edge_color(node_id, neighbor_id, edge_slope_info))
                        if global_segments:
                            lc_glob = LineCollection(global_segments, colors=global_colors, linewidths=0.6, alpha=0.5, zorder=2)
                            ax2.add_collection(lc_glob)
                            ax2.text(0.01, 0.01,
                                     "Archi PRM: verde = pendenza in avanti, rosso = traverso (laterale),\n"
                                     "piu' marcato = piu' vicino alla soglia di veto, grigio = nessun dato",
                                     transform=ax2.transAxes, fontsize=8, va='bottom', ha='left',
                                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.75),
                                     zorder=10, clip_on=True)

                        node_coords = np.array(list(prm_nodes.values()))
                        if node_coords.size > 0:
                            ax2.plot(node_coords[:, 0], node_coords[:, 1], 'k.', markersize=4, alpha=0.5, zorder=3)

                    if chosen_path and len(chosen_path) > 1:
                        path_x = [p[0] for p in chosen_path if p is not None]
                        path_y = [p[1] for p in chosen_path if p is not None]
                        ax2.plot(path_x, path_y, color='magenta', linewidth=3.5, linestyle='-', zorder=6, label='Chosen Path')

                    if env_info['traveled_arcs']:
                        segments = [[(x1, y1), (x2, y2)] for (x1, y1, x2, y2) in env_info['traveled_arcs']]
                        traveled_lc = LineCollection(segments, colors='blue', linewidths=2.0, alpha=0.6, zorder=5)
                        ax2.add_collection(traveled_lc)
                        ax2.plot([], [], color='blue', linewidth=2.0, alpha=0.6, label='Traveled Path')

                    rect_local = patches.Rectangle((local_x_min, local_y_min), local_x_max - local_x_min, local_y_max - local_y_min,
                                                   linewidth=2, edgecolor='cyan', facecolor='none', alpha=0.8, zorder=6, label='Current scan')
                    ax2.add_patch(rect_local)

                    for cell_x, cell_y, row, col, world_corners, cell_status in env_info['grid_cells']:
                        if cell_status == 1:
                            rect = patches.Polygon(world_corners, linewidth=1.5, edgecolor='darkgreen', facecolor='lightgreen', alpha=0.3, zorder=2)
                        elif cell_status == -1:
                            rect = patches.Polygon(world_corners, linewidth=1.5, edgecolor='darkred', facecolor='lightcoral', alpha=0.4, zorder=2)
                        else:
                            rect = patches.Polygon(world_corners, linewidth=1.0, edgecolor='gray', facecolor='none', alpha=0.5, linestyle='--', zorder=2)

                        ax2.add_patch(rect)
                        ax2.text(cell_x, cell_y, f'{row},{col}', ha='center', va='center', fontsize=7, color='black', weight='bold', zorder=3)

                    if chosen_point is not None:
                        ax2.plot(chosen_point[0], chosen_point[1], 'g*', markersize=20, markeredgewidth=2, label='Target', zorder=6)

                    ax2.set_xlabel('X [m] (VISION)', fontsize=12, fontweight='bold')
                    ax2.set_ylabel('Y [m] (VISION)', fontsize=12, fontweight='bold')
                    ax2.set_title(f'Iteration {iteration}: Global Map View', fontsize=13, fontweight='bold')
                    ax2.set_aspect('equal', adjustable='box')
                    ax2.grid(True, alpha=0.3)
                    ax2.legend(loc='upper right', fontsize=9)
                    plt.tight_layout()

                    global_save_path = f"{base_save_path}_global{ext}"
                    fig2.savefig(global_save_path, dpi=120, bbox_inches='tight')
                finally:
                    plt.close(fig2)

        except Exception as err:
            print(f"[VISUALIZATION ERROR] Async rendering failed: {err}")


def decide_next_move(robot_xy, path_waypoints, frontier, no_progress_count):
    """
    Cosa fare adesso, dato il fronte sicuro. Funzione pura: niente robot, niente I/O, quindi
    provabile fuori dal campo (vedi test_step3_frontier.py).

    path_waypoints: [(node_id, x, y), ...] -- i nodi ancora da raggiungere, il primo e' il
        prossimo. La posizione del robot NON e' inclusa.
    frontier: risultato di arcVerification.compute_safe_frontier sul percorso
        [robot] + waypoint.
    no_progress_count: scansioni consecutive in cui non si e' potuto avanzare.

    Restituisce un dict con 'kind':
      'reached_node'  il prossimo nodo e' gia' sotto il robot (< 5 cm): consumarlo e basta
      'move'          muoversi verso 'target'; 'full' = True se il target e' il nodo stesso
      'arrived'       non si puo' avanzare, ma l'obiettivo finale e' entro tolleranza
      'wait'          non si puo' avanzare: riscansionare da fermi prima di concludere
      'blocked'       non si puo' avanzare da FRONTIER_BLOCK_CONFIRM_SCANS scansioni
                      consecutive: blocco confermato da vicino. 'segment' = indice del
                      tratto del percorso in cui si e' fermato il fronte (0 = dal robot al
                      primo nodo).
    """
    rx, ry = robot_xy
    _, nx, ny = path_waypoints[0]
    seg_len = float(np.hypot(nx - rx, ny - ry))
    if seg_len < 0.05:
        return {'kind': 'reached_node'}

    adv = arcVerification.allowed_advance(frontier)
    if adv >= seg_len - 0.02:
        return {'kind': 'move', 'target': (nx, ny), 'full': True, 'dist': seg_len}
    if adv >= MIN_PARTIAL_STEP_M:
        f = adv / seg_len
        return {'kind': 'move', 'target': (rx + f * (nx - rx), ry + f * (ny - ry)),
                'full': False, 'dist': adv}

    if frontier['path_len'] <= GOAL_REACHED_TOLERANCE_M:
        return {'kind': 'arrived'}
    if no_progress_count + 1 < FRONTIER_BLOCK_CONFIRM_SCANS:
        return {'kind': 'wait'}

    polyline = [(rx, ry)] + [(x, y) for _, x, y in path_waypoints]
    _, seg_idx = arcVerification.point_along(polyline, frontier['dist'])
    return {'kind': 'blocked', 'segment': int(seg_idx)}


def rotation_plan(robot_yaw, robot_xy, target_xy, od_robot, snap=None):
    """
    Si puo' ruotare sul posto per guardare il punto da raggiungere?

    RISCRITTO il 2026-10-07. Prima bastava confrontare obstacle_distance sotto il robot con
    ROTATION_CLEARANCE_M = 0.65, cioe' chiedere che fosse libero TUTTO il cerchio di raggio
    0.60 attorno al robot: il caso peggiore su tutte le direzioni e tutti gli angoli.
    Ruotando di 39 gradi il corpo spazza invece un settore, e gli ostacoli stanno dove
    stanno. Misurato sulla missione del 2026-10-07 (iterazione 2, cella (1,0)): il
    confronto scalare ha rifiutato 9 rotazioni su 10, e di quelle 9 il controllo sul
    settore dice che 8 erano sicure. Tre di quei rifiuti hanno prodotto un blocco su un
    percorso che col muso in avanti era libero per 1.39 m.

    Ordine dei controlli, dal piu' economico:
      1. rotazione sotto ROTATION_CHECK_MIN_DYAW_DEG -- trascurabile, si ruota
      2. obstacle_distance sotto il robot >= ROTATION_CLEARANCE_M -- cerchio circoscritto
         libero, si ruota senza altri calcoli (prefiltro)
      3. spotGrid.rotation_is_clear() sulle celle davvero spazzate (0.26 ms)

    Senza `snap` (nessuna fotografia della griglia) si ricade sul solo confronto scalare,
    cioe' il comportamento precedente.

    Funzione pura, provabile fuori dal campo. Restituisce dict con:
      turn (bool), dyaw_deg, motion_dir (rad), margin (margine del settore spazzato, in
      metri, o None se non calcolato).
    """
    rx, ry = robot_xy
    tx, ty = target_xy
    motion_dir = float(np.arctan2(ty - ry, tx - rx))
    dyaw = float(np.degrees(np.arctan2(np.sin(motion_dir - robot_yaw), np.cos(motion_dir - robot_yaw))))

    if abs(dyaw) < spotGrid.ROTATION_CHECK_MIN_DYAW_DEG:
        return {'turn': True, 'dyaw_deg': dyaw, 'motion_dir': motion_dir, 'margin': None,
                'why': 'rotazione trascurabile'}
    if od_robot >= spotGrid.ROTATION_CLEARANCE_M:
        return {'turn': True, 'dyaw_deg': dyaw, 'motion_dir': motion_dir, 'margin': None,
                'why': 'cerchio circoscritto libero'}
    if snap is not None:
        ok, margin = spotGrid.rotation_is_clear(
            snap.obstacle_dist, snap.origin_x, snap.origin_y, snap.cell_size,
            rx, ry, robot_yaw, float(np.radians(dyaw)))
        return {'turn': bool(ok), 'dyaw_deg': dyaw, 'motion_dir': motion_dir,
                'margin': float(margin) if np.isfinite(margin) else None,
                'why': 'settore spazzato'}
    return {'turn': False, 'dyaw_deg': dyaw, 'motion_dir': motion_dir, 'margin': None,
            'why': 'nessun dato di griglia'}


SHORTCUT_MAX_DIST_M = 2.5          # oltre, il tratto non sta comunque nella finestra


def shortcut_index(robot_xy, path_waypoints, snap, global_map=None, prm=None, robot_node_id=None):
    """
    Scorciatoia (2026-10-06): l'indice del waypoint PIU' LONTANO raggiungibile in linea retta
    dal robot con il tratto interamente confermato libero dal fronte sicuro (muso compreso).
    0 = nessuna scorciatoia. Toglie le svolte inutili fra nodi del PRM.
    Il controllo e' il fronte stesso sui dati dal vivo: ostacoli, rugosita', pendenza.

    E anche la MAPPA GLOBALE, con lo stesso margine del PRM: alcune cose stanno solo li' --
    il punto di una caduta, per esempio, che ai sensori puo' sembrare terreno normale (e' per
    questo che il robot ci e' caduto). Gli archi del PRM le evitano gia'; una scorciatoia
    controllata solo dal vivo potrebbe riportarci sopra il robot.

    2026-10-07 -- E GLI ARCHI GIA' SCARTATI. Senza questo controllo la scorciatoia
    riagganciava un waypoint il cui collegamento diretto era appena stato invalidato, e il
    ciclo non usciva piu': il tratto veniva scartato, Dijkstra trovava una strada che passava
    comunque per quel nodo attraverso i nodi intermedi, e la scorciatoia ci ripuntava dritto.
    Osservato nella missione del 2026-10-07: ~40 giri identici sul nodo 225. Servono `prm` e
    `robot_node_id`; senza, il controllo si salta (comportamento precedente).
    """
    rx, ry = robot_xy
    for k in range(len(path_waypoints) - 1, 0, -1):
        wid, wx, wy = path_waypoints[k]
        if prm is not None and robot_node_id is not None:
            key = (min(robot_node_id, wid), max(robot_node_id, wid))
            if key in getattr(prm, 'tracker_blocked_edges', ()):
                continue
        d = float(np.hypot(wx - rx, wy - ry))
        if d > SHORTCUT_MAX_DIST_M:
            continue
        if global_map is not None:
            n = max(prm_graph.OCCUPANCY_SAMPLES_MIN, int(np.ceil(d * prm_graph.OCCUPANCY_SAMPLES_PER_M)) + 1)
            if any(global_map.is_occupied(rx + t * (wx - rx), ry + t * (wy - ry),
                                          safety_margin=spotGrid.PRM_EDGE_SAFETY_MARGIN_M)
                   for t in np.linspace(0.0, 1.0, n)[1:]):
                continue
        f = arcVerification.compute_safe_frontier([(rx, ry), (wx, wy)], snap)
        if f['reason'] == 'fine_percorso':
            return k
    return 0


ROTATION_ROOM_SEARCH_M = (0.15, 0.30, 0.45, 0.60, 0.80, 1.00, 1.20)
ROTATION_ROOM_DIRECTIONS_DEG = (180, 135, -135, 90, -90, 45, -45, 0)   # rispetto al muso: indietro prima


def find_rotation_room(robot_xy, robot_yaw, snap):
    """
    NON PIU' USATA dal ciclo principale (2026-10-07), lasciata per riferimento e per i test.

    Cercava il punto piu' vicino con spazio per ruotare, raggiungibile SENZA girarsi, cioe'
    camminando di traverso. Non serve piu' perche' la marcia di traverso e' stata eliminata:
    di traverso Spot e' largo 1.10 m e il fronte pretende 0.59 m invece di 0.30, quindi
    questa ricerca falliva proprio nelle situazioni per cui era stata scritta (misurato il
    2026-10-07: punti con spazio pieno a 0.03-0.56 m dal robot, nessuno raggiungibile).
    Al suo posto: arretramento dritto lungo l'asse del corpo, vedi retreat_straight().

    Funzione pura. Restituisce (x, y, direzione_relativa_gradi, distanza) oppure None.
    """
    rx, ry = robot_xy
    for dist in ROTATION_ROOM_SEARCH_M:
        for rel in ROTATION_ROOM_DIRECTIONS_DEG:
            d = robot_yaw + np.radians(rel)
            tx, ty = rx + dist * np.cos(d), ry + dist * np.sin(d)
            rows, cols, inside = arcVerification._cells(snap, np.array([tx]), np.array([ty]))
            if not inside[0] or snap.obstacle_dist[rows[0], cols[0]] < spotGrid.ROTATION_CLEARANCE_M:
                continue
            body = spotGrid.body_extents(robot_yaw, d)
            f = arcVerification.compute_safe_frontier([(rx, ry), (tx, ty)], snap, body=body)
            if arcVerification.allowed_advance(f) >= dist - 0.02:
                return float(tx), float(ty), float(rel), float(dist)
    return None


def plan_straight_retreat(robot_xy, robot_yaw, snap, max_dist=ROTATION_RETREAT_M):
    """
    2026-10-07. Quanto puo' arretrare il robot IN LINEA RETTA, all'indietro, senza girarsi.

    Non e' marcia obliqua: muovendosi lungo il proprio asse l'ingombro resta 0.50 m di
    larghezza e il fronte applica la soglia normale di 0.30 m, non i 0.59 della marcia di
    traverso. E' la via d'uscita quando la rotazione verso il prossimo punto non e'
    possibile: ci si stacca da cio' che impedisce di girarsi e al giro dopo si riprova da
    una posizione diversa.

    Misurato sulla missione del 2026-10-07: l'arretramento dritto era disponibile in TUTTE
    e 12 le scansioni dell'iterazione 1, da 1.08 a 1.37 m.

    Funzione pura. Restituisce (x, y, distanza) del punto raggiungibile, oppure None se non
    si puo' arretrare di almeno ROTATION_RETREAT_MIN_M.
    """
    rx, ry = robot_xy
    back_x = rx - max_dist * np.cos(robot_yaw)
    back_y = ry - max_dist * np.sin(robot_yaw)
    body = spotGrid.body_extents(robot_yaw, robot_yaw + np.pi)   # all'indietro: (0.55, 0.25)
    f = arcVerification.compute_safe_frontier([(rx, ry), (back_x, back_y)], snap, body=body)
    adv = arcVerification.allowed_advance(f)
    if adv < ROTATION_RETREAT_MIN_M:
        return None
    adv = min(adv, max_dist)
    return (float(rx - adv * np.cos(robot_yaw)), float(ry - adv * np.sin(robot_yaw)), float(adv))


def slope_usable_mask(is_valid_flat, cells_obstacle_dist, shape):
    """
    2026-10-07 (questione aperta B, chiusa). Maschera delle celle le cui QUOTE possono
    entrare nel calcolo della pendenza di un arco: attendibili (is_valid di correct_terrain)
    E non-ostacolo.

    La faccia verticale di un mobile non e' terreno in pendenza, e' un ostacolo -- ed e' gia'
    gestita come tale due volte: dal veto di occupazione del PRM e dal fronte sicuro, che si
    ferma a FRONTIER_CLEARANCE_M da qualunque cella con obstacle_distance bassa. Contarla
    ANCHE come pendenza la conta la terza volta, e con il peggiore degli effetti: non ferma
    il robot davanti al mobile (ci pensa il fronte), scollega il grafo PRM in tutta la stanza.

    Misurato sulla missione del 2026-10-07 13:58, stanza con mobili: 200 archi distinti
    vetati per pendenza, mediana longitudinale 1.487 contro soglia 0.600, massimo 2.092 --
    su un pavimento piano. La ripianificazione falliva da OGNI nodo (225, 227, 228) pur
    avendone 22-28 collegamenti ciascuno, quindi la ritirata era l'unico esito possibile.
    Rimisurato sulle 12 scansioni della missione 14:0x, 3607 archi campionati nella finestra:

        archi vetati per pendenza   40.3%  ->   0.1%
        archi senza dati pendenza    4.9%  ->  42.5%
        archi AMMESSI nel grafo     59.7%  ->  99.9%
        pendenza long. massima    2.2-3.3  ->  0.03-0.10

    Gli archi che perdono il dato NON diventano "liberi per decreto": diventano "senza dati"
    e un arco senza dati non viene vetato ne' in build_graph ne' in _try_add_edge (entra con
    un costo di ripiego), e chi lo percorre lo verifica metro per metro col fronte sicuro.
    E' la scelta esplicita del 2026-10-07: su quegli archi la sicurezza la da' il fronte, non
    una pendenza misurata su un armadio.

    La maschera si combina QUI e non dentro prm_graph perche' e' qui che si hanno in mano
    entrambi gli array, e perche' in prm_graph `current_valid_2d` serve a una cosa sola -- il
    profilo di pendenza per-arco (compute_arc_slope_profile) -- quindi restringerla non ha
    altri effetti. Chi legge prm_graph.update_local_grid_data trova `valid_2d` documentato
    come "celle attendibili": dal 2026-10-07 significa "attendibili come TERRENO".
    """
    valid = np.asarray(is_valid_flat, dtype=bool).reshape(shape)
    if cells_obstacle_dist is None:
        return valid
    od = np.asarray(cells_obstacle_dist, dtype=np.float64).reshape(shape)
    return valid & ~(od < spotGrid.OBSTACLE_THRESHOLD)


def graphnav_drop_breadcrumb(env, recordingInterface, robot_state_client, prm_node_id,
                             cell_row, cell_col, force=False):
    """
    2026-10-07. Lascia una briciola: un waypoint GraphNav nel punto in cui si trova il robot,
    legato al nodo PRM da cui sta passando.

    Serve per i ritorni. Finora si tornava indietro camminando ALL'INDIETRO sui tratti
    registrati (retreat_along_traveled_path): il robot non vede dove mette le zampe, non si
    rilocalizza, e un errore si accumula. Con una traccia di waypoint il ritorno lo fa
    GraphNav, che cammina in avanti, con la localizzazione e l'anti-ostacolo di Boston
    Dynamics. Decisione del 2026-10-07.

    Densita': un waypoint solo se il robot e' almeno GRAPHNAV_WAYPOINT_MIN_SPACING_M
    dall'ultimo lasciato, piu' sempre uno all'ingresso di una cella (force=True). Un
    waypoint per OGNI nodo PRM attraversato darebbe una mappa troppo densa: i nodi di sosta
    nascono dai passi parziali e nella missione del 2026-10-07 14:5x erano 37, molti a
    20-60 cm uno dall'altro.

    La traccia vive su env (_graphnav_trail) perche' deve sopravvivere ai tentativi di cella.
    Non solleva mai: una briciola mancata e' una briciola, non un motivo per fermare il robot.
    """
    if recordingInterface is None:
        return None
    trail = getattr(env, '_graphnav_trail', None)
    if trail is None:
        trail = env._graphnav_trail = []
    try:
        x, y, _, _ = spotUtils.getPosition(robot_state_client)
    except Exception as e:
        print(f"[BRICIOLA] Posizione non leggibile: {e}")
        return None
    if not force and trail:
        last = trail[-1]
        if float(np.hypot(x - last['x'], y - last['y'])) < GRAPHNAV_WAYPOINT_MIN_SPACING_M:
            return None
    try:
        recordingInterface.create_default_waypoint(cell_row=cell_row, cell_col=cell_col)
        loc = recordingInterface.get_localization_state()
        wp_id = loc.get('waypoint_id') if isinstance(loc, dict) else None
        wp_name = loc.get('waypoint_name') if isinstance(loc, dict) else None
    except Exception as e:
        print(f"[BRICIOLA] Waypoint non creato: {e}")
        return None
    if not wp_id:
        print("[BRICIOLA] Waypoint creato ma non localizzato: non lo metto nella traccia "
              "(un waypoint senza id non si puo' raggiungere).")
        return None
    crumb = {'wp_id': wp_id, 'wp_name': wp_name, 'x': float(x), 'y': float(y),
             'prm_node': prm_node_id, 'cell': (cell_row, cell_col)}
    trail.append(crumb)
    print(f"[BRICIOLA] Waypoint {wp_name} in ({x:.2f}, {y:.2f}), nodo PRM {prm_node_id} "
          f"(traccia: {len(trail)} waypoint).")
    return crumb


def graphnav_step_back(env, recordingInterface, robot_state_client, command_client,
                       min_dist_m=0.40):
    """
    2026-10-07. Torna al waypoint piu' recente che sia almeno min_dist_m DIETRO, con GraphNav.

    Si cerca indietro nella traccia perche' l'ultima briciola puo' essere stata lasciata qui,
    dove il robot e' adesso: tornarci non sposterebbe niente. Se nessuna briciola e' abbastanza
    lontana si usa comunque la piu' vecchia disponibile.

    Restituisce (ok, crumb). ok=True solo se GraphNav dichiara di essere ARRIVATO: in
    _graphnav_navigate_to lo STATUS_LOST e' un fallimento, non si forza la localizzazione
    sulla destinazione (era il difetto di navGraphUtils.navigate_to_waypoint, che su LOST
    dichiarava "arrivato" mentre il robot poteva essere altrove).

    La registrazione va FERMATA prima di navigare e ripresa dopo: GraphNav non naviga mentre
    registra. E' la stessa sequenza usata per gli spostamenti fra celle.
    """
    trail = getattr(env, '_graphnav_trail', None) or []
    if not trail:
        return False, None
    try:
        rx, ry, _, _ = spotUtils.getPosition(robot_state_client)
    except Exception:
        rx, ry = trail[-1]['x'], trail[-1]['y']
    idx = None
    for k in range(len(trail) - 1, -1, -1):
        if float(np.hypot(trail[k]['x'] - rx, trail[k]['y'] - ry)) >= min_dist_m:
            idx = k
            break
    if idx is None:
        idx = 0
    hop = len(trail) - 1 - idx
    crumb = trail[idx]
    recording_stopped = False
    try:
        recordingInterface.stop_recording()
        recording_stopped = True
    except Exception as e:
        print(f"[RITORNO] Non ho potuto fermare la registrazione ({e}): provo comunque.")
    ok = _graphnav_navigate_to(
        recordingInterface, crumb['wp_id'], robot_state_client, command_client,
        f"ritorno al waypoint {crumb['wp_name']} in ({crumb['x']:.2f}, {crumb['y']:.2f})")
    if recording_stopped:
        try:
            recordingInterface.start_recording()
        except Exception as e:
            print(f"[RITORNO] Registrazione non ripresa ({e}): le briciole successive "
                  "potrebbero non essere create.")
    if ok:
        # I tratti percorsi a zampe non descrivono piu' la strada da cui si e' arrivati:
        # GraphNav non li registra, e la ritirata di ripiego non deve usarli.
        env._traveled_arcs = []
        del trail[len(trail) - hop:]        # le briciole oltre questa non servono piu'
    return ok, crumb


def escape_by_backtracking(env, prm_graph, goal_id, recordingInterface, robot_state_client,
                           command_client, global_map, mobility_kwargs, pts, obstacle_mask,
                           max_hops=GRAPHNAV_MAX_BACKTRACK_HOPS, label=""):
    """
    2026-10-07. La via d'uscita quando da qui non si va da nessuna parte.

    Logica decisa il 2026-10-07: si torna indietro di un waypoint con GraphNav, si prova a
    ripianificare verso l'obiettivo da li'; se non si puo', si torna indietro di un altro, e
    cosi' via fino a max_hops. Dopo max_hops la cella si rimanda e si passa alla priorita'
    successiva. Prima si camminava all'indietro a zampe fino a esaurire i tratti registrati,
    senza mai ritentare la pianificazione nel mezzo.

    Se GraphNav non ce la fa (traccia vuota, non localizzato, STUCK, tempo scaduto) si ricade
    sulla ritirata a zampe: e' peggiore, ma e' l'ultima rete e non la togliamo. Nella trappola
    misurata il 2026-10-07 14:5x ruotare era impossibile (0.40 m contro 0.604 necessari), e
    GraphNav per percorrere un arco vuole girare il muso: li' potrebbe tornare STUCK proprio
    dove serve. Si vedra' sul campo.

    Restituisce dict: ok (si puo' continuare la missione), ids (nuovo percorso o None),
    robot_node (nuovo nodo del robot o None), how (cosa e' successo, per il log).
    """
    for hop in range(1, int(max_hops) + 1):
        ok, crumb = graphnav_step_back(env, recordingInterface, robot_state_client, command_client)
        if not ok:
            if crumb is None:
                print(f"[RITORNO] {label}: nessuna briciola GraphNav disponibile.")
            else:
                print(f"[RITORNO] {label}: GraphNav non e' riuscito a tornare a "
                      f"{crumb['wp_name']}.")
            MISSION_STATS['graphnav_backtrack_fail'] += 1
            print("[RITORNO] Ripiego sulla ritirata a zampe sui tratti percorsi.")
            retreat_along_traveled_path(env, robot_state_client, command_client,
                                        mobility_kwargs=mobility_kwargs,
                                        pts=pts, obstacle_mask=obstacle_mask)
            return {'ok': False, 'ids': None, 'robot_node': None, 'how': 'ritirata a zampe'}

        MISSION_STATS['graphnav_backtracks'] += 1
        try:
            ax, ay, _, _ = spotUtils.getPosition(robot_state_client)
        except Exception:
            ax, ay = crumb['x'], crumb['y']
        new_node = prm_graph.add_stop_node(
            ax, ay, global_map, spotGrid.PRM_EDGE_SAFETY_MARGIN_M,
            label=f"sosta (ritorno GraphNav a {crumb['wp_name']})")
        # Niente deroga alla mappa globale attorno a questo nodo: vedi il commento al
        # chiamante di set_robot_node nel ciclo principale.
        prm_graph.set_robot_node(new_node, 0.0)

        ids = prm_graph.find_path_dijkstra(new_node, goal_id)
        if ids is not None and len(ids) > 1:
            print(f"[RITORNO] {label}: tornato a {crumb['wp_name']} ({hop}/{max_hops}) e da qui "
                  f"c'e' un percorso con {len(ids) - 1} tratti. Riprendo.")
            return {'ok': True, 'ids': ids, 'robot_node': new_node, 'how': 'ripianificato'}
        print(f"[RITORNO] {label}: tornato a {crumb['wp_name']} ({hop}/{max_hops}) ma da qui il "
              f"grafo non offre percorsi verso l'obiettivo.")

    print(f"[RITORNO] {label}: {max_hops} ritorni senza trovare un percorso. Rimando la cella.")
    return {'ok': False, 'ids': None, 'robot_node': None, 'how': 'cella rimandata'}


def prune_prm_nodes_in_obstacles(prm, snap, min_clearance_m=PRM_NODE_MIN_CLEARANCE_M):
    """
    2026-10-07. Scarta dal grafo i nodi che cadono DENTRO un ostacolo.

    Gli archi che attraversano celle occupate erano gia' vetati (_occupancy_blocked), ma i
    nodi non venivano mai controllati: nascono dal campionatore globale prima che il robot
    abbia visto qualcosa. Nella missione del 2026-10-07 14:5x il nodo obiettivo della cella
    (0,1) aveva obstacle_distance **-0.21 m**, cioe' era fisicamente dentro un ostacolo, e
    i due ripieghi successivi 0.12 m: sono i tre "obiettivi riscelti" di quella missione,
    spesi per scoprire sul posto una cosa che si sapeva dai dati.

    Soglia min_clearance_m = FRONTIER_CLEARANCE_M (0.30): sotto quella il centro del corpo
    non ci sta e il fronte non ci arrivera' mai, quindi il nodo e' inutile sia come
    destinazione sia come punto di passaggio.

    Si scarta SOLO dove la cella e' stata davvero osservata: ignoto non e' occupato (e' la
    stessa regola per cui il fronte distingue 'fine_dati' da 'ostacolo'). Un nodo fuori
    dalla finestra o mai visto resta dov'e' e verra' giudicato quando lo si vedra'.

    Si rivaluta a ogni ciclo, quindi un nodo che si scopre dentro un muro cade subito; e
    siccome gli archi vengono tolti in modo NON permanente, un nodo giudicato male da una
    lettura rumorosa torna in gioco quando i dati freschi lo smentiscono.
    """
    ids = [nid for nid in prm.nodes if prm.edges.get(nid)]   # gia' isolati: niente da togliere
    if not ids:
        return []
    xy = np.array([prm.nodes[nid] for nid in ids], dtype=np.float64)
    rows, cols, inside = arcVerification._cells(snap, xy[:, 0], xy[:, 1])
    bad = np.zeros(len(ids), dtype=bool)
    if np.any(inside):
        r, c = rows[inside], cols[inside]
        bad[inside] = snap.seen[r, c] & (snap.obstacle_dist[r, c] < min_clearance_m)
    pruned = []
    for k in np.nonzero(bad)[0]:
        nid = ids[int(k)]
        for nb in [n for n, _ in prm.edges.get(nid, [])]:
            mark_edge_blocked_soft(prm, nid, nb)
        pruned.append(nid)
    return pruned


def scan_escape_directions(robot_xy, robot_yaw, snap, step_deg=ESCAPE_FAN_STEP_DEG,
                           min_advance_m=ESCAPE_FAN_MIN_ADVANCE_M, probe_m=3.0):
    """
    2026-10-07. Il giro completo: in quali direzioni il robot puo' DAVVERO andare da qui?

    Per ciascuna direzione del giro (passo step_deg) misura due cose indipendenti:
      - quanto spazio concede il fronte sicuro andando in linea retta in quella direzione;
      - se il robot puo' girarsi tanto da puntarci il muso (spotGrid.rotation_is_clear).
    Una direzione e' UTILIZZABILE solo se entrambe rispondono si'. Il movimento di traverso
    non esiste piu' (vedi il commento in testa al file), quindi una corsia larga che richiede
    una rotazione impossibile NON e' una via d'uscita, per quanto larga sia.

    Perche' serve (misurato sulla missione del 2026-10-07 14:5x, file
    iteration_2_cell_1_0_BLOCKED_vis.npz, robot in (-1.69, 1.87), od sotto di lui 0.402):

        direzione   spazio libero   rotazione necessaria   ruotabile
          -180            1.30 m           -73 gradi          NO
             0            1.09 m          +107 gradi          NO
          -105 (muso)      0.00 m           +2 gradi          si
        altre 21        0.00-0.24 m           --              NO

    Zero direzioni utilizzabili: il robot era incastrato in un angolo, con le due corsie
    aperte dietro rotazioni che in 0.40 m di spazio non si fanno. La ritirata sul tratto
    percorso era l'unica uscita ed era giusta -- ma il codice e' arrivato a quella
    conclusione scartando OTTO archi uno per uno e spendendo una ventina di scansioni,
    mentre questo ventaglio costa 24 valutazioni del fronte. Lo stesso schema si e'
    ripetuto al nodo 281 e, con rotazioni di +/-179 gradi, in cella (0,2).

    Funzione pura. Restituisce la lista delle direzioni esaminate, ciascuna con:
    dir_deg (assoluta), rel_deg (rotazione necessaria), advance, reason, rotatable, usable.
    """
    rx, ry = float(robot_xy[0]), float(robot_xy[1])
    out = []
    for deg in np.arange(-180.0, 180.0, abs(step_deg)):
        a = float(np.radians(deg))
        f = arcVerification.compute_safe_frontier(
            [(rx, ry), (rx + probe_m * np.cos(a), ry + probe_m * np.sin(a))], snap)
        adv = arcVerification.allowed_advance(f)
        rel = float(np.degrees(np.arctan2(np.sin(a - robot_yaw), np.cos(a - robot_yaw))))
        if abs(rel) < spotGrid.ROTATION_CHECK_MIN_DYAW_DEG:
            rotatable = True
        else:
            rotatable = bool(spotGrid.rotation_is_clear(
                snap.obstacle_dist, snap.origin_x, snap.origin_y, snap.cell_size,
                rx, ry, float(robot_yaw), float(np.radians(rel)))[0])
        out.append({'dir_deg': float(deg), 'rel_deg': rel, 'advance': float(adv),
                    'reason': f['reason'], 'rotatable': rotatable,
                    'usable': bool(rotatable and adv >= min_advance_m)})
    return out


def describe_escape_fan(dirs, max_rows=4):
    """Riga di log compatta: quante direzioni utilizzabili e le migliori, per i log."""
    usable = [d for d in dirs if d['usable']]
    best = sorted(dirs, key=lambda d: -d['advance'])[:max_rows]
    txt = "; ".join(f"{d['dir_deg']:+.0f} gradi: {d['advance']:.2f} m "
                    f"({'ruotabile' if d['rotatable'] else 'NON ruotabile'})" for d in best)
    return len(usable), txt


def is_trapped(robot_xy, robot_yaw, snap):
    """
    2026-10-07. True se NESSUNA direzione del giro e' insieme percorribile e raggiungibile
    girandosi: la condizione che l'utente ha posto per tornare indietro ("solo se non puo'
    piu' andare avanti ne' ruotare"), verificata in una scansione invece che in venti.
    Restituisce (trapped, dirs) -- dirs serve per il log, cosi' nei dati si vede perche'.
    """
    dirs = scan_escape_directions(robot_xy, robot_yaw, snap)
    return (not any(d['usable'] for d in dirs)), dirs


def arrival_has_exit(snap, arrival_xy, heading, min_advance_m=ESCAPE_FAN_MIN_ADVANCE_M):
    """
    2026-10-07. Dal punto in cui sto per fermarmi, ci sara' un'uscita?

    Il fronte sicuro guarda la strada DAVANTI, non se dalla posa d'arrivo si potra'
    ripartire: nella missione del 2026-10-07 14:5x il robot e' entrato in un angolo senza
    uscite con un ultimo passo di 23 cm, ricostruito dalla ritirata
    ((0.40, 1.76) -> (-0.71, 1.70) -> (-1.47, 1.79) -> (-1.69, 1.87)).

    Si valuta il ventaglio nel punto d'arrivo usando la fotografia CORRENTE: e' una stima
    (da li' il robot vedra' di piu'), ma e' la stessa informazione su cui si sta decidendo
    di andarci, quindi la decisione e' almeno coerente con se stessa. Per non pagarla su
    ogni movimento la chiama solo il ciclo principale, e solo per i passi corti o verso
    spazi stretti -- vedi ARRIVAL_EXIT_CHECK_MAX_STEP_M / _OD_M.

    Restituisce (ha_uscita, dirs).
    """
    ax, ay = float(arrival_xy[0]), float(arrival_xy[1])
    # Stessa griglia, ma con il robot dichiarato nel punto d'arrivo: e' cio' che cambia il
    # riferimento della regola di uscita del fronte e il centro delle rotazioni provate.
    probe = arcVerification.GridSnapshot(
        snap.obstacle_dist, snap.rough_veto, snap.seen, snap.terrain,
        snap.origin_x, snap.origin_y, snap.cell_size, ax, ay,
        timestamp=snap.time, valid=snap.valid, blocked_latch=snap.blocked_latch)
    dirs = scan_escape_directions((ax, ay), float(heading), probe, min_advance_m=min_advance_m)
    return any(d['usable'] for d in dirs), dirs


def mark_edge_blocked_soft(prm, idx1, idx2):
    """
    2026-10-07. Toglie un arco dal grafo ADESSO, senza condannarlo per sempre.

    prm_graph.mark_edge_invalid() lo mette anche in `tracker_blocked_edges`, che non viene
    MAI ricreato -- ne' da refresh_local_edge_weights ne' da build_graph. Giusto per un
    blocco confermato da tre scansioni ferme; sbagliato per [SUBITO], che scarta dopo UNA
    scansione: nella missione del 2026-10-07 14:5x otto archi attorno al nodo 3164 sono
    finiti fuori dal grafo per tutto il resto della missione a causa di un ostacolo
    marginale a 9 cm dal robot, con il risultato che quel nodo e' rimasto isolato.

    Qui si tolgono solo l'arco e la sua validita': alla prossima rivalutazione, se i dati
    freschi dicono che e' libero, l'arco torna -- che e' esattamente il comportamento di
    rientro che refresh_local_edge_weights e' stato scritto per avere.
    """
    key = (min(idx1, idx2), max(idx1, idx2))
    prm.edge_validity[key] = False
    prm.traversed_edges.discard(key)
    prm.edges[idx1] = [(n, w) for n, w in prm.edges.get(idx1, []) if n != idx2]
    prm.edges[idx2] = [(n, w) for n, w in prm.edges.get(idx2, []) if n != idx1]


def plan_cone_move(robot_xy, robot_yaw, snap, goal_xy, max_dyaw_deg=CONE_SWEEP_MAX_DEG,
                   step_deg=CONE_SWEEP_STEP_DEG, min_gain_m=CONE_MIN_GAIN_M):
    """
    2026-10-07. "Ruota quanto PUOI, non quanto vorresti, e cammina."

    Quando la rotazione verso il prossimo nodo e' rifiutata, la domanda giusta non e' piu'
    "posso girarmi verso il nodo?" ma "qual e' la rotazione piu' grande che posso fare qui,
    e in quale direzione dentro quel cono conviene andare?".

    Perche' serve, misurato sulla missione del 2026-10-07 14:0x (file
    iteration_0_cell_0_1_replan_12_vis.npz, robot in (-2.28, 1.84), obiettivo a 1.26 m):

        rotazione      avanzamento libero   ruotabile   distanza obiettivo dopo
          0 gradi            1.30 m            si           0.50 m  (-0.76)
        +22 gradi (voluta)     --              NO             --
        arretrando 0.60 m      --              --           1.83 m  (+0.57)

    Andare DRITTI, senza ruotare affatto, portava il robot da 1.26 a 0.50 m dall'obiettivo.
    Il codice ha invece arretrato, sette volte di fila, 3.5 m -- e nel punto d'arrivo di
    ogni arretramento la rotazione era comunque impossibile (margine -0.09 m), quindi quegli
    arretramenti erano inutili prima ancora di essere fatti. Il robot era in un corridoio da
    ~60 cm di corsia libera: lì girare richiede 1.21 m di diametro e NON si puo' fare, ma
    camminare si', e il corridoio puntava a 22 gradi dall'obiettivo.

    Criterio di scelta (deciso il 2026-10-07): fra le direzioni che AVVICINANO all'obiettivo
    di almeno min_gain_m, si prende quella con piu' spazio confermato libero. Non la piu'
    libera in assoluto: quella porterebbe il robot via dalla cella.

    Funzione pura. Restituisce None se nessuna direzione del cono avvicina, altrimenti
    dict con: dyaw_deg (rotazione da fare, gradi, puo' essere 0), target (x, y), dist
    (avanzamento), gain (metri guadagnati verso l'obiettivo), reason (motivo del fronte).
    """
    rx, ry = float(robot_xy[0]), float(robot_xy[1])
    gx, gy = float(goal_xy[0]), float(goal_xy[1])
    d0 = float(np.hypot(gx - rx, gy - ry))

    best = None
    rels = np.arange(-abs(max_dyaw_deg), abs(max_dyaw_deg) + 1e-9, abs(step_deg))
    for rel in rels:
        # La rotazione deve essere possibile: sotto la soglia di trascurabilita' non si
        # controlla nulla (e' quello che fa anche rotation_plan), sopra si misura il settore.
        if abs(rel) >= spotGrid.ROTATION_CHECK_MIN_DYAW_DEG:
            ok, _ = spotGrid.rotation_is_clear(
                snap.obstacle_dist, snap.origin_x, snap.origin_y, snap.cell_size,
                rx, ry, robot_yaw, float(np.radians(rel)))
            if not ok:
                continue

        heading = robot_yaw + float(np.radians(rel))
        far_x, far_y = rx + 3.0 * np.cos(heading), ry + 3.0 * np.sin(heading)
        f = arcVerification.compute_safe_frontier([(rx, ry), (far_x, far_y)], snap)
        adv = arcVerification.allowed_advance(f)
        if adv < MIN_PARTIAL_STEP_M:
            continue

        nx, ny = rx + adv * np.cos(heading), ry + adv * np.sin(heading)
        gain = d0 - float(np.hypot(gx - nx, gy - ny))
        if gain < min_gain_m:
            continue
        cand = {'dyaw_deg': float(rel), 'target': (float(nx), float(ny)), 'dist': float(adv),
                'gain': float(gain), 'reason': f['reason']}
        # Piu' spazio libero vince; a pari spazio, la rotazione piu' piccola.
        if best is None or (cand['dist'], -abs(cand['dyaw_deg'])) > (best['dist'], -abs(best['dyaw_deg'])):
            best = cand
    return best


def retreat_would_help(snap, back_xy, robot_yaw, dyaw_deg):
    """
    2026-10-07. L'arretramento di 0.60 m renderebbe possibile la rotazione che serve?

    Misurato sulla missione del 2026-10-07: nei due momenti esaminati la risposta era NO
    (margine -0.09 m a (-2.28, 1.84) e -0.00 m a (-3.91, 1.74)), e il robot ha arretrato
    comunque, sette volte, perche' nessuno gliel'aveva chiesto. In un corridoio la geometria
    0.60 m piu' indietro e' la stessa: arretrare non apre spazio, lo sposta.

    Si valuta la rotazione nel punto di ARRIVO usando la fotografia corrente -- e' una stima,
    perche' da li' il robot vedra' qualcosa di piu', ma e' la stessa informazione su cui si
    e' deciso di arretrare, quindi almeno la decisione e' coerente con se stessa.
    """
    if abs(dyaw_deg) < spotGrid.ROTATION_CHECK_MIN_DYAW_DEG:
        return True, None
    ok, margin = spotGrid.rotation_is_clear(
        snap.obstacle_dist, snap.origin_x, snap.origin_y, snap.cell_size,
        float(back_xy[0]), float(back_xy[1]), float(robot_yaw), float(np.radians(dyaw_deg)))
    return bool(ok), margin


def attempt_enter_cell_from_position(local_grid, global_grid, robot_state_client, command_client,
                                     env, target_row, target_col, global_sampler, prm_graph, mission_folder=None,
                                     iteration=0,
                                     recordingInterface=None, verification_tracker=None, mobility_kwargs=None):

    """Attempt to enter a target cell from the current robot position.

    mobility_kwargs: optional dict of gait overrides (locomotion_hint, ground_mu_hint,
        swing_height) forwarded all the way down to movements.relative_move via navigate_to.
    """
    mobility_kwargs = mobility_kwargs or {}

    print(f"\n[ATTEMPT] Trying to enter cell ({target_row},{target_col}) from current position...")

    # Il robot arriva qui da un altro movimento (navigazione fra celle, ritirata): se e'
    # caduto nel frattempo si rialza prima di tutto. Solleva RobotFallenError se non puo'.
    # Poi esce dal disco pericoloso ripercorrendo i propri passi: rialzato, e' ancora DENTRO
    # il punto della caduta, e da li' ogni arco del PRM verrebbe scartato.
    if _check_and_recover_fall(robot_state_client, command_client,
                               global_grid.global_occupancy_map, "all'inizio del tentativo") is not None:
        retreat_along_traveled_path(env, robot_state_client, command_client,
                                    mobility_kwargs=mobility_kwargs)

    # Recuperiamo il client nativo dall'istanza della nostra classe LocalGrid
    local_grid_client = local_grid.local_grid_client

    # 1. Richiedi TUTTI i layer necessari usando la nuova funzione
    grids_data, main_proto, all_grids_proto = local_grid.return_local_grid(
        ['obstacle_distance', 'terrain', 'terrain_valid'],
        robot_state_client
    )

    if grids_data is None or 'obstacle_distance' not in grids_data:
        print("[ERROR] Rilevamento griglie locali fallito.")
        return False

    # 2. Estrai i valori dal dizionario
    pts = grids_data['obstacle_distance']['pts']
    cells_obstacle_dist = grids_data['obstacle_distance']['values']
    terrain_values = grids_data['terrain']['values']
    valid_values = grids_data['terrain_valid']['values']

    # Estrai metadati geometrici
    num_x = main_proto.local_grid.extent.num_cells_x
    num_y = main_proto.local_grid.extent.num_cells_y
    cell_size = main_proto.local_grid.extent.cell_size

    # =========================================================================
    # --- MODIFICA GLOBAL MAP: Gestione persistente della mappa di occupazione ---
    # =========================================================================
    if global_grid.global_occupancy_map is None:
        global_grid.global_occupancy_map = spotGrid.GlobalGrid(resolution=cell_size)
        print("[MAP] Inizializzata nuova mappa di occupazione globale persistente.")

    # Mappa globale persistente delle altezze -- stessa risoluzione nativa della griglia
    # locale (nessuna riduzione), stesso schema lazy-init della mappa di occupazione.
    if global_grid.global_terrain_map is None:
        global_grid.global_terrain_map = spotGrid.GlobalTerrainGrid(resolution=cell_size)
        print("[MAP] Inizializzata nuova mappa globale persistente del terreno (altezza).")

    global_map = global_grid.global_occupancy_map
    global_terrain_map = global_grid.global_terrain_map

    transforms_snapshot = main_proto.local_grid.transforms_snapshot
    vision_tform_body = get_a_tform_b(transforms_snapshot, VISION_FRAME_NAME, BODY_FRAME_NAME)
    robot_x, robot_y, robot_z = vision_tform_body.position.x, vision_tform_body.position.y, vision_tform_body.position.z

    quat = vision_tform_body.rotation
    robot_yaw = np.arctan2(2.0 * (quat.w * quat.z + quat.x * quat.y), 1.0 - 2.0 * (quat.y ** 2 + quat.z ** 2))

    footprint_mask_2d = local_grid.compute_robot_footprint_mask(
        pts, robot_x, robot_y, robot_yaw, num_x, num_y
    )

    # Correzione terreno (fill invalidi + impronta robot) -- unica fonte di verità
    terrain_real, is_valid = local_grid.correct_terrain(
        terrain_values, valid_values, num_x, num_y,
        robot_footprint_mask=footprint_mask_2d, cell_size=cell_size,
        unwritten_value=_raw_zero(grids_data), unwritten_scale=_raw_scale(grids_data)
    )
    _log_ground_filter(local_grid, "scan-iniziale")

    # Calcolo derivate geometriche (Pendenza e Rugosità) sul terreno GIA' corretto
    grad_values, rough_values = local_grid.compute_gradient_and_roughness(
        terrain_real, valid_values, num_x, num_y, cell_size, is_valid=is_valid
    )

    grad_for_prm = grad_values.copy()
    rough_for_prm = rough_values.copy()

    # Fusione (VETO): pura logica di soglia, il terreno è già stato corretto sopra.
    # Il gradiente NON entra più in questo veto per-cella -- vedi note in fuse_obstacle_mask().
    obstacle_mask = local_grid.fuse_obstacle_mask(
        cells_obstacle_dist=cells_obstacle_dist,
        rough_values=rough_values,
        is_valid=is_valid,
        obstacle_threshold=spotGrid.OBSTACLE_THRESHOLD,
        rough_threshold=spotGrid.ROUGH_THRESHOLD
    )

    _log_footprint_diagnostic(local_grid, pts, robot_x, robot_y, robot_yaw,
                              num_x, num_y, obstacle_mask, "scan-cella",
                              cells_obstacle_dist=cells_obstacle_dist,
                              rough_values=rough_values, is_valid=is_valid)

    # Aggiorna la mappa di occupazione globale con i dati appena processati
    global_map.update(pts, _mask_for_global_map(obstacle_mask, is_valid))
    # Aggiorna anche la mappa globale del terreno -- solo altezze già corrette e valide
    global_terrain_map.update(pts, terrain_real, is_valid)

    # Rendiamo la mappa terreno disponibile al PRM come fallback per la pendenza
    prm_graph.set_global_terrain_map(global_terrain_map)

    # Salvataggio periodico su disco delle mappe globali (.pkl)
    if mission_folder is not None:
        pkl_map_path = os.path.join(mission_folder, "global_occupancy_map.pkl")
        global_map.save_map(pkl_map_path)
        pkl_terrain_path = os.path.join(mission_folder, "global_terrain_map.pkl")
        global_terrain_map.mission_z0 = _MISSION_CTX.get('mission_z0')
        global_terrain_map.save_map(pkl_terrain_path)

        # Le mappe globali crescono per tutta la missione e non vengono mai potate: a
        # risoluzione nativa (3 cm) un'area di 25x25 m sono ~0.7 milioni di celle per
        # mappa, cioe' qualche centinaio di MB fra occupazione e terreno.
        print(f"[MAP] Dimensione mappe globali -- occupazione: {len(global_map.grid)} celle, "
              f"terreno: {len(global_terrain_map.height_count)} celle")

    save_path = os.path.join(mission_folder,
                             f"iteration_{iteration}_cell_{target_row}_{target_col}.png") if mission_folder else None

    # =========================================================================
    # SALVATAGGIO DEI VALORI NUMERICI IN FORMATO NumPy (.npy)
    # =========================================================================
    if mission_folder is not None:
        try:
            obstacle_grid_2d = obstacle_mask.reshape((num_y, num_x))
            terrain_grid_2d = terrain_real.reshape((num_y, num_x))

            npy_obstacle_path = os.path.join(mission_folder,
                                             f"iteration_{iteration}_cell_{target_row}_{target_col}_obstacles.npy")
            npy_terrain_path = os.path.join(mission_folder,
                                            f"iteration_{iteration}_cell_{target_row}_{target_col}_terrain.npy")

            np.save(npy_obstacle_path, obstacle_grid_2d)
            np.save(npy_terrain_path, terrain_grid_2d)

            print(f"[DATA SAVE] Matrici .npy salvate per cella ({target_row},{target_col})")

        except Exception as e:
            print(f"[DATA SAVE] Impossibile salvare i file .npy: {e}")

    # ================================================================== #
    # --- [CORREZIONE PRM] AGGIORNAMENTO LOCALE DI GRADIENTE E RUGOSITÀ ---
    # ================================================================== #
    local_x_min, local_x_max = pts[:, 0].min(), pts[:, 0].max()
    local_y_min, local_y_max = pts[:, 1].min(), pts[:, 1].max()

    grid_frame_name = main_proto.local_grid.frame_name_local_grid_data
    vision_tform_grid = get_a_tform_b(transforms_snapshot, VISION_FRAME_NAME, grid_frame_name)
    grid_origin_x = vision_tform_grid.position.x
    grid_origin_y = vision_tform_grid.position.y

    grad_values_2d = grad_for_prm.reshape((num_y, num_x))
    rough_values_2d = rough_for_prm.reshape((num_y, num_x))
    terrain_values_2d = terrain_real.reshape((num_y, num_x))

    prm_graph.update_local_grid_data(
        grad_values_2d=grad_values_2d,
        rough_values_2d=rough_values_2d,
        terrain_values_2d=terrain_values_2d,
        valid_2d=slope_usable_mask(is_valid, cells_obstacle_dist, (num_y, num_x)),
        grid_origin_x=grid_origin_x,
        grid_origin_y=grid_origin_y,
        cell_size=cell_size
    )

    # Aggiorna gradienti e rugosità SOLO per i nodi PRM che ricadono all'interno della local grid
    for node_id, (nx, ny) in prm_graph.nodes.items():
        if local_x_min <= nx <= local_x_max and local_y_min <= ny <= local_y_max:
            dx = nx - grid_origin_x
            dy = ny - grid_origin_y

            col_idx = int(dx / cell_size)
            row_idx = int(dy / cell_size)

            if 0 <= row_idx < num_y and 0 <= col_idx < num_x:
                flat_idx = row_idx * num_x + col_idx
                if flat_idx < len(grad_values):
                    prm_graph.node_gradients[node_id] = grad_values[flat_idx]
                if flat_idx < len(rough_values):
                    prm_graph.node_roughness[node_id] = rough_values[flat_idx]
        else:
            if node_id not in prm_graph.node_gradients:
                prm_graph.node_gradients[node_id] = 0.0
            if node_id not in prm_graph.node_roughness:
                prm_graph.node_roughness[node_id] = 0.0

    t_build = time.perf_counter()
    prm_graph.build_graph(global_map=global_map, edge_safety_margin=spotGrid.PRM_EDGE_SAFETY_MARGIN_M)
    _timing_add('costruzione grafo', time.perf_counter() - t_build)
    print(f"[TEMPO] costruzione completa del grafo: {time.perf_counter() - t_build:.2f} s")

    # ================================================================== #

    # Ricerca del punto target e pianificazione
    target_x, target_y, valid_samples, rejected_samples = find_best_point_in_cell(
        robot_x, robot_y, env, target_row, target_col, pts, cells_obstacle_dist, global_sampler,
        global_map=global_map)

    if target_x is None or target_y is None:
        print(f"[FAIL] No clear path found to cell ({target_row},{target_col}) from current position")
        visualize_grid_with_candidates(
            pts=pts, terrain_real=terrain_real, obstacle_mask=obstacle_mask,
            robot_x=robot_x, robot_y=robot_y,
            candidates={'rejected': rejected_samples, 'valid': valid_samples},
            chosen_point=(target_x, target_y), iteration=iteration, env=env,
            save_path=save_path, prm_graph=prm_graph, chosen_path=None,
            cells_obstacle_dist=cells_obstacle_dist, valid_values=valid_values,
            grad_values=grad_values, rough_values=rough_values
        )
        return False

    # --- Nodi di partenza e di arrivo: SEMPRE nuovi, nella posizione reale ---------------
    robot_node_id = prm_graph.add_stop_node(
        robot_x, robot_y, global_map, spotGrid.PRM_EDGE_SAFETY_MARGIN_M,
        label="partenza tentativo")
    prm_graph.set_robot_node(robot_node_id, 0.0)
    # Briciola GraphNav del punto di partenza del tentativo: e' quello a cui serve poter
    # tornare se la cella si rivela un vicolo cieco (2026-10-07).
    graphnav_drop_breadcrumb(env, recordingInterface, robot_state_client, robot_node_id,
                             target_row, target_col, force=True)
    goal_id = prm_graph.add_stop_node(
        target_x, target_y, global_map, spotGrid.PRM_EDGE_SAFETY_MARGIN_M,
        label=f"obiettivo cella ({target_row},{target_col})", is_robot=False)
    prm_graph.set_robot_node(robot_node_id, 0.0)
    path_ids = prm_graph.find_path_dijkstra(robot_node_id, goal_id)

    if path_ids is None:
        print(
            f"[GRAFO DISCONNESSO] Nessun percorso nel PRM dal nodo {robot_node_id} al nodo "
            f"{goal_id}: i due nodi stanno in componenti separate del grafo. Non e' un "
            f"errore di Dijkstra -- vuol dire che tutti gli archi di collegamento sono "
            f"stati vetati (occupazione o pendenza). Vedi i conteggi in [PRM MISSION SUMMARY].")
        print(f" -> Coordinate partenza robot: ({robot_x:.2f}, {robot_y:.2f})")
        print(f" -> Coordinate target cella: ({target_x:.2f}, {target_y:.2f})")

    def _waypoints_from_ids(ids):
        """[(node_id, x, y), ...] senza il primo nodo (la posizione del robot)."""
        out = []
        for nid in ids[1:]:
            nx, ny = prm_graph.get_node_position(nid)
            out.append((nid, nx, ny))
        return out

    full_path_coords = None
    path_waypoints = []

    if path_ids is not None:
        full_path_coords = [prm_graph.get_node_position(nid) for nid in path_ids]
        path_waypoints = _waypoints_from_ids(path_ids)
    else:
        path_waypoints = None

    visualize_grid_with_candidates(
        pts=pts, terrain_real=terrain_real, obstacle_mask=obstacle_mask,
        robot_x=robot_x, robot_y=robot_y,
        candidates={'rejected': rejected_samples, 'valid': valid_samples},
        chosen_point=(target_x, target_y), iteration=iteration, env=env,
        save_path=save_path, prm_graph=prm_graph, chosen_path=full_path_coords,
        cells_obstacle_dist=cells_obstacle_dist, valid_values=valid_values,
        grad_values=grad_values, rough_values=rough_values
    )

    # ===================================================================================
    # CICLO DI NAVIGAZIONE CON IL FRONTE SICURO (2026-10-06, rivisto il 2026-10-07)
    #
    # A ogni giro: scansione fresca, aggiornamento mappe e PRM, eventuale ripianificazione,
    # poi il FRONTE: fin dove il percorso e' confermato libero adesso. Il robot avanza
    # fino a li' (meno meta' corpo e margine), sul primo tratto del percorso.
    #
    # 2026-10-07, tre cambiamenti:
    #   - il muso punta SEMPRE il prossimo punto: la marcia di traverso e' eliminata, e se
    #     la rotazione non e' possibile si arretra dritti (vedi plan_straight_retreat);
    #   - la rotazione si giudica sulle celle davvero spazzate (spotGrid.rotation_is_clear),
    #     non sul cerchio circoscritto;
    #   - se l'obiettivo risulta irraggiungibile sui dati freschi se ne scegle un altro
    #     nella cella, invece di aspettare e dichiarare un blocco (goal_still_reachable).
    # ===================================================================================
    replan_counter = 0
    no_progress = 0
    block_replans = 0
    loop_iter = 0
    rotation_fails = {}     # tratto -> quante volte la rotazione verso di esso e' stata rifiutata
    retreats_in_a_row = 0   # arretramenti consecutivi: azzerato da un avanzamento vero
    early_replans = 0       # ripianificazioni alla prima cella bloccante (ramo 'wait')
    arrival_refusals = 0    # passi rifiutati perche' la posa d'arrivo non ha uscite
    goal_repicks = 0        # quante volte si e' riscelto il punto obiettivo
    tried_goals = []        # punti obiettivo gia' provati e risultati irraggiungibili

    while path_waypoints and len(path_waypoints) > 0:
        loop_iter += 1
        t_iter = time.perf_counter()
        if loop_iter > MAX_LOOP_ITERATIONS:
            print(f"[STALLO] {MAX_LOOP_ITERATIONS} giri del ciclo di navigazione senza arrivare: "
                  f"interrompo il tentativo sulla cella ({target_row},{target_col}).")
            if verification_tracker is not None:
                verification_tracker.clear_path()
            return False

        if _check_and_recover_fall(robot_state_client, command_client, global_map,
                                   "durante il ciclo di navigazione") is not None:
            if verification_tracker is not None:
                verification_tracker.clear_path()
            retreat_along_traveled_path(env, robot_state_client, command_client,
                                        mobility_kwargs=mobility_kwargs)
            return False

        robot_x, robot_y, robot_z, robot_quat = spotUtils.getPosition(robot_state_client)

        robot_yaw = np.arctan2(2.0 * (robot_quat.w * robot_quat.z + robot_quat.x * robot_quat.y),
                               1.0 - 2.0 * (robot_quat.y ** 2 + robot_quat.z ** 2))

        # 1. LETTURA AGGIORNATA DELLA SCENA LOCALE (fetch fresco, PRIMA di tutto il resto)
        grids_data_up, main_proto_up, all_grids_proto_up = local_grid.return_local_grid(
            ['obstacle_distance', 'terrain', 'terrain_valid'],
            robot_state_client
        )

        if grids_data_up is None or 'obstacle_distance' not in grids_data_up:
            print("[ERROR] Rilevamento locale fallito durante il tracking. Arresto precauzionale.")
            if verification_tracker is not None:
                verification_tracker.clear_path()
            return False
        t_scan = time.perf_counter()

        pts_up = grids_data_up['obstacle_distance']['pts']
        cells_obs_up = grids_data_up['obstacle_distance']['values']
        terrain_up = grids_data_up['terrain']['values']
        valid_up = grids_data_up['terrain_valid']['values']

        lg_proto_up = main_proto_up
        num_x_up = lg_proto_up.local_grid.extent.num_cells_x
        num_y_up = lg_proto_up.local_grid.extent.num_cells_y
        cell_size_up = lg_proto_up.local_grid.extent.cell_size

        transforms_snapshot_up = main_proto_up.local_grid.transforms_snapshot
        grid_frame_name_up = main_proto_up.local_grid.frame_name_local_grid_data
        vision_tform_grid_up = get_a_tform_b(transforms_snapshot_up, VISION_FRAME_NAME, grid_frame_name_up)
        grid_origin_x_up = vision_tform_grid_up.position.x
        grid_origin_y_up = vision_tform_grid_up.position.y

        footprint_mask_2d = local_grid.compute_robot_footprint_mask(
            pts_up, robot_x, robot_y, robot_yaw, num_x_up, num_y_up
        )

        terrain_real_up, is_valid_up, unwritten_up = local_grid.correct_terrain(
            terrain_up, valid_up, num_x_up, num_y_up,
            robot_footprint_mask=footprint_mask_2d, cell_size=cell_size_up,
            unwritten_value=_raw_zero(grids_data_up), unwritten_scale=_raw_scale(grids_data_up),
            return_unwritten=True
        )
        _log_ground_filter(local_grid, "scan-passo")

        grad_up, rough_up = local_grid.compute_gradient_and_roughness(
            terrain_real_up, valid_up, num_x_up, num_y_up, cell_size_up, is_valid=is_valid_up
        )

        obstacle_mask_updated = local_grid.fuse_obstacle_mask(
            cells_obstacle_dist=cells_obs_up,
            rough_values=rough_up,
            is_valid=is_valid_up,
            obstacle_threshold=spotGrid.OBSTACLE_THRESHOLD,
            rough_threshold=spotGrid.ROUGH_THRESHOLD
        )

        _log_footprint_diagnostic(local_grid, pts_up, robot_x, robot_y, robot_yaw,
                                  num_x_up, num_y_up, obstacle_mask_updated, "scan-passo",
                                  cells_obstacle_dist=cells_obs_up,
                                  rough_values=rough_up, is_valid=is_valid_up)

        t_proc = time.perf_counter()
        global_map.update(pts_up, _mask_for_global_map(obstacle_mask_updated, is_valid_up))
        global_terrain_map.update(pts_up, terrain_real_up, is_valid_up)
        t_maps = time.perf_counter()

        # 2. iniettiamo i dati locali freschi nel PRM e ripianifichiamo dal nodo attuale
        grad_up_2d = grad_up.reshape((num_y_up, num_x_up))
        rough_up_2d = rough_up.reshape((num_y_up, num_x_up))
        terrain_up_2d = terrain_real_up.reshape((num_y_up, num_x_up))

        prm_graph.update_local_grid_data(
            grad_values_2d=grad_up_2d,
            rough_values_2d=rough_up_2d,
            terrain_values_2d=terrain_up_2d,
            valid_2d=slope_usable_mask(is_valid_up, cells_obs_up, (num_y_up, num_x_up)),
            grid_origin_x=grid_origin_x_up,
            grid_origin_y=grid_origin_y_up,
            cell_size=cell_size_up
        )
        touched_nodes = prm_graph.refresh_local_edge_weights(
            global_map=global_map, edge_safety_margin=spotGrid.PRM_EDGE_SAFETY_MARGIN_M)

        # Fotografia della scansione appena fatta: serve gia' qui per scartare i nodi negli
        # ostacoli, e poi al fronte sicuro. Va creata UNA volta sola per scansione:
        # make_grid_snapshot aggiorna anche il filtro temporale degli ostacoli.
        snap = arcVerification.make_grid_snapshot(
            cells_obs_up, rough_up, is_valid_up, valid_up, terrain_real_up,
            num_x_up, num_y_up, grid_origin_x_up, grid_origin_y_up, cell_size_up,
            robot_x, robot_y, unwritten=unwritten_up)

        # Nodi DENTRO gli ostacoli: fuori dal grafo (2026-10-07). Vedi
        # prune_prm_nodes_in_obstacles -- gli archi erano gia' vetati, i nodi mai.
        pruned_nodes = prune_prm_nodes_in_obstacles(prm_graph, snap)
        if pruned_nodes:
            MISSION_STATS['nodes_pruned'] += len(pruned_nodes)
            print(f"[NODI] {len(pruned_nodes)} nodi scartati: meno di "
                  f"{PRM_NODE_MIN_CLEARANCE_M:.2f} m di spazio libero dove li vediamo "
                  f"(primi: {pruned_nodes[:8]}).")

        if touched_nodes:
            current_plan_ids = [robot_node_id] + [w[0] for w in path_waypoints]
            chosen_path_ids = prm_graph.find_path_dijkstra(robot_node_id, goal_id,
                                                           current_path_ids=current_plan_ids, margin=0.10)

            if chosen_path_ids is None:
                print(f"[REPLAN FALLITO] Nessun percorso esistente da {robot_node_id} a "
                      f"{goal_id} nel grafo aggiornato. Mantengo il percorso precedente: il "
                      f"fronte sicuro decide comunque fin dove si puo' avanzare.")
            elif chosen_path_ids != current_plan_ids:
                path_waypoints = _waypoints_from_ids(chosen_path_ids)
                full_path_coords = [prm_graph.get_node_position(nid) for nid in chosen_path_ids]
                no_progress = 0   # percorso nuovo: la conferma di un blocco riparte da zero
                print(f"[REPLAN] Percorso aggiornato ({len(touched_nodes)} nodi locali rivalutati): "
                      f"{len(path_waypoints)} waypoint rimanenti.")

                replan_counter += 1
                MISSION_STATS['replan_count'] += 1
                replan_save_path = os.path.join(
                    mission_folder, f"iteration_{iteration}_cell_{target_row}_{target_col}_replan_{replan_counter}.png"
                ) if mission_folder else None

                visualize_grid_with_candidates(
                    pts=pts_up, terrain_real=terrain_real_up, obstacle_mask=obstacle_mask_updated,
                    robot_x=robot_x, robot_y=robot_y,
                    candidates={'rejected': [], 'valid': []},
                    chosen_point=(target_x, target_y), iteration=iteration, env=env,
                    save_path=replan_save_path, prm_graph=prm_graph, chosen_path=full_path_coords,
                    cells_obstacle_dist=cells_obs_up, valid_values=valid_up,
                    grad_values=grad_up, rough_values=rough_up,
                    include_diagnostics=False  # lightweight: main view + global map only
                )
            else:
                print(f"[REPLAN STABILE] Percorso invariato da {robot_node_id} a {goal_id} "
                      f"({len(touched_nodes)} nodi locali rivalutati): l'alternativa non "
                      f"supera la soglia di stabilita'. Proseguo sul piano corrente.")

        if not path_waypoints:
            break
        t_prm = time.perf_counter()

        # 3. FRONTE SICURO sulla scansione appena fatta ---------------------------------
        # La scorciatoia rispetta gli archi gia' scartati (2026-10-07): vedi shortcut_index.
        k_short = shortcut_index((robot_x, robot_y), path_waypoints, snap, global_map,
                                 prm=prm_graph, robot_node_id=robot_node_id)
        if k_short > 0:
            skipped = [w[0] for w in path_waypoints[:k_short]]
            path_waypoints = path_waypoints[k_short:]
            print(f"[SCORCIATOIA] Tratto diretto confermato libero verso il nodo {path_waypoints[0][0]}: "
                  f"salto {len(skipped)} nodi intermedi {skipped}.")
            MISSION_STATS['shortcuts'] += 1
        polyline = [(robot_x, robot_y)] + [(x, y) for _, x, y in path_waypoints]
        frontier = arcVerification.compute_safe_frontier(polyline, snap)
        print(arcVerification.describe_frontier(frontier))

        path_version = None
        if verification_tracker is not None:
            path_version = verification_tracker.update_path(polyline[1:])

        decision = decide_next_move((robot_x, robot_y), path_waypoints, frontier, no_progress)
        next_node_id, next_x, next_y = path_waypoints[0]

        # 4. ROTAZIONE (riscritta il 2026-10-07) -----------------------------------------
        # Il muso punta SEMPRE il punto da raggiungere. Se la rotazione non e' possibile NON
        # si cammina di traverso (di traverso Spot e' largo 1.10 m e il fronte pretende 0.59
        # invece di 0.30: il vecchio ripiego rendeva il robot piu' largo proprio dove lo
        # spazio e' poco). Si arretra dritti lungo l'asse del corpo e al giro dopo si riprova
        # da li'; alla seconda volta sullo stesso tratto il tratto si scarta.
        keep_heading = False     # mai piu' True: tenuto per i dati salvati e per navigate_to
        if decision['kind'] == 'move':
            rot = rotation_plan(robot_yaw, (robot_x, robot_y), decision['target'],
                                frontier['od_robot'], snap=snap)
            if not rot['turn']:
                MISSION_STATS['rotation_refusals'] += 1
                edge_key = (robot_node_id, next_node_id)
                rotation_fails[edge_key] = rotation_fails.get(edge_key, 0) + 1
                margin_txt = (f"margine {rot['margin']:+.2f} m sulle celle spazzate"
                              if rot['margin'] is not None else f"motivo: {rot['why']}")
                print(f"[ROTAZIONE] Servono {rot['dyaw_deg']:+.0f} gradi verso il nodo "
                      f"{next_node_id}, ma il corpo toccherebbe ({margin_txt}; "
                      f"obstacle_distance sotto il robot {frontier['od_robot']:.2f} m). "
                      f"Non cammino di traverso: tentativo "
                      f"{rotation_fails[edge_key]}/{MAX_ROTATION_FAILS_SAME_EDGE}.")

                # --- 4a. PRIMA DI TUTTO: posso ANDARE AVANTI dentro il cono di rotazione che
                # mi e' concesso? (2026-10-07) E' la correzione piu' importante delle
                # quattro: nella missione del 2026-10-07 andare dritti, senza ruotare,
                # avvicinava l'obiettivo di 0.76 m, e il robot ha arretrato perdendone 0.57.
                # Vedi plan_cone_move per i numeri misurati.
                cone = plan_cone_move((robot_x, robot_y), robot_yaw, snap, (target_x, target_y))
                if cone is not None:
                    print(f"[CONO] Non posso girarmi di {rot['dyaw_deg']:+.0f} gradi, ma ruotando di "
                          f"{cone['dyaw_deg']:+.0f} posso avanzare {cone['dist']:.2f} m "
                          f"({cone['reason']}) e avvicinarmi di {cone['gain']:.2f} m. Vado, "
                          f"invece di arretrare.")
                    if verification_tracker is not None:
                        verification_tracker.clear_path()
                    MISSION_STATS['cone_moves'] += 1
                    cx, cy = cone['target']
                    ok_c, dist_c, _ = navigate_to(
                        cx, cy, robot_x, robot_y, robot_state_client, command_client,
                        get_a_tform_b(lg_proto_up.local_grid.transforms_snapshot,
                                      VISION_FRAME_NAME, BODY_FRAME_NAME),
                        should_abort=None, keep_heading=False, **mobility_kwargs)
                    if dist_c > 0.05:
                        if not hasattr(env, '_traveled_arcs'):
                            env._traveled_arcs = []
                        ax_c, ay_c, _, _ = spotUtils.getPosition(robot_state_client)
                        env._traveled_arcs.append((robot_x, robot_y, ax_c, ay_c))
                        robot_node_id = prm_graph.add_stop_node(
                            ax_c, ay_c, global_map, spotGrid.PRM_EDGE_SAFETY_MARGIN_M,
                            came_from=robot_node_id, label="sosta (avanzamento nel cono)")
                        prm_graph.set_robot_node(robot_node_id, 0.0)
                        retreats_in_a_row = 0      # ha fatto strada vera: la catena si rompe
                    else:
                        print(f"[CONO] Non sono riuscito ad avanzare (percorsi {dist_c:.2f} m).")
                    no_progress = 0
                    continue

                # --- 4b. Arretrare, ma SOLO se serve a qualcosa (2026-10-07).
                back = plan_straight_retreat((robot_x, robot_y), robot_yaw, snap)
                helps, margin_back = (False, None)
                if back is not None:
                    helps, margin_back = retreat_would_help(
                        snap, (back[0], back[1]), robot_yaw, rot['dyaw_deg'])
                    if not helps:
                        MISSION_STATS['retreats_suppressed'] += 1
                        mb = "" if margin_back is None else f" (margine {margin_back:+.2f} m da li')"
                        print(f"[ARRETRAMENTO] Potrei arretrare di {back[2]:.2f} m, ma da li' la "
                              f"rotazione di {rot['dyaw_deg']:+.0f} gradi resterebbe impossibile{mb}: "
                              f"non lo faccio, sposterei il problema di 60 cm.")
                if (back is not None and helps
                        and retreats_in_a_row < MAX_STRAIGHT_RETREATS_IN_A_ROW
                        and rotation_fails[edge_key] < MAX_ROTATION_FAILS_SAME_EDGE):
                    bx, by, bdist = back
                    retreats_in_a_row += 1
                    print(f"[ARRETRAMENTO] Indietro dritto di {bdist:.2f} m verso "
                          f"({bx:.2f}, {by:.2f}), senza girarmi, per fare spazio alla rotazione "
                          f"({retreats_in_a_row}/{MAX_STRAIGHT_RETREATS_IN_A_ROW} consecutivi).")
                    if verification_tracker is not None:
                        verification_tracker.clear_path()
                    MISSION_STATS['straight_retreats'] += 1
                    ok_back, dist_back = movements.move_to_world_point_without_turning(
                        bx, by, "vision", command_client, robot_state_client, **mobility_kwargs)
                    if dist_back > 0.05:
                        if not hasattr(env, '_traveled_arcs'):
                            env._traveled_arcs = []
                        ax_b, ay_b, _, _ = spotUtils.getPosition(robot_state_client)
                        env._traveled_arcs.append((robot_x, robot_y, ax_b, ay_b))
                        robot_node_id = prm_graph.add_stop_node(
                            ax_b, ay_b, global_map, spotGrid.PRM_EDGE_SAFETY_MARGIN_M,
                            came_from=robot_node_id, label="sosta (arretramento per ruotare)")
                        prm_graph.set_robot_node(robot_node_id, 0.0)
                    else:
                        print(f"[ARRETRAMENTO] Non sono riuscito ad arretrare "
                              f"(percorsi {dist_back:.2f} m).")
                    no_progress = 0
                    continue

                if back is None:
                    print("[ARRETRAMENTO] Non c'e' spazio per arretrare dritto.")
                elif helps and retreats_in_a_row >= MAX_STRAIGHT_RETREATS_IN_A_ROW:
                    print(f"[ARRETRAMENTO] Gia' {retreats_in_a_row} arretramenti di fila senza "
                          f"combinare niente: smetto di arretrare.")

                # Ne' il cono ne' l'arretramento: e' una trappola o solo un arco sbagliato?
                # (2026-10-07) Se e' una trappola non ha senso scartare archi: in cella (0,2)
                # ne sono stati scartati quattro di fila, tutti chiedendo rotazioni di +/-179
                # gradi, prima di arrivare comunque alla ritirata.
                trapped_rot, fan_rot = is_trapped((robot_x, robot_y), robot_yaw, snap)
                n_us_r, fan_txt_r = describe_escape_fan(fan_rot)
                if trapped_rot:
                    MISSION_STATS['traps_detected'] += 1
                    print(f"[TRAPPOLA] Non posso girarmi verso {next_node_id} e nessuna delle "
                          f"{len(fan_rot)} direzioni del giro e' insieme percorribile e "
                          f"raggiungibile girandosi. Le piu' larghe: {fan_txt_r}. Ripercorro il "
                          f"tratto da cui sono venuto.")
                    if verification_tracker is not None:
                        verification_tracker.clear_path()
                    # Ritorno con GraphNav invece che a zampe all'indietro (2026-10-07): si torna di un
                    # waypoint, si riprova a pianificare da li', e cosi' via fino a
                    # GRAPHNAV_MAX_BACKTRACK_HOPS. La ritirata a zampe resta dentro come ultima rete.
                    esc = escape_by_backtracking(
                        env, prm_graph, goal_id, recordingInterface, robot_state_client,
                        command_client, global_map, mobility_kwargs, pts_up, obstacle_mask_updated,
                        label=f"dal nodo {robot_node_id}")
                    if esc['ok']:
                        path_waypoints = _waypoints_from_ids(esc['ids'])
                        full_path_coords = [prm_graph.get_node_position(nid) for nid in esc['ids']]
                        robot_node_id = esc['robot_node']
                        no_progress = 0
                        early_replans = 0
                        retreats_in_a_row = 0
                        continue
                    return False
                print(f"[ROTAZIONE] Direzioni utilizzabili da qui: {n_us_r}/{len(fan_rot)} "
                      f"({fan_txt_r}).")
                print(f"[ROTAZIONE] Scarto il tratto {robot_node_id}->{next_node_id}: "
                      f"richiede di girarsi dove non c'e' spazio DA QUI. Ripianifico.")
                # Scarto NON permanente (2026-10-07): un rifiuto di rotazione dipende dalla POSA
                # del robot, non dall'arco -- lo stesso arco, affrontato da un altro punto o con
                # un altro muso, puo' essere percorribilissimo. Nella missione del 2026-10-07
                # 14:5x in cella (0,2) quattro archi sono stati condannati per sempre perche'
                # chiedevano rotazioni di +/-179 gradi da un punto stretto. Permanente resta solo
                # cio' che e' confermato da tre scansioni ferme o da un fallimento fisico.
                mark_edge_blocked_soft(prm_graph, robot_node_id, next_node_id)
                new_ids = prm_graph.find_path_dijkstra(robot_node_id, goal_id)
                if new_ids is not None:
                    path_waypoints = _waypoints_from_ids(new_ids)
                    full_path_coords = [prm_graph.get_node_position(nid) for nid in new_ids]
                    no_progress = 0
                    continue
                print("[ROTAZIONE] Nessuna alternativa nel grafo dal punto in cui sono. "
                      "Mi stacco arretrando e rimando la cella.")
                if verification_tracker is not None:
                    verification_tracker.clear_path()
                # Ritorno con GraphNav invece che a zampe all'indietro (2026-10-07): si torna di un
                # waypoint, si riprova a pianificare da li', e cosi' via fino a
                # GRAPHNAV_MAX_BACKTRACK_HOPS. La ritirata a zampe resta dentro come ultima rete.
                esc = escape_by_backtracking(
                    env, prm_graph, goal_id, recordingInterface, robot_state_client,
                    command_client, global_map, mobility_kwargs, pts_up, obstacle_mask_updated,
                    label=f"dal nodo {robot_node_id}")
                if esc['ok']:
                    path_waypoints = _waypoints_from_ids(esc['ids'])
                    full_path_coords = [prm_graph.get_node_position(nid) for nid in esc['ids']]
                    robot_node_id = esc['robot_node']
                    no_progress = 0
                    early_replans = 0
                    retreats_in_a_row = 0
                    continue
                return False
        t_front = time.perf_counter()

        _timing_add('scansione', t_scan - t_iter)
        _timing_add('elaborazione griglia', t_proc - t_scan)
        _timing_add('mappe globali', t_maps - t_proc)
        _timing_add('PRM e ripianificazione', t_prm - t_maps)
        _timing_add('fronte e decisione', t_front - t_prm)
        print(f"[TEMPO] scansione {t_scan - t_iter:.2f} s | elaborazione {t_proc - t_scan:.2f} | "
              f"mappe {t_maps - t_proc:.2f} | PRM {t_prm - t_maps:.2f} | fronte {t_front - t_prm:.2f} "
              f"| decisione: {decision['kind']}")

        if SAVE_SCAN_BUNDLES and mission_folder is not None:
            gf = local_grid.last_ground_filter
            shape_up = (num_y_up, num_x_up)
            _save_npz_async(
                os.path.join(mission_folder, "scans",
                             f"scan_it{iteration:03d}_cell{target_row}_{target_col}_{loop_iter:03d}.npz"),
                time=time.time(), iteration=iteration, cell=np.array([target_row, target_col]),
                loop_iter=loop_iter,
                terrain_raw=np.asarray(terrain_up, dtype=np.float32).reshape(shape_up),
                terrain_valid_raw=np.asarray(valid_up, dtype=np.float32).reshape(shape_up),
                obstacle_distance=np.asarray(cells_obs_up, dtype=np.float32).reshape(shape_up),
                terrain_corrected=np.asarray(terrain_real_up, dtype=np.float32).reshape(shape_up),
                is_valid=np.asarray(is_valid_up, dtype=bool).reshape(shape_up),
                roughness=np.asarray(rough_up, dtype=np.float32).reshape(shape_up),
                gradient=np.asarray(grad_up, dtype=np.float32).reshape(shape_up),
                obstacle_mask=np.asarray(obstacle_mask_updated, dtype=np.int8).reshape(shape_up),
                footprint_mask=(np.zeros(shape_up, bool) if footprint_mask_2d is None
                                else np.asarray(footprint_mask_2d, dtype=bool).reshape(shape_up)),
                grid_origin=np.array([grid_origin_x_up, grid_origin_y_up]), cell_size=cell_size_up,
                robot_xyz=np.array([robot_x, robot_y, robot_z]), robot_yaw=robot_yaw,
                ground_z=np.nan if gf.get('ground_z') is None else gf['ground_z'],
                ground_n_above=gf.get('n_above', 0),
                mission_z0=_mission_z0(),
                terrain_raw_zero=np.nan if _raw_zero(grids_data_up) is None else _raw_zero(grids_data_up),
                terrain_scale=np.nan if not _raw_scale(grids_data_up) else _raw_scale(grids_data_up),
                unwritten=np.asarray(unwritten_up, dtype=bool).reshape(shape_up),
                polyline=np.array(polyline), path_node_ids=np.array([w[0] for w in path_waypoints]),
                robot_node_id=robot_node_id, goal_xy=np.array([target_x, target_y]),
                frontier_dist=frontier['dist'], frontier_reason=frontier['reason'],
                frontier_stop_xy=np.array(frontier['stop_xy']),
                frontier_value=np.nan if frontier['value'] is None else frontier['value'],
                frontier_od_robot=frontier['od_robot'], frontier_half_along=frontier['half_along'],
                frontier_clearance=frontier['clearance_nominal'], frontier_path_len=frontier['path_len'],
                allowed_advance=arcVerification.allowed_advance(frontier),
                keep_heading=keep_heading, no_progress=no_progress,
                decision_kind=decision['kind'],
                decision_target=np.array(decision.get('target', (np.nan, np.nan)), dtype=np.float64))
            MISSION_STATS['scan_bundles'] += 1

        if decision['kind'] == 'reached_node':
            robot_node_id = next_node_id
            prm_graph.set_robot_node(robot_node_id, 0.0)
            path_waypoints.pop(0)
            no_progress = 0
            continue

        if decision['kind'] == 'arrived':
            print(f"[OBIETTIVO] Il fronte non permette gli ultimi {frontier['path_len']:.2f} m "
                  f"({frontier['reason']}), entro la tolleranza di {GOAL_REACHED_TOLERANCE_M:.2f} m: "
                  f"considero raggiunto l'obiettivo.")
            break

        if decision['kind'] == 'wait':
            # 2026-10-07 -- Prima di insistere: l'obiettivo e' ancora raggiungibile sui dati
            # FRESCHI? Il punto viene scelto all'inizio del tentativo, da metri di distanza,
            # quando la cella e' ancora tutta ignota; arrivandoci puo' risultare a ridosso di
            # un muro, e allora nessun numero di scansioni lo rendera' raggiungibile. Nella
            # missione del 2026-10-07 l'obiettivo aveva obstacle_distance 0.27 contro i 0.30
            # richiesti dal fronte, e il robot si e' bloccato a 63 cm da esso mentre entro
            # 80 cm c'erano 808 celle valide.
            if goal_repicks < MAX_BLOCK_REPLANS and not goal_still_reachable(snap, target_x, target_y):
                goal_repicks += 1
                MISSION_STATS['goal_repicks'] += 1
                r_g, c_g, in_g = arcVerification._cells(
                    snap, np.array([float(target_x)]), np.array([float(target_y)]))
                od_goal = float(snap.obstacle_dist[r_g[0], c_g[0]]) if in_g[0] else float('nan')
                print(f"[TARGET] L'obiettivo ({target_x:.2f}, {target_y:.2f}) ha "
                      f"obstacle_distance {od_goal:.2f} m, sotto i "
                      f"{spotGrid.FRONTIER_CLEARANCE_M:.2f} m che il fronte pretende: non e' "
                      f"raggiungibile da nessuna direzione. Ne cerco un altro nella cella "
                      f"({goal_repicks}/{MAX_BLOCK_REPLANS}).")
                tried_goals.append((float(target_x), float(target_y)))
                new_tx, new_ty, _vs, _rs = find_best_point_in_cell(
                    robot_x, robot_y, env, target_row, target_col, pts_up, cells_obs_up,
                    global_sampler, global_map=global_map, exclude=tried_goals)
                if new_tx is not None and new_ty is not None:
                    target_x, target_y = new_tx, new_ty
                    goal_id = prm_graph.add_stop_node(
                        target_x, target_y, global_map, spotGrid.PRM_EDGE_SAFETY_MARGIN_M,
                        label=f"obiettivo riscelto cella ({target_row},{target_col})",
                        is_robot=False)
                    new_ids = prm_graph.find_path_dijkstra(robot_node_id, goal_id)
                    if new_ids is not None:
                        path_waypoints = _waypoints_from_ids(new_ids)
                        full_path_coords = [prm_graph.get_node_position(nid) for nid in new_ids]
                        no_progress = 0
                        print(f"[REPLAN] Verso il nuovo obiettivo ({target_x:.2f}, "
                              f"{target_y:.2f}): {len(path_waypoints)} waypoint.")
                        continue
                    print("[TARGET] Nessun percorso nel grafo verso il nuovo obiettivo.")
                else:
                    print("[TARGET] Nessun altro punto valido in questa cella.")

            # 2026-10-07 -- RIPIANIFICA SUBITO, alla PRIMA cella bloccante, invece di
            # restare fermo tre scansioni sullo stesso arco.
            #
            # Prima: il fronte diceva "ostacolo", il codice stampava "Mantengo il percorso
            # precedente" e riguardava da fermo fino alla conferma. Ma con una soglia secca
            # e il rumore misurato su obstacle_distance (1.5 cm tipici, 7.3 cm di punta,
            # contro un margine di 5 cm) quelle tre scansioni erano tre lanci di moneta
            # sulla stessa scena: nella missione del 2026-10-07 fra la scansione 003 e la
            # 004, robot FERMO e percorso IDENTICO, la stessa cella ha letto 0.296 e poi
            # 0.336 su soglia 0.300, e il verdetto e' passato da "ostacolo, 0.04 m" a
            # "libero, 1.35 m". Aspettare non aggiungeva informazione, aggiungeva rumore.
            #
            # Ora quel rumore e' filtrato (mediana + isteresi in arcVerification), quindi
            # una cella bloccante e' un'informazione da usare: l'arco si scarta e si cerca
            # subito un percorso che passi da un'altra parte. Le scansioni da ferma
            # restano come ultima rete, per dichiarare la cella irraggiungibile.
            if (early_replans < MAX_EARLY_REPLANS
                    and frontier['reason'] in ('ostacolo', 'rugosita', 'pendenza')
                    and path_waypoints):
                poly_now = [(robot_x, robot_y)] + [(x, y) for _, x, y in path_waypoints]
                _, seg_blk = arcVerification.point_along(poly_now, frontier['dist'])
                seg_blk = int(seg_blk)
                ea = robot_node_id if seg_blk == 0 else path_waypoints[seg_blk - 1][0]
                eb = path_waypoints[min(seg_blk, len(path_waypoints) - 1)][0]
                sx_b, sy_b = frontier['stop_xy']

                # Prima di scartare l'ennesimo arco: sono in trappola? (2026-10-07)
                # Nella missione del 2026-10-07 14:5x otto archi uscenti dallo stesso nodo
                # sono stati scartati uno per uno, tutti fermati dallo stesso ostacolo a 9 cm
                # dal robot -- tutti i 25 archi di quel nodo gli passavano accanto. Il blocco
                # era un LUOGO, non un arco. Il ventaglio lo vede in una scansione.
                trapped, fan = is_trapped((robot_x, robot_y), robot_yaw, snap)
                n_us, fan_txt = describe_escape_fan(fan)
                if trapped:
                    MISSION_STATS['traps_detected'] += 1
                    print(f"[TRAPPOLA] Nessuna delle {len(fan)} direzioni del giro e' insieme "
                          f"percorribile e raggiungibile girandosi. Le piu' larghe: {fan_txt}. "
                          f"Non ho dove andare ne' come girarmi: ripercorro il tratto da cui "
                          f"sono venuto, senza scartare altri archi.")
                    if verification_tracker is not None:
                        verification_tracker.clear_path()
                    # Ritorno con GraphNav invece che a zampe all'indietro (2026-10-07): si torna di un
                    # waypoint, si riprova a pianificare da li', e cosi' via fino a
                    # GRAPHNAV_MAX_BACKTRACK_HOPS. La ritirata a zampe resta dentro come ultima rete.
                    esc = escape_by_backtracking(
                        env, prm_graph, goal_id, recordingInterface, robot_state_client,
                        command_client, global_map, mobility_kwargs, pts_up, obstacle_mask_updated,
                        label=f"dal nodo {robot_node_id}")
                    if esc['ok']:
                        path_waypoints = _waypoints_from_ids(esc['ids'])
                        full_path_coords = [prm_graph.get_node_position(nid) for nid in esc['ids']]
                        robot_node_id = esc['robot_node']
                        no_progress = 0
                        early_replans = 0
                        retreats_in_a_row = 0
                        continue
                    return False

                early_replans += 1
                MISSION_STATS['early_replans'] += 1
                print(f"[SUBITO] {frontier['reason']} in ({sx_b:.2f}, {sy_b:.2f}) sul tratto "
                      f"{ea}->{eb}: lo scarto adesso e cerco un altro percorso, senza "
                      f"aspettare la conferma da fermo ({early_replans}/{MAX_EARLY_REPLANS}). "
                      f"Direzioni utilizzabili da qui: {n_us}/{len(fan)}.")
                # Scarto NON permanente: dopo una sola scansione l'arco va tolto dal piano,
                # non condannato per la missione (vedi mark_edge_blocked_soft).
                mark_edge_blocked_soft(prm_graph, ea, eb)
                new_ids = prm_graph.find_path_dijkstra(robot_node_id, goal_id)
                if new_ids is not None and len(new_ids) > 1:
                    path_waypoints = _waypoints_from_ids(new_ids)
                    full_path_coords = [prm_graph.get_node_position(nid) for nid in new_ids]
                    no_progress = 0
                    print(f"[SUBITO] Percorso alternativo con {len(path_waypoints)} waypoint, "
                          f"dal punto in cui sono.")
                    continue
                print("[SUBITO] Il grafo non offre alternative da qui: resto sul percorso e "
                      "lascio decidere il fronte.")

            no_progress += 1
            print(f"[FRONTE] Non posso avanzare nemmeno di {MIN_PARTIAL_STEP_M:.2f} m: riguardo "
                  f"da fermo ({no_progress}/{FRONTIER_BLOCK_CONFIRM_SCANS} scansioni).")
            time.sleep(FRONTIER_RESCAN_WAIT_S)
            continue

        if decision['kind'] == 'blocked':
            k = decision['segment']
            edge_a = robot_node_id if k == 0 else path_waypoints[k - 1][0]
            edge_b = path_waypoints[min(k, len(path_waypoints) - 1)][0]
            sx, sy = frontier['stop_xy']
            print(f"[BLOCCO] Confermato da vicino: {FRONTIER_BLOCK_CONFIRM_SCANS} scansioni consecutive "
                  f"senza poter avanzare. Motivo: {frontier['reason']} in ({sx:.2f}, {sy:.2f}), "
                  f"sul tratto {edge_a}->{edge_b}.")
            MISSION_STATS['frontier_blocks'] += 1

            if mission_folder is not None:
                try:
                    npz_blocked = os.path.join(
                        mission_folder,
                        f"iteration_{iteration}_cell_{target_row}_{target_col}_BLOCKED_{block_replans + 1}.npz")
                    np.savez_compressed(
                        npz_blocked,
                        obstacle_distance=snap.obstacle_dist, rough_veto=snap.rough_veto,
                        seen=snap.seen, terrain=snap.terrain,
                        terrain_raw=np.asarray(terrain_up).reshape((num_y_up, num_x_up)),
                        terrain_valid_raw=np.asarray(valid_up).reshape((num_y_up, num_x_up)),
                        obstacle_mask=np.asarray(obstacle_mask_updated).reshape((num_y_up, num_x_up)),
                        roughness=rough_up_2d,
                        grid_origin_x=snap.origin_x, grid_origin_y=snap.origin_y, cell_size=snap.cell_size,
                        robot_xy=np.array([robot_x, robot_y]), robot_yaw=robot_yaw,
                        polyline=np.array(polyline), frontier_dist=frontier['dist'],
                        frontier_reason=frontier['reason'], frontier_stop_xy=np.array(frontier['stop_xy']),
                        ground_z=np.nan if local_grid.last_ground_filter.get('ground_z') is None
                        else local_grid.last_ground_filter['ground_z'])
                    print(f"[DATA SAVE] Dati del blocco salvati in {os.path.basename(npz_blocked)}")
                except Exception as e:
                    print(f"[DATA SAVE] Impossibile salvare i dati del blocco: {e}")

            # Trappola? (2026-10-07) Se non c'e' nessuna direzione utilizzabile, ripianificare
            # e' inutile per costruzione: il problema non e' l'arco, e' il posto.
            trapped_blk, fan_blk = is_trapped((robot_x, robot_y), robot_yaw, snap)
            n_us_b, fan_txt_b = describe_escape_fan(fan_blk)
            print(f"[BLOCCO] Direzioni utilizzabili da qui: {n_us_b}/{len(fan_blk)} ({fan_txt_b}).")
            if trapped_blk:
                MISSION_STATS['traps_detected'] += 1
                print("[TRAPPOLA] Nessuna via d'uscita girandomi: non scarto altri archi, "
                      "ripercorro il tratto da cui sono venuto.")
                if verification_tracker is not None:
                    verification_tracker.clear_path()
                # Ritorno con GraphNav invece che a zampe all'indietro (2026-10-07): si torna di un
                # waypoint, si riprova a pianificare da li', e cosi' via fino a
                # GRAPHNAV_MAX_BACKTRACK_HOPS. La ritirata a zampe resta dentro come ultima rete.
                esc = escape_by_backtracking(
                    env, prm_graph, goal_id, recordingInterface, robot_state_client,
                    command_client, global_map, mobility_kwargs, pts_up, obstacle_mask_updated,
                    label=f"dal nodo {robot_node_id}")
                if esc['ok']:
                    path_waypoints = _waypoints_from_ids(esc['ids'])
                    full_path_coords = [prm_graph.get_node_position(nid) for nid in esc['ids']]
                    robot_node_id = esc['robot_node']
                    no_progress = 0
                    early_replans = 0
                    retreats_in_a_row = 0
                    continue
                return False

            # Qui il blocco e' confermato da FRONTIER_BLOCK_CONFIRM_SCANS scansioni ferme:
            # l'invalidazione resta permanente (a differenza di [SUBITO], vedi
            # mark_edge_blocked_soft).
            prm_graph.mark_edge_invalid(edge_a, edge_b)
            block_replans += 1
            no_progress = 0

            new_ids = None
            if block_replans <= MAX_BLOCK_REPLANS:
                new_ids = prm_graph.find_path_dijkstra(robot_node_id, goal_id)
            if new_ids is not None:
                path_waypoints = _waypoints_from_ids(new_ids)
                full_path_coords = [prm_graph.get_node_position(nid) for nid in new_ids]
                print(f"[REPLAN] Dopo il blocco ({block_replans}/{MAX_BLOCK_REPLANS}): percorso "
                      f"alternativo con {len(path_waypoints)} waypoint, dal punto in cui sono.")
                continue

            print(f"[BLOCCO] Nessuna alternativa dal punto in cui sono"
                  + (f" (gia' {MAX_BLOCK_REPLANS} ripianificazioni dopo blocchi)."
                     if block_replans > MAX_BLOCK_REPLANS else ".")
                  + " Mi stacco arretrando e rimando la cella.")
            if verification_tracker is not None:
                verification_tracker.clear_path()

            # Ritorno con GraphNav invece che a zampe all'indietro (2026-10-07): si torna di un
            # waypoint, si riprova a pianificare da li', e cosi' via fino a
            # GRAPHNAV_MAX_BACKTRACK_HOPS. La ritirata a zampe resta dentro come ultima rete.
            esc = escape_by_backtracking(
                env, prm_graph, goal_id, recordingInterface, robot_state_client,
                command_client, global_map, mobility_kwargs, pts_up, obstacle_mask_updated,
                label=f"dal nodo {robot_node_id}")
            if esc['ok']:
                path_waypoints = _waypoints_from_ids(esc['ids'])
                full_path_coords = [prm_graph.get_node_position(nid) for nid in esc['ids']]
                robot_node_id = esc['robot_node']
                no_progress = 0
                early_replans = 0
                retreats_in_a_row = 0
                continue

            blocked_save_path = os.path.join(mission_folder,
                                             f"iteration_{iteration}_cell_{target_row}_{target_col}_BLOCKED.png") if mission_folder else None
            visualize_grid_with_candidates(
                pts=pts_up, terrain_real=terrain_real_up, obstacle_mask=obstacle_mask_updated,
                robot_x=robot_x, robot_y=robot_y,
                candidates={'rejected': [], 'valid': []},
                chosen_point=frontier['stop_xy'], iteration=iteration, env=env,
                save_path=blocked_save_path, prm_graph=prm_graph, chosen_path=full_path_coords,
                cells_obstacle_dist=cells_obs_up, valid_values=valid_up,
                grad_values=grad_up, rough_values=rough_up
            )
            return False

        # --- decision['kind'] == 'move' ----------------------------------------------------
        move_x, move_y = decision['target']
        is_partial_step = not decision['full']

        # --- Controllo preventivo sulla posa d'arrivo (2026-10-07) -------------------------
        # Solo per i passi corti o verso spazi stretti: sono quelli con cui si entra negli
        # angoli senza uscita. Nella missione del 2026-10-07 14:5x l'ultimo passo prima della
        # trappola era di 23 cm. Vedi arrival_has_exit.
        step_len = float(np.hypot(move_x - robot_x, move_y - robot_y))
        r_a, c_a, in_a = arcVerification._cells(snap, np.array([move_x]), np.array([move_y]))
        od_arrival = float(snap.obstacle_dist[r_a[0], c_a[0]]) if in_a[0] else float('inf')
        if (arrival_refusals < MAX_ARRIVAL_REFUSALS
                and (step_len <= ARRIVAL_EXIT_CHECK_MAX_STEP_M or od_arrival <= ARRIVAL_EXIT_CHECK_OD_M)):
            heading_arr = float(np.arctan2(move_y - robot_y, move_x - robot_x))
            has_exit, fan_arr = arrival_has_exit(snap, (move_x, move_y), heading_arr)
            if not has_exit:
                arrival_refusals += 1
                MISSION_STATS['arrival_refusals'] += 1
                n_us_a, fan_txt_a = describe_escape_fan(fan_arr)
                print(f"[ARRIVO] Il passo di {step_len:.2f} m verso ({move_x:.2f}, {move_y:.2f}) "
                      f"(obstacle_distance li' {od_arrival:.2f} m) mi porterebbe dove non ci sono "
                      f"uscite: {n_us_a}/{len(fan_arr)} direzioni utilizzabili ({fan_txt_a}). "
                      f"Non ci vado ({arrival_refusals}/{MAX_ARRIVAL_REFUSALS}), scarto il tratto "
                      f"{robot_node_id}->{next_node_id} e ripianifico.")
                mark_edge_blocked_soft(prm_graph, robot_node_id, next_node_id)
                new_ids = prm_graph.find_path_dijkstra(robot_node_id, goal_id)
                if new_ids is not None and len(new_ids) > 1:
                    path_waypoints = _waypoints_from_ids(new_ids)
                    full_path_coords = [prm_graph.get_node_position(nid) for nid in new_ids]
                    no_progress = 0
                    continue
                print("[ARRIVO] Nessuna alternativa nel grafo: ci vado comunque, "
                      "il fronte decide fin dove.")

        no_progress = 0
        retreats_in_a_row = 0   # 2026-10-07: un avanzamento vero rompe la catena di arretramenti
        if is_partial_step:
            print(f"[PASSO PARZIALE] Verso il nodo {next_node_id}: avanzo di {decision['dist']:.2f} m "
                  f"su {np.hypot(next_x - robot_x, next_y - robot_y):.2f} fino a "
                  f"({move_x:.2f}, {move_y:.2f}), poi riguardo con una scansione fresca.")
        else:
            print(f"[OK] Tratto {robot_node_id}->{next_node_id} confermato libero dal fronte. Eseguo movimento...")

        vision_tform_body_current = get_a_tform_b(lg_proto_up.local_grid.transforms_snapshot, VISION_FRAME_NAME,
                                                  BODY_FRAME_NAME)

        # --- Misura della difficolta' di QUESTO segmento (per il gait e per il CSV).
        segment_long_slope, segment_lat_slope, segment_roughness = prm_graph.sample_terrain_between_points(
            robot_x, robot_y, move_x, move_y,
            node_idx1=robot_node_id, node_idx2=next_node_id
        )
        slope_source = prm_graph.last_slope_source
        print(f"[SLOPE-SOURCE] Arco {robot_node_id}->{next_node_id}: "
              f"{str(slope_source).upper()} "
              f"(long={segment_long_slope:.3f}, lat={segment_lat_slope:.3f})")

        seg_diag = _segment_diagnostics(terrain_up_2d, grid_origin_x_up, grid_origin_y_up, cell_size_up,
                                        robot_x, robot_y, move_x, move_y)
        if seg_diag['long_interp'] is not None:
            print(f"[SLOPE-CHECK] Arco {robot_node_id}->{next_node_id}: in uso long={segment_long_slope:.3f} "
                  f"lat={segment_lat_slope:.3f} | interpolata long={seg_diag['long_interp']:.3f} "
                  f"lat={seg_diag['lat_interp']:.3f} | campioni dentro griglia: "
                  f"lungo arco {seg_diag['coverage_along']:.0%}, laterali {seg_diag['coverage_lateral']:.0%}")
        else:
            print(f"[SLOPE-CHECK] Arco {robot_node_id}->{next_node_id}: nessun dato di terreno live "
                  f"per la stima interpolata (fonte pendenza: {str(slope_source).upper()})")

        near_thresh = spotGrid.SLOPE_THRESHOLD * (1.0 - spotGrid.NEAR_THRESHOLD_MARGIN_FRACTION)
        if (segment_long_slope >= near_thresh or segment_lat_slope >= near_thresh) and mission_folder is not None:
            try:
                npz_near = os.path.join(
                    mission_folder,
                    f"iteration_{iteration}_cell_{target_row}_{target_col}_"
                    f"arc{robot_node_id}-{next_node_id}_NEARLIMIT.npz")
                np.savez_compressed(
                    npz_near,
                    obstacle_grid=np.asarray(obstacle_mask_updated).reshape((num_y_up, num_x_up)),
                    terrain_grid=np.asarray(terrain_real_up).reshape((num_y_up, num_x_up)),
                    grid_origin_x=grid_origin_x_up, grid_origin_y=grid_origin_y_up, cell_size=cell_size_up,
                    arc_xy=np.array([robot_x, robot_y, move_x, move_y]),
                    long_slope=segment_long_slope, lat_slope=segment_lat_slope,
                    long_interp=np.nan if seg_diag['long_interp'] is None else seg_diag['long_interp'],
                    lat_interp=np.nan if seg_diag['lat_interp'] is None else seg_diag['lat_interp'],
                    slope_threshold=spotGrid.SLOPE_THRESHOLD)
                MISSION_STATS['near_threshold_saves'] += 1
                margine = spotGrid.SLOPE_THRESHOLD - max(segment_long_slope, segment_lat_slope)
                print(f"[NEAR-LIMIT] Arco {robot_node_id}->{next_node_id} vicino soglia "
                      f"(long={segment_long_slope:.3f}, lat={segment_lat_slope:.3f}, "
                      f"soglia={spotGrid.SLOPE_THRESHOLD:.3f}, margine residuo={margine:+.3f}) "
                      f"-- dati salvati in {os.path.basename(npz_near)}")
            except Exception as e:
                print(f"[DATA SAVE] Impossibile salvare il file vicino soglia: {e}")

        if mobility_kwargs:
            segment_mobility_kwargs = mobility_kwargs
            segment_tier = 'override'
        else:
            segment_gradient = max(segment_long_slope, segment_lat_slope)
            segment_mobility_kwargs = select_mobility_params(segment_gradient, segment_roughness)
            segment_tier = MISSION_STATS['last_tier']

        # --- Condizione di arresto a meta' movimento ---------------------------------
        abort_state = {'reason': None, 'bad': 0}
        move_t0 = time.time()
        live_frontiers = []

        def _should_abort_move():
            if verification_tracker is None:
                return False
            f = verification_tracker.get_frontier()
            if f is None or f.get('path_version') != path_version or f['time'] <= move_t0:
                return False
            frx, fry = f['robot_xy']
            remaining = float(np.hypot(move_x - frx, move_y - fry))
            reach = arcVerification.allowed_advance(f)
            if not live_frontiers or live_frontiers[-1][0] != f['time']:
                live_frontiers.append((f['time'], frx, fry, f['reason'], float(f['dist']), reach, remaining,
                                       abort_state['bad']))
            if reach + FRONTIER_ABORT_TOLERANCE_M < remaining:
                abort_state['bad'] += 1
                if abort_state['bad'] >= FRONTIER_ABORT_CONFIRM:
                    sxa, sya = f['stop_xy']
                    abort_state['reason'] = (f"il fronte si e' accorciato ({f['reason']} in "
                                             f"({sxa:.2f}, {sya:.2f})): raggiungibili {reach:.2f} m "
                                             f"su {remaining:.2f} m mancanti")
                    return True
            else:
                abort_state['bad'] = 0
            return False

        # --- Esecuzione con registrazione dell'inclinazione reale del corpo ---
        tilt_recorder = BodyTiltRecorder(robot_state_client)
        tilt_recorder.start()
        try:
            success_move, distance_traveled, distance_commanded = navigate_to(
                move_x, move_y, robot_x, robot_y, robot_state_client, command_client,
                vision_tform_body_current, should_abort=_should_abort_move,
                keep_heading=keep_heading,
                **segment_mobility_kwargs
            )
        finally:
            tilt_stats = tilt_recorder.stop()
        move_duration = time.time() - move_t0
        _timing_add('movimento', move_duration)
        try:   # solo log: non deve mai interrompere la missione
            _seg_id = [iteration, target_row, target_col, loop_iter, robot_node_id, next_node_id]
            _append_csv("trajectory.csv",
                        ['time', 'iteration', 'cell_row', 'cell_col', 'loop_iter', 'node_from', 'node_to',
                         'x', 'y', 'yaw_deg', 'roll_deg', 'pitch_deg', 'target_x', 'target_y'],
                        [[t] + _seg_id + [x, y, yw, r, pt, move_x, move_y]
                         for (t, x, y, yw, r, pt) in getattr(tilt_recorder, 'samples', [])])
            _append_csv("frontier_live.csv",
                        ['time', 'iteration', 'cell_row', 'cell_col', 'loop_iter', 'node_from', 'node_to',
                         'robot_x', 'robot_y', 'reason', 'frontier_dist', 'reachable_m', 'remaining_m',
                         'consecutive_bad', 'aborted'],
                        [[lf[0]] + _seg_id + list(lf[1:]) + [abort_state['reason'] is not None]
                         for lf in live_frontiers])
        except Exception as e:
            print(f"[LOG] Traiettoria/fronti del movimento non registrati: {e}")

        # --- Riga del CSV per questo segmento (anche se il movimento e' fallito) ---
        try:
            seg_dist = float(np.hypot(move_x - robot_x, move_y - robot_y))
            cost_dist = prm_graph.alpha * seg_dist
            cost_long = prm_graph.beta * segment_long_slope
            cost_lat = prm_graph.beta_lateral * segment_lat_slope
            cost_rough = prm_graph.gamma * segment_roughness
            _log_segment({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'iteration': iteration, 'cell_row': target_row, 'cell_col': target_col,
                'node_from': robot_node_id, 'node_to': next_node_id,
                'x_from': robot_x, 'y_from': robot_y, 'x_to': move_x, 'y_to': move_y,
                'dist_m': seg_dist,
                'slope_source': slope_source,
                'long_slope': segment_long_slope, 'lat_slope': segment_lat_slope,
                'long_interp': seg_diag['long_interp'], 'lat_interp': seg_diag['lat_interp'],
                'roughness': segment_roughness,
                'coverage_along': seg_diag['coverage_along'], 'coverage_lateral': seg_diag['coverage_lateral'],
                'cost_dist': cost_dist, 'cost_long': cost_long, 'cost_lat': cost_lat,
                'cost_rough': cost_rough, 'cost_total': cost_dist + cost_long + cost_lat + cost_rough,
                'tier': segment_tier, 'gait_override': bool(mobility_kwargs),
                'success': bool(success_move),
                'dist_traveled_m': distance_traveled, 'dist_commanded_m': distance_commanded,
                'duration_s': move_duration,
                'roll_start_deg': tilt_stats['roll_start'], 'pitch_start_deg': tilt_stats['pitch_start'],
                'roll_end_deg': tilt_stats['roll_end'], 'pitch_end_deg': tilt_stats['pitch_end'],
                'roll_peak_abs_deg': tilt_stats['roll_peak_abs'], 'pitch_peak_abs_deg': tilt_stats['pitch_peak_abs'],
                'tilt_samples': tilt_stats['samples'], 'tilt_errors': tilt_stats['errors'],
            })
            if tilt_stats['samples']:
                print(f"[TILT] Arco {robot_node_id}->{next_node_id}: picco roll={tilt_stats['roll_peak_abs']:.1f} deg, "
                      f"pitch={tilt_stats['pitch_peak_abs']:.1f} deg ({tilt_stats['samples']} campioni) | "
                      f"stimato long={segment_long_slope:.3f} (~{np.degrees(np.arctan(segment_long_slope)):.1f} deg), "
                      f"lat={segment_lat_slope:.3f} (~{np.degrees(np.arctan(segment_lat_slope)):.1f} deg) | "
                      f"tier={segment_tier}, {move_duration:.1f}s")
        except Exception as e:
            print(f"[LOG] Impossibile registrare il segmento: {e}")

        if success_move:
            print(f"[INFO] Spostamento completato con successo su ({move_x:.2f}, {move_y:.2f})")

            if not hasattr(env, '_traveled_arcs'):
                env._traveled_arcs = []
            env._traveled_arcs.append((robot_x, robot_y, move_x, move_y))

            if is_partial_step:
                ax, ay, _, _ = spotUtils.getPosition(robot_state_client)
                robot_node_id = prm_graph.add_stop_node(
                    ax, ay, global_map, spotGrid.PRM_EDGE_SAFETY_MARGIN_M,
                    came_from=robot_node_id, label="sosta (passo parziale)")
                prm_graph.set_robot_node(robot_node_id, 0.0)
            else:
                prm_graph.traversed_edges.add((min(robot_node_id, next_node_id),
                                               max(robot_node_id, next_node_id)))
                robot_node_id = next_node_id
                prm_graph.set_robot_node(robot_node_id, 0.0)
                path_waypoints.pop(0)

            # Briciola per i ritorni (2026-10-07): un waypoint GraphNav ogni
            # GRAPHNAV_WAYPOINT_MIN_SPACING_M di strada percorsa.
            graphnav_drop_breadcrumb(env, recordingInterface, robot_state_client,
                                     robot_node_id, target_row, target_col)

        elif abort_state['reason'] is not None:
            print(f"[STOP] Movimento verso ({move_x:.2f}, {move_y:.2f}) interrotto dopo "
                  f"{distance_traveled:.2f}m: {abort_state['reason']}.")
            MISSION_STATS['frontier_aborts'] += 1
            if distance_traveled > 0.05:
                if not hasattr(env, '_traveled_arcs'):
                    env._traveled_arcs = []
                ax, ay, _, _ = spotUtils.getPosition(robot_state_client)
                env._traveled_arcs.append((robot_x, robot_y, ax, ay))
                robot_node_id = prm_graph.add_stop_node(
                    ax, ay, global_map, spotGrid.PRM_EDGE_SAFETY_MARGIN_M,
                    came_from=robot_node_id, label="sosta (arresto)")
                prm_graph.set_robot_node(robot_node_id, 0.0)
            continue

        else:
            print(f"[FAIL] Comando di movimento fallito meccanicamente per ({move_x:.2f}, {move_y:.2f}) "
                  f"(percorsi {distance_traveled:.2f}m su {distance_commanded:.2f}m richiesti)")

            _check_and_recover_fall(robot_state_client, command_client, global_map,
                                    "dopo il fallimento del movimento")

            if distance_traveled > 0.05:
                if not hasattr(env, '_traveled_arcs'):
                    env._traveled_arcs = []
                ax, ay, _, _ = spotUtils.getPosition(robot_state_client)
                env._traveled_arcs.append((robot_x, robot_y, ax, ay))

            prm_graph.mark_edge_invalid(robot_node_id, next_node_id)
            print(f"[INFO] Tratto {robot_node_id}-{next_node_id} scartato dopo fallimento fisico.")
            if verification_tracker is not None:
                verification_tracker.clear_path()

            # Ritorno con GraphNav invece che a zampe all'indietro (2026-10-07): si torna di un
            # waypoint, si riprova a pianificare da li', e cosi' via fino a
            # GRAPHNAV_MAX_BACKTRACK_HOPS. La ritirata a zampe resta dentro come ultima rete.
            esc = escape_by_backtracking(
                env, prm_graph, goal_id, recordingInterface, robot_state_client,
                command_client, global_map, mobility_kwargs, pts_up, obstacle_mask_updated,
                label=f"dal nodo {robot_node_id}")
            if esc['ok']:
                path_waypoints = _waypoints_from_ids(esc['ids'])
                full_path_coords = [prm_graph.get_node_position(nid) for nid in esc['ids']]
                robot_node_id = esc['robot_node']
                no_progress = 0
                early_replans = 0
                retreats_in_a_row = 0
                continue

            fail_save_path = os.path.join(mission_folder,
                                          f"iteration_{iteration}_cell_{target_row}_{target_col}_MOVE_FAIL.png") if mission_folder else None
            visualize_grid_with_candidates(
                pts=pts_up, terrain_real=terrain_real_up, obstacle_mask=obstacle_mask_updated,
                robot_x=robot_x, robot_y=robot_y,
                candidates={'rejected': [], 'valid': []},
                chosen_point=(next_x, next_y), iteration=iteration, env=env,
                save_path=fail_save_path, prm_graph=prm_graph, chosen_path=full_path_coords,
                cells_obstacle_dist=cells_obs_up, valid_values=valid_up,
                grad_values=grad_up, rough_values=rough_up
            )
            return False

    if verification_tracker is not None:
        verification_tracker.clear_path()

    # Controllo finale della posizione a fine percorso
    robot_x, robot_y, _, _ = spotUtils.getPosition(robot_state_client)
    return env.is_point_in_cell(robot_x, robot_y, target_row, target_col)

def select_mobility_params(gradient, roughness):
    """
    Three-tier gait selection based on how difficult the terrain is for the SPECIFIC
    segment about to be walked (see prm_graph.PRM.sample_terrain_between_points), instead
    of a single fixed gait profile for the whole mission.

    Classification uses whichever of gradient/roughness is WORSE (i.e. crossing into a
    harder tier on EITHER metric is enough) -- same OR logic as the obstacle veto in
    spotGrid.fuse_obstacle_mask. Thresholds and mu values live in spotGrid.GAIT_* --
    starting points only, retune against how Spot actually behaves on your terrain.

    Tiers:
      PLAIN    -- flat, smooth ground: normal gait (HINT_AUTO), medium swing height,
                  high ground_mu_hint (assume good grip)
      MODERATE -- some slope/roughness, still comfortably walkable: normal gait,
                  high swing height, medium ground_mu_hint
      HARD     -- noticeably difficult but not yet blocked (blocking is a separate,
                  much higher threshold in fuse_obstacle_mask): crawl gait, high swing
                  height, low ground_mu_hint (assume less grip, be conservative)
    """
    # 2026-10-06: tarato per l'ESTERNO. PLAIN aveva SWING_HEIGHT_LOW (passo da pavimento)
    # e nessun livello aveva un limite di velocita'. Anche un prato in piano non e' un
    # pavimento: passo piu' alto di un gradino per tutti i livelli, e velocita' limitate.
    if gradient < spotGrid.GAIT_PLAIN_SLOPE_MAX and roughness < spotGrid.GAIT_PLAIN_ROUGH_MAX:
        tier = "PLAIN"
        params = dict(locomotion_hint=spot_command_pb2.HINT_AUTO,
                      swing_height=spot_command_pb2.SWING_HEIGHT_MEDIUM,
                      ground_mu_hint=spotGrid.GAIT_MU_PLAIN,
                      max_linear_vel=spotGrid.GAIT_VEL_PLAIN,
                      max_angular_vel=spotGrid.GAIT_ANG_PLAIN)
    elif gradient < spotGrid.GAIT_MODERATE_SLOPE_MAX and roughness < spotGrid.GAIT_MODERATE_ROUGH_MAX:
        tier = "MODERATE"
        params = dict(locomotion_hint=spot_command_pb2.HINT_AUTO,
                      swing_height=spot_command_pb2.SWING_HEIGHT_HIGH,
                      ground_mu_hint=spotGrid.GAIT_MU_MODERATE,
                      max_linear_vel=spotGrid.GAIT_VEL_MODERATE,
                      max_angular_vel=spotGrid.GAIT_ANG_MODERATE)
    else:
        tier = "HARD"
        params = dict(locomotion_hint=spot_command_pb2.HINT_CRAWL,
                      swing_height=spot_command_pb2.SWING_HEIGHT_HIGH,
                      ground_mu_hint=spotGrid.GAIT_MU_HARD,
                      max_linear_vel=spotGrid.GAIT_VEL_HARD,
                      max_angular_vel=spotGrid.GAIT_ANG_HARD)

    print(f"[GAIT] Terrain tier: {tier} (gradient={gradient:.3f}, roughness={roughness:.3f}) -> "
          f"locomotion_hint={params['locomotion_hint']}, swing_height={params['swing_height']}, "
          f"ground_mu_hint={params['ground_mu_hint']}, "
          f"vel_max={params['max_linear_vel']:.2f} m/s, rot_max={params['max_angular_vel']:.2f} rad/s")
    MISSION_STATS['gait_tier_counts'][tier] += 1
    MISSION_STATS['last_tier'] = tier
    return params


def navigate_to(target_x, target_y, robot_x, robot_y, robot_state_client, command_client, vision_tform_body,
                 should_abort=None, keep_heading=False, **mobility_kwargs):
    """
    keep_heading: se True il robot NON ruota prima di muoversi, raggiunge il punto con
        l'orientamento che ha (anche di lato o all'indietro).

        2026-10-07: il ciclo di navigazione non lo usa piu' (passa sempre False). Di
        traverso Spot occupa 1.10 m invece di 0.50 e il fronte pretende 0.59 m di
        obstacle_distance invece di 0.30, quindi il ripiego "non ruoto, vado di lato"
        rendeva il robot piu' largo proprio dove lo spazio e' poco: misurato sulle
        missioni del 2026-10-07, sei blocchi su percorsi liberi per 1.3-1.4 m. Il
        parametro resta per l'arretramento (che viaggia lungo l'asse del corpo, non di
        traverso: li' l'ingombro e' quello normale) e per i test.

    mobility_kwargs: optional overrides forwarded to movements.relative_move, e.g.
        locomotion_hint=spot_command_pb2.HINT_CRAWL,
        ground_mu_hint=0.4,
        swing_height=spot_command_pb2.SWING_HEIGHT_HIGH

    Returns:
        (success: bool, distance_traveled: float, distance_commanded: float)

    NOTE: movements.relative_move() returns a (success, distance_traveled) TUPLE.
    Previously this function returned that tuple as-is under the name
    "success_move", and the caller did `if success_move:` -- a non-empty tuple is
    ALWAYS truthy in Python regardless of its contents, so a mechanically FAILED
    move (e.g. (False, 0.0)) was indistinguishable from a real success. That bug
    made it impossible to ever detect the robot getting physically stuck.
    """
    dx, dy = target_x - robot_x, target_y - robot_y
    distance = np.sqrt(dx ** 2 + dy ** 2)
    target_yaw = np.arctan2(dy, dx)

    quat = vision_tform_body.rotation
    current_yaw = np.arctan2(2.0 * (quat.w * quat.z + quat.x * quat.y), 1.0 - 2.0 * (quat.y ** 2 + quat.z ** 2))
    dyaw = np.arctan2(np.sin(target_yaw - current_yaw), np.cos(target_yaw - current_yaw))

    if keep_heading:
        print(f"[INFO] Raggiungo il punto senza ruotare ({distance:.2f} m, "
              f"{np.degrees(dyaw):.0f} gradi rispetto al muso)...")
        move_success, distance_traveled = movements.move_to_world_point_without_turning(
            target_x, target_y, "vision", command_client, robot_state_client,
            should_abort=should_abort, **mobility_kwargs)
    else:
        print("[INFO] Step 1: Rotating to face target...")
        rotate_success, _ = movements.relative_move(0, 0, dyaw, "vision", command_client, robot_state_client,
                                                    should_abort=should_abort, **mobility_kwargs)
        if not rotate_success:
            print(f"[FAIL] Rotation step failed -- aborting before the forward move even started.")
            return False, 0.0, distance

        print(f"[INFO] Step 2: Moving forward {distance:.2f}m...")
        move_success, distance_traveled = movements.relative_move(distance, 0, 0, "vision", command_client,
                                                                   robot_state_client,
                                                                   should_abort=should_abort, **mobility_kwargs)

    # Sanity check: relative_move() can report success (trajectory "settled") while
    # having covered almost none of the requested distance -- e.g. Spot stopping
    # dead against a real physical obstacle, especially since relative_move() sets
    # disable_vision_foot_obstacle_avoidance=True, so Spot has no independent
    # low-level safety net of its own here. A claimed success with near-zero
    # progress on a real move is treated as a genuine failure, not a success.
    MIN_PROGRESS_FRACTION = 0.5
    MIN_PROGRESS_ABS_M = 0.05
    required_progress = max(MIN_PROGRESS_ABS_M, MIN_PROGRESS_FRACTION * distance)

    if move_success and distance > MIN_PROGRESS_ABS_M and distance_traveled < required_progress:
        print(f"[FAIL] Movimento riportato come riuscito ma percorsi solo {distance_traveled:.2f}m "
              f"su {distance:.2f}m richiesti -- trattato come fallimento meccanico reale (probabile "
              f"ostacolo fisico non rilevato).")
        move_success = False

    return move_success, distance_traveled, distance



def find_new_borders(env, robot_row, robot_col, path, frontier):
    new_borders = env.get_adjacent_frontier_cells(robot_row, robot_col, path)
    new_borders_cells = []
    if len(new_borders) != 0:
        for new_border in new_borders:
            if new_border not in frontier and env.is_cell_visited(new_border[0], new_border[1]) != 1:
                new_borders_cells.append(new_border)
    return new_borders_cells


def retreat_along_traveled_path(env, robot_state_client, command_client,
                                max_segments=RETREAT_MAX_SEGMENTS,
                                mobility_kwargs=None,
                                pts=None, obstacle_mask=None):
    """
    Ritirata: torna indietro sui propri passi camminando ALL'INDIETRO, senza mai girarsi.

    Quando il robot si infila in uno spazio stretto e li' non esiste piu' un percorso
    verso l'obiettivo (e' successo nella missione del 2026-10-05 13:12: prima
    [REPLAN FALLITO], poi l'arco successivo BLOCCATO), girarsi non e' una via d'uscita --
    serve spazio che in quel punto non c'e'. Arretrare invece ripercorre il varco da cui
    si e' entrati, che per costruzione era abbastanza largo da passarci. E arretrare lungo
    l'asse del corpo non e' marcia di traverso: l'ingombro resta 0.50 m di larghezza.

    I punti da cui si e' passati sono gia' registrati in env._traveled_arcs, riempito a
    ogni spostamento completato. Si ripercorrono a ritroso, uno per uno, mantenendo
    l'orientamento attuale (vedi movements.move_to_world_point_without_turning).

    Se vengono passati `pts` e `obstacle_mask` della scansione corrente, ogni tratto
    viene prima controllato sui dati freschi: il fatto di esserci passati poco fa e' un
    ottimo indizio, non una garanzia, e non ha senso arretrare dentro qualcosa che nel
    frattempo si vede occupato. In quel caso la ritirata si ferma li' e lo dichiara.

    Returns:
        (segmenti_percorsi: int, distanza_totale: float)
    """
    mobility_kwargs = mobility_kwargs or {}
    traveled = getattr(env, '_traveled_arcs', None)
    if not traveled:
        print("[RITIRATA] Nessun tratto percorso registrato: non so da dove sono arrivato, "
              "non arretro.")
        return 0, 0.0

    done, total = 0, 0.0
    for _ in range(max_segments):
        if not traveled:
            print("[RITIRATA] Esauriti i tratti percorsi registrati.")
            break

        from_x, from_y, to_x, to_y = traveled[-1]
        robot_x, robot_y, _, _ = spotUtils.getPosition(robot_state_client)
        seg_len = float(np.hypot(from_x - robot_x, from_y - robot_y))

        gap = float(np.hypot(to_x - robot_x, to_y - robot_y))
        if gap > RETREAT_CONTINUITY_M:
            print(f"[RITIRATA] Non sono dove finiva l'ultimo tratto registrato (distanza {gap:.2f} m, "
                  f"es. dopo uno spostamento con GraphNav): non arretro alla cieca.")
            del traveled[:]
            break
        if seg_len > RETREAT_MAX_SEGMENT_M:
            print(f"[RITIRATA] Il tratto di rientro e' lungo {seg_len:.2f} m (oltre "
                  f"{RETREAT_MAX_SEGMENT_M:.1f} m): non lo ripercorro all'indietro.")
            break

        if seg_len < RETREAT_MIN_SEGMENT_M:
            # Tratto troppo corto per valere un comando a se': lo consumiamo e proseguiamo.
            traveled.pop()
            continue

        if pts is not None and obstacle_mask is not None:
            try:
                if arcVerification.is_arc_in_fov(robot_x, robot_y, from_x, from_y, pts):
                    status = arcVerification.verify_arc_safety(robot_x, robot_y, from_x, from_y,
                                                               pts, obstacle_mask)
                    if status == 'blocked':
                        print(f"[RITIRATA] Il tratto di rientro verso ({from_x:.2f}, {from_y:.2f}) "
                              f"risulta ostruito nella scansione corrente. Mi fermo qui.")
                        break
            except Exception as e:
                print(f"[RITIRATA] Controllo del tratto di rientro non riuscito ({e}); "
                      f"procedo comunque, e' la via da cui sono venuto.")

        print(f"[RITIRATA] Tratto {done + 1}/{max_segments}: rientro verso "
              f"({from_x:.2f}, {from_y:.2f}), {seg_len:.2f} m all'indietro.")
        success, dist = movements.move_to_world_point_without_turning(
            from_x, from_y, "vision", command_client, robot_state_client, **mobility_kwargs)
        total += dist
        traveled.pop()

        if not success:
            print(f"[RITIRATA] Il rientro si e' fermato dopo {dist:.2f} m. "
                  f"Interrompo la ritirata.")
            break
        done += 1

    if done:
        print(f"[RITIRATA] Completata: {done} tratti, {total:.2f} m percorsi all'indietro.")
    else:
        print("[RITIRATA] Nessun tratto percorso.")
    return done, total


def finalize_or_defer_blocked_cell(env, row, col):
    """
    Call this right after env.mark_cell_side_explored(row, col, side_bit), whenever an
    entry attempt into (row, col) has just failed from one side.

    Only marks the cell permanently blocked (env.mark_cell_blocked -> value=-1) once
    EVERY side that has an in-bounds neighbor has been tried and failed
    (env.all_sides_explored). Otherwise the cell is deliberately left at its unvisited
    value (0) -- it is NOT retried immediately. It simply stays a normal, legitimate
    frontier candidate: get_adjacent_frontier_cells / get_lowest_rank_unexplored_cell /
    get_lowest_rank_from_frontier_list already treat any value-0 cell as fair game,
    ranked by its precomputed serpentine position, exactly like a cell that was never
    attempted at all. That's the whole fix -- a cell that fails once should compete for
    priority through the SAME frontier/rank logic as everything else, and only get
    tried again later (from whichever side hasn't been tried yet) when its turn comes
    up naturally, instead of being eagerly retried out-of-band right away or being
    written off as blocked after a single failed side.
    """
    if env.all_sides_explored(row, col):
        env.mark_cell_blocked(row, col)
        print(f"[BLOCKED] Cell ({row},{col}) permanently blocked -- every reachable side "
              f"has now been tried and failed.")
    else:
        print(f"[DEFER] Cell ({row},{col}) not fully explored yet (sides tried so far: "
              f"{bin(env.get_cell_sides_status(row, col))}) -- leaving it as an open "
              f"frontier candidate; it will be retried later, in normal priority order, "
              f"from a side that hasn't been tried yet.")


def easy_walk(options):
    robot, lease_client, robot_state_client, client_metadata = spotLogInUtils.setLogInfo(options)
    estop = spotLogInUtils.SimpleEstop(robot, options.name + "_estop")

    local_grid = spotGrid.LocalGrid(robot)
    _MISSION_CTX['local_grid'] = local_grid
    global_grid = spotGrid.GlobalGrid()

    recordingInterface = navGraphUtils.RecordingInterface(robot, options.download_filepath, client_metadata)
    recordingInterface.stop_recording()
    recordingInterface.clear_map()

    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        # --- Cartelle di missione e log, create subito cosi' la copia dell'output su file
        # (mission_log.txt) cattura anche tutto l'avvio e la costruzione del grafo.
        mission_timestamp = datetime.now().strftime("Mission_%d-%m-%Y_%H-%M-%S")
        base_graph_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graph")
        graph_folder = os.path.join(base_graph_folder, mission_timestamp)
        mission_map_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MissionMap", mission_timestamp)
        mission_log_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MissionLogs", mission_timestamp)
        os.makedirs(graph_folder, exist_ok=True)
        os.makedirs(mission_map_folder, exist_ok=True)
        os.makedirs(mission_log_folder, exist_ok=True)
        mission_folder = mission_map_folder
        mission_log_path = os.path.join(mission_log_folder, "mission_log.txt")
        _install_stdout_tee(mission_log_path)
        try:
            _MISSION_CTX['log_folder'] = mission_log_folder
            _MISSION_CTX['segment_log'] = SegmentLog(os.path.join(mission_log_folder, "segments.csv"))
            _MISSION_CTX['summary_done'] = False
            print(f"[LOG] Cartella log: {mission_log_folder}  |  dati e immagini: {mission_map_folder}")
            print(f"[CONFIG] SLOPE_THRESHOLD={spotGrid.SLOPE_THRESHOLD}  "
                  f"LATERAL_SLOPE_COST_MULTIPLIER={spotGrid.LATERAL_SLOPE_COST_MULTIPLIER}  "
                  f"OBSTACLE_THRESHOLD={spotGrid.OBSTACLE_THRESHOLD}  ROUGH_THRESHOLD={spotGrid.ROUGH_THRESHOLD}  "
                  f"NEAR_THRESHOLD_MARGIN_FRACTION={spotGrid.NEAR_THRESHOLD_MARGIN_FRACTION}")
            print(f"[CONFIG] PROFILO ARCO: base={spotGrid.SLOPE_BASELINE_M}m  "
                  f"fetta_laterale=+/-{spotGrid.LATERAL_SLICE_HALF_WIDTH_M}m  "
                  f"copertura_minima={spotGrid.MIN_ARC_SAMPLE_FRACTION:.0%}")
            print(f"[CONFIG] GAIT: plain<{spotGrid.GAIT_PLAIN_SLOPE_MAX}/{spotGrid.GAIT_PLAIN_ROUGH_MAX}  "
                  f"moderate<{spotGrid.GAIT_MODERATE_SLOPE_MAX}/{spotGrid.GAIT_MODERATE_ROUGH_MAX}  "
                  f"mu={spotGrid.GAIT_MU_PLAIN}/{spotGrid.GAIT_MU_MODERATE}/{spotGrid.GAIT_MU_HARD}")
            # 2026-10-07: le tre cose cambiate oggi, stampate per poter leggere i log senza
            # dover indovinare con quale versione sono stati prodotti.
            print(f"[CONFIG] TERRENO: terrain_valid >= {spotGrid.TERRAIN_VALID_MIN} "
                  f"(era > 0.0, dipendeva dal segno di un residuo numerico)  "
                  f"celle mai scritte: plateau >= {spotGrid.UNWRITTEN_MIN_CELLS} celle, "
                  f"almeno {spotGrid.UNWRITTEN_MIN_SPOT_INVALID:.0%} marcate non valide da Spot")
            print(f"[CONFIG] ROTAZIONE: prefiltro {spotGrid.ROTATION_CLEARANCE_M:.3f} m "
                  f"(aria {spotGrid.ROTATION_CLEARANCE_AIR_M:.2f} m, separata dai "
                  f"{spotGrid.ROBOT_CLEARANCE_AIR_M:.2f} m dell'avanzamento), poi controllo sul "
                  f"settore spazzato a passi di {spotGrid.ROTATION_SWEEP_STEP_DEG:.0f} gradi; "
                  f"MARCIA DI TRAVERSO DISATTIVATA")
            print(f"[CONFIG] SE NON PUO' RUOTARE: 1) ventaglio nel cono +/-{CONE_SWEEP_MAX_DEG:.0f} gradi "
                  f"a passi di {CONE_SWEEP_STEP_DEG:.0f}, si va nella direzione piu' libera fra quelle "
                  f"che avvicinano di almeno {CONE_MIN_GAIN_M:.2f} m; 2) arretramento dritto fino a "
                  f"{ROTATION_RETREAT_M:.2f} m SOLO se da li' la rotazione diventa possibile, "
                  f"max {MAX_STRAIGHT_RETREATS_IN_A_ROW} consecutivi; 3) si scarta il tratto")
            print(f"[CONFIG] TRAPPOLA: ventaglio a 360 gradi a passi di {ESCAPE_FAN_STEP_DEG:.0f} "
                  f"({int(360/ESCAPE_FAN_STEP_DEG)} direzioni); una direzione vale se concede "
                  f"{ESCAPE_FAN_MIN_ADVANCE_M:.2f} m E il robot puo' girarsi per puntarla. Zero "
                  f"direzioni -> si ripercorre il tratto da cui si e' venuti SUBITO, senza "
                  f"scartare archi. Controllo sulla posa d'arrivo per i passi sotto "
                  f"{ARRIVAL_EXIT_CHECK_MAX_STEP_M:.2f} m o con obstacle_distance sotto "
                  f"{ARRIVAL_EXIT_CHECK_OD_M:.2f} m, max {MAX_ARRIVAL_REFUSALS} rifiuti. "
                  f"Scarto da [SUBITO] e da [ARRIVO]: NON permanente")
            print(f"[CONFIG] RITORNI: briciola GraphNav ogni "
                  f"{GRAPHNAV_WAYPOINT_MIN_SPACING_M:.2f} m, ritorno con GraphNav fino a "
                  f"{GRAPHNAV_MAX_BACKTRACK_HOPS} waypoint indietro riprovando la "
                  f"pianificazione a ogni tappa, poi si rimanda la cella; ritirata a zampe "
                  f"solo come ultima rete. Nodi PRM scartati sotto "
                  f"{PRM_NODE_MIN_CLEARANCE_M:.2f} m di spazio libero, solo dove visti. "
                  f"Deroga di {prm_graph.STOP_NODE_MAP_IGNORE_M:.2f} m alla mappa globale "
                  f"attorno al robot: ora solo dopo una caduta")
            print(f"[CONFIG] OSTACOLI: mediana su {arcVerification.OBSTACLE_MEDIAN_FRAMES} scansioni "
                  f"+ isteresi {arcVerification.OBSTACLE_HYSTERESIS_M:.2f} m attorno a "
                  f"{spotGrid.FRONTIER_CLEARANCE_M:.2f} m (rumore misurato 1.5 cm tipici, 7.3 di punta); "
                  f"celle-ostacolo escluse dal campionamento della pendenza d'arco")
            print(f"[CONFIG] OBIETTIVO: tolleranza d'arrivo {GOAL_REACHED_TOLERANCE_M:.2f} m "
                  f"(= meta' corpo + margine del fronte), riscelta del punto fino a "
                  f"{MAX_BLOCK_REPLANS} volte se risulta irraggiungibile; ripianificazione alla "
                  f"PRIMA cella bloccante, fino a {MAX_EARLY_REPLANS} volte per tentativo")
        except Exception as e:
            print(f"[LOG] Impossibile inizializzare il CSV dei segmenti: {e}")

        command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        robot.time_sync.wait_for_sync()
        robot.logger.info('Powering on robot...')
        robot.power_on()
        assert robot.is_powered_on(), 'Robot power on failed.'
        robot.logger.info('Robot powered on.')
        blocking_stand(command_client)

        # NOTA (2026-10-07): qui c'era un print "[INIT] Initializing Velodyne client...".
        # Il Velodyne NON e' in uso -- griglie e mappe locali vengono tutte dalle telecamere
        # di Spot (l'import di velodyneClient e' commentato in cima al file). La riga
        # faceva solo credere, leggendo i log, che ci fosse un sensore in piu'.
        recordingInterface.clear_map()
        recordingInterface.start_recording()

        recordingInterface.initialize_with_fiducial(robot_state_client, 549)

        start_row, start_col = 0, 0
        recordingInterface.create_default_waypoint(cell_row=start_row, cell_col=start_col)

        # --- Mission Envoirment definition
        # Dimensioni della griglia di missione. Default 5x5 celle da 5 m (esterno). Per una prova
        # al chiuso si puo' ridurre SENZA toccare il codice con la variabile d'ambiente
        #   SPOT_MISSION_GRID="righe,colonne,lato_cella"   es. SPOT_MISSION_GRID="2,3,3"
        grid_rows, grid_cols, grid_cell = 5, 5, 5.0
        _env_grid = os.environ.get("SPOT_MISSION_GRID", "").strip()
        if _env_grid:
            try:
                _r, _c, _cs = _env_grid.split(",")
                grid_rows, grid_cols, grid_cell = int(_r), int(_c), float(_cs)
            except Exception:
                print(f"[CONFIG] SPOT_MISSION_GRID='{_env_grid}' non valido (atteso 'righe,colonne,lato'): "
                      f"uso il default 5,5,5.")
        print(f"[CONFIG] Griglia di missione: {grid_rows} x {grid_cols} celle da {grid_cell:.1f} m")
        env = environmentMap.EnvironmentMap(rows=grid_rows, cols=grid_cols, cell_size=grid_cell)  # cols in front, rows on the left

        # --- Mission-wide movement/gait parameters ---
        # Empty by default: gait/swing-height/ground_mu_hint are now chosen dynamically
        # PER SEGMENT based on sampled terrain difficulty (see select_mobility_params()),
        # instead of one fixed profile for the whole mission. To force a fixed profile
        # instead (e.g. for testing), set mobility_kwargs to a non-empty dict here --
        # attempt_enter_cell_from_position() honors it as an explicit override and skips
        # dynamic selection entirely when it's non-empty. Example fixed override:
        #   mobility_kwargs = dict(
        #       locomotion_hint=spot_command_pb2.HINT_CRAWL,
        #       ground_mu_hint=0.4,
        #       swing_height=spot_command_pb2.SWING_HEIGHT_HIGH
        #   )
        mobility_kwargs = {}

        # --- [FIX CRUCIALE] Ricaviamo la posizione di boot PRIMA di configurare ed elaborare il grafo PRM ---
        x_boot, y_boot, z_boot, quat_boot = spotUtils.getPosition(robot_state_client)
        yaw_boot = np.arctan2(2.0 * (quat_boot.w * quat_boot.z + quat_boot.x * quat_boot.y),
                              1.0 - 2.0 * (quat_boot.y ** 2 + quat_boot.z ** 2))
        env.set_origin(x_boot, y_boot, yaw_boot, start_row=start_row, start_col=start_col)
        _set_mission_z0(robot_state_client, z_boot)

        gb_sampler = global_sampler.GlobalSampler(env, 5)
        gb_sampler.sample_global_grid()

        prm = prm_graph.PRM(min_edge_length=0.5, max_edge_length=2, connection_radius=3)
        _MISSION_CTX['prm'] = prm  # per il riepilogo di fine missione (anche se interrotta)
        prm.add_nodes_from_sampler(gb_sampler)

        # Inseriamo il punto iniziale (Boot Node) dentro la lista dei nodi permanenti prima di generare gli archi
        current_max_id = max(prm.nodes.keys(), default=-1)
        start_node_id = current_max_id + 1
        prm.add_node(start_node_id, x_boot, y_boot)

        # Inseriamo forzatamente tutti i centri geometrici delle celle come nodi del PRM
        current_max_id = start_node_id
        for r in range(env.rows):
            for c in range(env.cols):
                world_pos = env.get_world_position_from_cell(r, c)
                if world_pos is not None:
                    current_max_id += 1
                    prm.add_node(current_max_id, world_pos[0], world_pos[1])

        # NON costruiamo il grafo qui. Questa chiamata era interamente sprecata:
        # a questo punto non esiste ancora nessun dato di terreno (current_terrain_2d
        # e' None, quindi ogni arco cadeva sul fallback statico dei nodi) e
        # global_grid.global_occupancy_map e' ancora None (quindi nessun veto di
        # occupazione veniva applicato). Il grafo prodotto era privo di informazione
        # reale, e attempt_enter_cell_from_position() lo ricostruisce comunque da zero
        # prima di qualunque uso: tra le due build il grafo non viene mai letto da
        # nessuno. Restava solo il costo del ciclo O(N^2) sui nodi.
        # Per ripristinare il comportamento precedente basta rimettere qui:
        #   prm.build_graph(global_map=global_grid.global_occupancy_map,
        #                   edge_safety_margin=spotGrid.PRM_EDGE_SAFETY_MARGIN_M)
        verification_tracker = arcVerification.ArcVerificationTracker(robot)
        _MISSION_CTX['tracker'] = verification_tracker
        verification_tracker.start()

        recordingInterface.set_download_filepath(graph_folder)
        path = env.generate_serpentine_path(start_cell=env.start_cell)

        frontier = []
        visualization_counter = 0

        x, y, z, _ = spotUtils.getPosition(robot_state_client)
        robot_row, robot_col = env.get_cell_from_world(x, y)
        frontier.extend(find_new_borders(env, robot_row, robot_col, path, frontier))

        while (True):
            print(frontier)
            x, y, z, _ = spotUtils.getPosition(robot_state_client)
            robot_row, robot_col = env.get_cell_from_world(x, y)

            borders = env.get_adjacent_frontier_cells(robot_row, robot_col, path)
            borders_in_frontier = [b for b in borders if any((f[0] == b[0] and f[1] == b[1]) for f in frontier)]

            if len(borders_in_frontier) != 0:
                selected_border = min(borders_in_frontier, key=lambda b: b[2])
                # Cella da cui si tenta l'ingresso: serve dopo per segnare il LATO provato.
                # Prima si usava la cella in cui il robot si trovava DOPO il tentativo, che una
                # ritirata o un passo parziale possono aver cambiato (lato sbagliato o nullo).
                origin_row, origin_col = robot_row, robot_col
                check = attempt_enter_cell_from_position(local_grid, global_grid, robot_state_client, command_client, env,
                                                         selected_border[0], selected_border[1], gb_sampler, prm,
                                                         mission_folder,
                                                         visualization_counter, recordingInterface, verification_tracker,
                                                         mobility_kwargs=mobility_kwargs)
                visualization_counter += 1
                frontier.remove(selected_border)

                x_new, y_new, _, _ = spotUtils.getPosition(robot_state_client)
                robot_row, robot_col = env.get_cell_from_world(x_new, y_new)

                if check:
                    env.update_position(x_new, y_new)
                    recordingInterface.create_default_waypoint(cell_row=selected_border[0], cell_col=selected_border[1])
                    env.add_waypoint(x_new, y_new)
                    env.mark_cell_visited(selected_border[0], selected_border[1])
                    frontier.extend(find_new_borders(env, robot_row, robot_col, path, frontier))
                else:
                    if (selected_border[0], selected_border[1]) == (robot_row, robot_col):
                        env.update_position(x_new, y_new)
                        recordingInterface.create_default_waypoint(cell_row=selected_border[0],
                                                                   cell_col=selected_border[1])
                        env.add_waypoint(x_new, y_new)
                        env.mark_cell_visited(selected_border[0], selected_border[1])
                        frontier.extend(find_new_borders(env, robot_row, robot_col, path, frontier))
                    else:
                        # 1. Gli archi scartati nel tentativo fallito (blocchi confermati da vicino,
                        # fallimenti fisici) sono gia' nel PRM tramite mark_edge_invalid e
                        # build_graph li rispetta: Dijkstra non riproporra' la stessa strada.

                        # 2. Ricalcoliamo la posizione attuale (gestisce sia il fallimento in partenza che a metà strada)
                        x_curr, y_curr, _, _ = spotUtils.getPosition(robot_state_client)

                        # 3. Estrapoliamo nuovamente i dati locali per trovare il target point corretto
                        grids_data_rec, main_proto_rec, _ = local_grid.return_local_grid(
                            ['obstacle_distance', 'terrain', 'terrain_valid'], robot_state_client
                        )

                        if grids_data_rec and 'obstacle_distance' in grids_data_rec:
                            pts = grids_data_rec['obstacle_distance']['pts']
                            cells_obstacle_dist = grids_data_rec['obstacle_distance']['values']
                            terrain_values = grids_data_rec['terrain']['values']
                            valid_values = grids_data_rec['terrain_valid']['values']

                            num_x = main_proto_rec.local_grid.extent.num_cells_x
                            num_y = main_proto_rec.local_grid.extent.num_cells_y
                            cell_size = main_proto_rec.local_grid.extent.cell_size

                            terrain_corrected_rec, is_valid_rec = local_grid.correct_terrain(
                                terrain_values, valid_values, num_x, num_y, cell_size=cell_size,
                                unwritten_value=_raw_zero(grids_data_rec), unwritten_scale=_raw_scale(grids_data_rec)
                            )
                            grad_values, rough_values = local_grid.compute_gradient_and_roughness(
                                terrain_corrected_rec, valid_values, num_x, num_y, cell_size,
                                is_valid=is_valid_rec
                            )

                            grad_2d = grad_values.reshape((num_y, num_x))
                            rough_2d = rough_values.reshape((num_y, num_x))
                            terrain_2d_rec = terrain_corrected_rec.reshape((num_y, num_x))

                            transforms_snapshot = main_proto_rec.local_grid.transforms_snapshot
                            grid_frame_name = main_proto_rec.local_grid.frame_name_local_grid_data
                            vision_tform_grid = get_a_tform_b(transforms_snapshot, VISION_FRAME_NAME, grid_frame_name)
                            grid_origin_x = vision_tform_grid.position.x
                            grid_origin_y = vision_tform_grid.position.y

                            prm.update_local_grid_data(
                                grad_values_2d=grad_2d,
                                rough_values_2d=rough_2d,
                                terrain_values_2d=terrain_2d_rec,
                                valid_2d=slope_usable_mask(is_valid_rec, cells_obstacle_dist, (num_y, num_x)),
                                grid_origin_x=grid_origin_x,
                                grid_origin_y=grid_origin_y,
                                cell_size=cell_size
                            )
                            prm.build_graph(global_map=global_grid.global_occupancy_map,
                                            edge_safety_margin=spotGrid.PRM_EDGE_SAFETY_MARGIN_M)

                            target_x, target_y, _, _ = find_best_point_in_cell(
                                x_curr, y_curr, env, selected_border[0], selected_border[1],
                                pts, cells_obstacle_dist, gb_sampler,
                                global_map=global_grid.global_occupancy_map
                            )
                        else:
                            print(
                                "[RECOVERY ERROR] Impossibile recuperare il layer obstacle_distance per il ricalcolo.")
                            target_x, target_y = None, None

                        if target_x is not None and target_y is not None:
                            start_id = prm.add_stop_node(
                                x_curr, y_curr, global_grid.global_occupancy_map,
                                spotGrid.PRM_EDGE_SAFETY_MARGIN_M, label="verifica recupero (robot)")
                            goal_id = prm.add_stop_node(
                                target_x, target_y, global_grid.global_occupancy_map,
                                spotGrid.PRM_EDGE_SAFETY_MARGIN_M, label="verifica recupero (obiettivo)",
                                is_robot=False)
                            prm.set_robot_node(start_id)

                            path_ids = prm.find_path_dijkstra(start_id, goal_id)

                            # Il costo in questo PRM corrisponde al numero di archi (ovvero numero di nodi - 1)
                            COSTO_SOGLIA = 4

                            if path_ids is not None and (len(path_ids) - 1) <= COSTO_SOGLIA:
                                print(
                                    f"[RECOVERY] Trovato percorso alternativo con costo {len(path_ids) - 1} <= {COSTO_SOGLIA}. Riprovo l'ingresso...")

                                retry_check = attempt_enter_cell_from_position(
                                    local_grid, global_grid, robot_state_client, command_client, env,
                                    selected_border[0], selected_border[1], gb_sampler, prm,
                                    mission_folder, visualization_counter, recordingInterface, verification_tracker,
                                    mobility_kwargs=mobility_kwargs
                                )
                                visualization_counter += 1

                                if retry_check:
                                    x_final, y_final, _, _ = spotUtils.getPosition(robot_state_client)
                                    env.update_position(x_final, y_final)
                                    recordingInterface.create_default_waypoint(cell_row=selected_border[0],
                                                                               cell_col=selected_border[1])
                                    env.add_waypoint(x_final, y_final)
                                    env.mark_cell_visited(selected_border[0], selected_border[1])
                                    robot_row_final, robot_col_final = env.get_cell_from_world(x_final, y_final)
                                    frontier.extend(
                                        find_new_borders(env, robot_row_final, robot_col_final, path, frontier))
                                else:
                                    print(
                                        "[FAIL] Anche il percorso alternativo ha fallito. Abbandono la cella e continuo l'algoritmo normale.")
                                    side_bit = env.get_side_bit_facing_origin(origin_row, origin_col, selected_border[0],
                                                                              selected_border[1])
                                    env.mark_cell_side_explored(selected_border[0], selected_border[1], side_bit)
                                    finalize_or_defer_blocked_cell(env, selected_border[0], selected_border[1])
                                    if not env.is_cell_blocked(selected_border[0], selected_border[1]):
                                        frontier.append(selected_border)
                            else:
                                print(
                                    f"[SKIP] Nessun percorso alternativo valido o costo superiore a {COSTO_SOGLIA}. Continuo l'algoritmo normale.")
                                side_bit = env.get_side_bit_facing_origin(origin_row, origin_col, selected_border[0],
                                                                          selected_border[1])
                                env.mark_cell_side_explored(selected_border[0], selected_border[1], side_bit)
                                finalize_or_defer_blocked_cell(env, selected_border[0], selected_border[1])
                                if not env.is_cell_blocked(selected_border[0], selected_border[1]):
                                    frontier.append(selected_border)
                        else:
                            print(
                                "[SKIP] Impossibile trovare un target point valido nella cella. Continuo l'algoritmo normale.")
                            side_bit = env.get_side_bit_facing_origin(origin_row, origin_col, selected_border[0],
                                                                      selected_border[1])
                            env.mark_cell_side_explored(selected_border[0], selected_border[1], side_bit)
                            finalize_or_defer_blocked_cell(env, selected_border[0], selected_border[1])
                            if not env.is_cell_blocked(selected_border[0], selected_border[1]):
                                frontier.append(selected_border)

            else:
                # NOTE: a blocked cell (-1) can never have an unexplored side left, since
                # finalize_or_defer_blocked_cell() only marks a cell blocked once every
                # reachable side has already been tried and failed.
                already_skipped_this_round = set()
                target_row = target_col = rank = nearest_cell = None
                while True:
                    remaining = [c for c in frontier if (c[0], c[1]) not in already_skipped_this_round]
                    lowest_rank_cell = env.get_lowest_rank_from_frontier_list(remaining, path)
                    if lowest_rank_cell is None:
                        break
                    candidate_row, candidate_col, candidate_rank = lowest_rank_cell
                    waypoints_by_cell = recordingInterface.get_all_manual_waypoints_with_cells()
                    candidate_nearest_cell = recordingInterface.find_nearest_waypoint_cell_to_target(
                        (candidate_row, candidate_col), waypoints_by_cell, env)
                    if candidate_nearest_cell is None or \
                            recordingInterface.get_manual_waypoint_by_cell(*candidate_nearest_cell) is None:
                        # Nessuna cella visitata con waypoint da cui partire: prima qui il codice
                        # andava in errore (indice su None) e la missione terminava.
                        print(f"[SKIP-ROUND] Cella ({candidate_row},{candidate_col}): nessun waypoint "
                              f"GraphNav da cui avvicinarsi. Provo la successiva.")
                        already_skipped_this_round.add((candidate_row, candidate_col))
                        continue
                    candidate_side_bit = env.get_side_bit_facing_origin(candidate_nearest_cell[0],
                                                                        candidate_nearest_cell[1],
                                                                        candidate_row, candidate_col)
                    if env.get_cell_sides_status(candidate_row, candidate_col) & candidate_side_bit:
                        print(f"[SKIP-ROUND] Cell ({candidate_row},{candidate_col}) would be approached "
                              f"from the same side already tried and failed -- trying the next-lowest-rank "
                              f"candidate instead for this pass.")
                        already_skipped_this_round.add((candidate_row, candidate_col))
                        continue
                    target_row, target_col, rank = candidate_row, candidate_col, candidate_rank
                    nearest_cell = candidate_nearest_cell
                    break

                if target_row is not None:
                    x_current, y_current, _, _ = spotUtils.getPosition(robot_state_client)

                    recordingInterface.stop_recording()
                    nearest_wp = recordingInterface.get_manual_waypoint_by_cell(nearest_cell[0], nearest_cell[1])
                    navigation_success = _graphnav_navigate_to(
                        recordingInterface, nearest_wp['id'], robot_state_client, command_client,
                        f"verso il waypoint della cella {nearest_cell} per la cella ({target_row},{target_col})")

                    # GraphNav non registra i tratti percorsi: quelli vecchi non descrivono piu'
                    # la strada da cui il robot e' arrivato, la ritirata non deve usarli.
                    env._traveled_arcs = []
                    if _check_and_recover_fall(robot_state_client, command_client,
                                               global_grid.global_occupancy_map,
                                               "durante lo spostamento con GraphNav") is not None:
                        recordingInterface.start_recording()
                        print(f"[CADUTA] Rialzato dopo lo spostamento con GraphNav: segno il lato della "
                              f"cella ({target_row},{target_col}) come provato, per non ripetere la stessa strada.")
                        side_bit = env.get_side_bit_facing_origin(nearest_cell[0], nearest_cell[1],
                                                                  target_row, target_col)
                        env.mark_cell_side_explored(target_row, target_col, side_bit)
                        finalize_or_defer_blocked_cell(env, target_row, target_col)
                        if env.is_cell_blocked(target_row, target_col):
                            frontier.remove((target_row, target_col, rank))
                        continue

                    if navigation_success:
                        recordingInterface.start_recording()
                        check = attempt_enter_cell_from_position(local_grid, global_grid, robot_state_client, command_client,
                                                                 env, target_row, target_col, gb_sampler, prm,
                                                                 mission_folder,
                                                                 visualization_counter, recordingInterface, verification_tracker,
                                                                 mobility_kwargs=mobility_kwargs)
                        visualization_counter += 1

                        if check:
                            x_final, y_final, _, _ = spotUtils.getPosition(robot_state_client)
                            env.update_position(x_final, y_final)
                            recordingInterface.create_default_waypoint(cell_row=target_row, cell_col=target_col)
                            env.add_waypoint(x_final, y_final)
                            env.mark_cell_visited(target_row, target_col)
                            frontier.remove((target_row, target_col, rank))
                            robot_row, robot_col = env.get_cell_from_world(x_final, y_final)
                            frontier.extend(find_new_borders(env, robot_row, robot_col, path, frontier))
                        else:
                            # Mark this side explored; only remove from frontier if that was the
                            # LAST untried side (cell is now truly blocked).
                            side_bit = env.get_side_bit_facing_origin(nearest_cell[0], nearest_cell[1],
                                                                      target_row, target_col)
                            env.mark_cell_side_explored(target_row, target_col, side_bit)
                            finalize_or_defer_blocked_cell(env, target_row, target_col)
                            if env.is_cell_blocked(target_row, target_col):
                                frontier.remove((target_row, target_col, rank))
                    else:
                        # Navigation itself failed to even reach the staging waypoint -- same rule:
                        # only drop from frontier if this was the last untried side.
                        recordingInterface.start_recording()
                        side_bit = env.get_side_bit_facing_origin(nearest_cell[0], nearest_cell[1],
                                                                  target_row, target_col)
                        env.mark_cell_side_explored(target_row, target_col, side_bit)
                        finalize_or_defer_blocked_cell(env, target_row, target_col)
                        if env.is_cell_blocked(target_row, target_col):
                            frontier.remove((target_row, target_col, rank))
                else:
                    if len(frontier) > 0:
                        frontier.remove(frontier[0])

            if len(frontier) == 0:
                break

        env.print_map()
        x_final, y_final, _, _ = spotUtils.getPosition(robot_state_client)
        final_row, final_col = env.get_cell_from_world(x_final, y_final)
        recordingInterface.create_default_waypoint(cell_row=final_row, cell_col=final_col)

        # --- FINAL SCAN: capture the area around the last position before walking away ---
        print("[FINAL SCAN] Acquiring one last look-around before returning to start...")
        x_final, y_final, z_final, quat_final = spotUtils.getPosition(robot_state_client)
        final_yaw = np.arctan2(2.0 * (quat_final.w * quat_final.z + quat_final.x * quat_final.y),
                               1.0 - 2.0 * (quat_final.y ** 2 + quat_final.z ** 2))

        grids_data_final, main_proto_final, _ = local_grid.return_local_grid(
            ['obstacle_distance', 'terrain', 'terrain_valid'], robot_state_client
        )

        if grids_data_final is not None and 'obstacle_distance' in grids_data_final:
            pts_final = grids_data_final['obstacle_distance']['pts']
            cells_obs_final = grids_data_final['obstacle_distance']['values']
            terrain_final = grids_data_final['terrain']['values']
            valid_final = grids_data_final['terrain_valid']['values']

            num_x_final = main_proto_final.local_grid.extent.num_cells_x
            num_y_final = main_proto_final.local_grid.extent.num_cells_y
            cell_size_final = main_proto_final.local_grid.extent.cell_size

            footprint_mask_final = local_grid.compute_robot_footprint_mask(
                pts_final, x_final, y_final, final_yaw, num_x_final, num_y_final
            )

            terrain_real_final, is_valid_final = local_grid.correct_terrain(
                terrain_final, valid_final, num_x_final, num_y_final,
                robot_footprint_mask=footprint_mask_final, cell_size=cell_size_final,
                unwritten_value=_raw_zero(grids_data_final), unwritten_scale=_raw_scale(grids_data_final)
            )
            _log_ground_filter(local_grid, "scan-finale")

            grad_final, rough_final = local_grid.compute_gradient_and_roughness(
                terrain_real_final, valid_final, num_x_final, num_y_final, cell_size_final,
                is_valid=is_valid_final
            )

            obstacle_mask_final = local_grid.fuse_obstacle_mask(
                cells_obstacle_dist=cells_obs_final,
                rough_values=rough_final,
                is_valid=is_valid_final,
                obstacle_threshold=spotGrid.OBSTACLE_THRESHOLD,
                rough_threshold=spotGrid.ROUGH_THRESHOLD
            )

            # Fold this last scan into the persistent global maps, same as every other iteration
            global_grid.global_occupancy_map.update(pts_final,
                                                    _mask_for_global_map(obstacle_mask_final, is_valid_final))
            global_grid.global_terrain_map.update(pts_final, terrain_real_final, is_valid_final)

            final_save_path = os.path.join(
                mission_folder, f"iteration_{visualization_counter}_FINAL_SCAN.png"
            ) if mission_folder else None

            visualize_grid_with_candidates(
                pts=pts_final, terrain_real=terrain_real_final, obstacle_mask=obstacle_mask_final,
                robot_x=x_final, robot_y=y_final,
                candidates={'rejected': [], 'valid': []},
                chosen_point=None, iteration=visualization_counter, env=env,
                save_path=final_save_path, prm_graph=prm, chosen_path=None,
                cells_obstacle_dist=cells_obs_final, valid_values=valid_final,
                grad_values=grad_final, rough_values=rough_final
            )
            visualization_counter += 1
            print("[FINAL SCAN] Done -- global map now includes the final look-around.")
        else:
            print("[FINAL SCAN] WARNING: local grid fetch failed, proceeding without a final scan.")

        # ---------------------------------------------------------------------
        # RIEPILOGO DI MISSIONE -- solo diagnostica, va letto dopo il volo.
        # ---------------------------------------------------------------------
        _emit_mission_summary_once()

        recordingInterface.auto_close_loops(True, False)
        recordingInterface.stop_recording()
        recordingInterface.optimize_anchoring()
        if not _graphnav_return_to_start(recordingInterface, robot_state_client, command_client):
            print("[GRAPHNAV] Ritorno a wp_0 non riuscito: il robot si siede dove si trova.")

        command_client.robot_command(RobotCommandBuilder.synchro_sit_command(), end_time_secs=time.time() + 20)
        sleep(3)
        robot.power_off(cut_immediately=False)
        recordingInterface.download_full_graph()
        estop.stop()


def main():
    options = SimpleNamespace()
    options.name = "easyWalk"
    options.hostname = "192.168.80.3"
    options.verbose = False
    options.recording_user_name = ""
    options.recording_session_name = ""
    options.download_filepath = os.getcwd()

    try:
        easy_walk(options)
        return True
    except movements.RobotFallenError as exc:
        print("\n" + "=" * 70)
        print(f"[CADUTA] MISSIONE INTERROTTA: {exc}")
        print("[CADUTA] Il robot e' a terra e NON ha tentato di tornare alla base.")
        print("[CADUTA] Prendere il controllo dal tablet, verificare il robot, poi rialzarlo a mano.")
        print("=" * 70)
        return False
    except Exception as exc:
        logger = bosdyn.client.util.get_logger()
        logger.error('Hello, Spot! threw an exception: %r', exc)
        try:
            import traceback
            print(f"[LOG] La missione e' terminata con un'eccezione: {exc!r}")
            print(traceback.format_exc())
        except Exception:
            pass
        return False
    finally:
        _emit_mission_summary_once()
        _remove_stdout_tee()


if __name__ == '__main__':
    if not main():
        sys.exit(1)