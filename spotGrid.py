#spotSDK/spotGrid.py


import numpy as np
from scipy.ndimage import uniform_filter, distance_transform_edt
import bosdyn.client
from bosdyn.client.frame_helpers import *
from bosdyn.client.frame_helpers import get_a_tform_b
from bosdyn.api import local_grid_pb2
from bosdyn.client.local_grid import LocalGridClient

import pickle
import time
import os

# =========================================================================
# TERRAIN / OBSTACLE FUSION THRESHOLDS -- single source of truth.
# These are used to decide whether a terrain cell counts as an obstacle in
# fuse_obstacle_mask() below, in arcVerification.py's background arc checker,
# and in every attempt_enter_cell_from_position() call in easy_walk.py.
# Change the values ONLY here -- every caller imports spotGrid and reads
# these directly, so the cell-entry check, the mid-motion arc verifier, and
# the final-scan pass can never drift out of sync with each other again.
# =========================================================================
OBSTACLE_THRESHOLD = 0.15   # obstacle_distance (m): distance <= this -> blocked
SLOPE_THRESHOLD = 0.6       # gradient (tan of slope): tan(28deg)=~0.53, tan(25deg)=~0.46
ROUGH_THRESHOLD = 0.15      # local roughness (std-dev of terrain height, m)
STEP_THRESHOLD = 0.40       # matches Spot's own >40cm obstacle-platform convention

# =========================================================================
# VALIDITA' DEL TERRENO (2026-10-07). Il layer `terrain_valid` di Spot contiene 0 e 1,
# ma unpack_grid gli applica cell_value_scale e cell_value_offset come a qualunque altro
# layer: lo 0 diventa +/-5.5e-17 e IL SEGNO CAMBIA da una scansione all'altra. Il test
# `> 0.0` risultava quindi vero o falso a caso. Misurato sulle 16 scansioni registrate il
# 2026-10-07: in 10 OGNI cella risultava valida (100%), nelle altre 6 il conteggio era
# corretto (82-87%). Conseguenza: le celle che Spot dichiara NON attendibili -- comprese
# quelle che i sensori non hanno mai scritto -- entravano nel terreno come dati buoni, e
# la loro deviazione standard produceva falsi ostacoli: in media 276 veti di sola
# rugosita' per scansione, fino a 1138. In una scansione il robot risultava in piedi su
# una cella marcata ostacolo.
#
# Si confronta con 0.5: i valori veri sono 0 e 1, qualunque soglia in mezzo va bene, e
# 0.5 e' insensibile al residuo numerico in entrambi i segni.
# =========================================================================
TERRAIN_VALID_MIN = 0.5

# =========================================================================
# FILTRO SULL'ALTEZZA DEL SUOLO (2026-10-06) -- vedi LocalGrid.correct_terrain().
#
# Il layer `terrain` di Spot e' espresso rispetto al riferimento fissato
# all'ACCENSIONE. Le celle che i sensori non scrivono restano al valore iniziale
# (~0) e Spot le dichiara comunque valide. Nella missione del 2026-10-05 16:05 il
# robot era stato acceso un piano piu' su: quelle celle stavano 3.226 m sopra il
# suolo reale, e la loro rugosita' (1.585 contro soglia 0.15) ha prodotto il 100%
# degli ostacoli che hanno fatto scattare i blocchi.
#
# La regola non guarda lo zero (fragile) ma la distanza dal suolo SU CUI IL ROBOT
# STA IN PIEDI, stimato come mediana delle celle valide entro GROUND_REF_RADIUS_M
# dal robot. Cosi' non importa piu' dove il robot e' stato acceso.
#
# ASIMMETRICA, di proposito:
#   - SOPRA il suolo oltre la soglia: non e' superficie calpestabile (muro, albero,
#     cella mai scritta). Esclusa da pendenza e rugosita' e riempita con il suolo
#     vicino. L'eventuale ostacolo resta rilevato da obstacle_distance, che e' un
#     layer separato e non viene toccato.
#   - SOTTO il suolo oltre la soglia: puo' essere un VUOTO VERO (muretto su una
#     strada, scala che scende, fosso). NON viene toccata: continua a generare veti
#     come prima. Riempirla con la quota del suolo nasconderebbe un dirupo.
#
# 1.2 m: la finestra arriva a ~1.9 m dal robot; 1.2 m di dislivello a quella
# distanza sono gia' ~32 gradi, oltre il limite di Spot. Nessuna salita che il
# robot potrebbe davvero affrontare viene scartata.
#
# NOTA (2026-10-07): questo filtro NON copre le celle mai scritte nel caso normale.
# Quelle stanno allo zero verticale della griglia locale, che con il robot in piedi
# cade ~0.93 m sopra il pavimento -- sotto 1.2 m, quindi il filtro non le vede. Il
# caso del 2026-10-05 (3.226 m) era l'eccezione, non la regola. Se ne occupa
# _cells_never_written, che dal 2026-10-07 funziona davvero.
# =========================================================================
GROUND_HEIGHT_TOLERANCE_M = 1.2
GROUND_REF_RADIUS_M = 0.7            # celle sotto/attorno al robot usate per stimare il suolo
GROUND_REF_FALLBACK_RADIUS_M = 1.2   # se sotto il robot ci sono troppe poche celle valide
GROUND_REF_MIN_CELLS = 50

# -------------------------------------------------------------------------
# CELLE MAI SCRITTE -- vedi LocalGrid._cells_never_written().
#
# Le celle che i sensori non hanno mai scritto restano tutte alla STESSA quota
# esatta. Spot le dichiara quasi sempre non valide (misurato il 2026-10-07:
# 98.2-100%), quindi con TERRAIN_VALID_MIN corretto sono gia' in gran parte
# gestite; questa regola e' la rete per il resto.
#
# RISCRITTA il 2026-10-07. La versione precedente PREVEDEVA la quota, uguale a
# cell_value_offset ("il valore grezzo 0"), e non e' mai scattata: zero celle in
# tutte e 16 le scansioni registrate, mentre il plateau era di 1845-3710 celle
# (11-23% della griglia). Due errori indipendenti, entrambi fatali -- vedi la
# docstring di _cells_never_written. Ora la quota si ricava dai dati.
# -------------------------------------------------------------------------
UNWRITTEN_MATCH_FRACTION = 0.25    # tolleranza: 0.25 * passo di quantizzazione
UNWRITTEN_MATCH_FALLBACK_M = 1e-4  # se il passo non e' noto
UNWRITTEN_MIN_CELLS = 100          # celle che devono condividere la quota perche' sia un plateau
UNWRITTEN_MIN_SPOT_INVALID = 0.80  # frazione del plateau che Spot deve marcare non valida

# =========================================================================
# FRONTE SICURO (2026-10-06) -- vedi arcVerification.compute_safe_frontier().
#
# Sostituisce la verifica "arco per arco". A ogni scansione si percorre il
# cammino pianificato a partire dal robot e ci si ferma al primo punto non
# confermato libero: ostacolo, rugosita', pendenza, oppure fine dei dati. Il
# robot non supera mai quel punto. Niente verdetti ricordati: si ricalcola
# tutto da zero a ogni scansione.
#
# Il robot NON e' piu' un punto:
#   FRONTIER_CLEARANCE_M: distanza minima da un ostacolo per la LINEA CENTRALE
#     del percorso. Spot e' largo 0.50 m: con la vecchia soglia di 0.15 m il
#     fianco (a 0.25 m dal centro) poteva sovrapporsi a un ostacolo di 10 cm.
#     0.25 di meta' larghezza + 0.05 di aria. Si legge direttamente da
#     obstacle_distance, che e' gia' "distanza dall'ostacolo piu' vicino".
#   FRONTIER_BODY_HALF_LENGTH_M: il muso sta 0.55 m davanti al centro. Il
#     centro si ferma quindi a meta' lunghezza + FRONTIER_MARGIN_M prima del
#     primo punto non libero, cosi' anche il muso resta su terreno confermato.
#   FRONTIER_MIN_CLEARANCE_M / FRONTIER_ESCAPE_DIST_M: se il robot si trova GIA'
#     piu' vicino di FRONTIER_CLEARANCE_M a qualcosa (spazio stretto), nel primo
#     metro si accetta di non avvicinarsi piu' di quanto gia' sia, cosi' puo'
#     uscirne. Mai sotto FRONTIER_MIN_CLEARANCE_M. E' esattamente la situazione
#     della missione del 2026-10-05 13:12, in cui il robot non riusciva piu' a
#     uscire da uno spazio stretto.
#   FRONTIER_ROBOT_RADIUS_M: dentro questo raggio dal robot le celle "non viste"
#     non fermano il fronte -- e' il terreno su cui il robot sta in piedi, e al
#     primo scan risulta non valido perche' coperto dal corpo.
# =========================================================================
ROBOT_HALF_LENGTH_M = 0.55          # Spot: 1.10 x 0.50 m
ROBOT_HALF_WIDTH_M = 0.25
ROBOT_CLEARANCE_AIR_M = 0.05        # aria lasciata fra corpo e ostacolo, ovunque
FRONTIER_CLEARANCE_M = ROBOT_HALF_WIDTH_M + ROBOT_CLEARANCE_AIR_M      # = 0.30
FRONTIER_MIN_CLEARANCE_M = 0.05
FRONTIER_ESCAPE_DIST_M = 1.0
FRONTIER_BODY_HALF_LENGTH_M = ROBOT_HALF_LENGTH_M                       # = 0.55
FRONTIER_MARGIN_M = 0.10

# =========================================================================
# ROTAZIONE SUL POSTO (2026-10-06, passo 4) -- vedi easy_walk.attempt_enter_cell...
# Ruotando sul posto gli spigoli del corpo descrivono un cerchio di raggio pari
# alla semidiagonale, sqrt(0.55^2 + 0.25^2) = 0.60 m. Prima nessuno controllava
# che quel cerchio fosse libero. Se attorno al robot c'e' meno di questo spazio,
# il robot NON ruota: raggiunge il punto camminando con l'orientamento che ha,
# e il fronte viene ricalcolato con l'ingombro vero del corpo rispetto alla
# direzione di marcia (di traverso Spot occupa 0.55 m per lato, non 0.25).
# Sotto ROTATION_CHECK_MIN_DYAW_DEG la rotazione e' cosi' piccola che lo spazio
# spazzato e' trascurabile e non si controlla.
# =========================================================================
# Aria sulla ROTAZIONE, separata da quella sull'avanzamento (2026-10-07, scelta esplicita).
# Il costo dell'errore non e' simmetrico: un falso "non posso avanzare" costa una scansione
# persa, un falso "non posso ruotare" e' costato 3.5 m di arretramento in catena nella
# missione del 2026-10-07. Inoltre il margine di 0.05 m e' PIU' PICCOLO del rumore misurato
# su obstacle_distance (1.5 cm tipici, 7.3 cm di punta), quindi su quella soglia non stava
# proteggendo niente: stava decidendo il rumore. A zero, la rotazione si rifiuta solo se il
# corpo compenetra davvero, e chi avanza dopo la rotazione lo verifica col fronte sicuro.
#
# Misurato sugli 11 momenti di rifiuto reali della missione del 2026-10-07 14:0x:
# aria 0.05 -> 2 rotazioni concesse, aria 0.02 -> 2, aria 0.00 -> 3. Cioe' questo cambio
# da solo sblocca UN caso su undici: gli altri rifiuti sono veri (il corpo compenetra da
# 4 a 19 cm anche senza margine, in corridoi da ~60 cm di corsia libera). Sta scritto qui
# perche' non venga ricordato come la correzione del problema: non lo e'.
ROTATION_CLEARANCE_AIR_M = 0.0
ROTATION_CLEARANCE_M = float(np.hypot(ROBOT_HALF_LENGTH_M, ROBOT_HALF_WIDTH_M)) + ROTATION_CLEARANCE_AIR_M  # ~0.604
ROTATION_CHECK_MIN_DYAW_DEG = 10.0


def body_extents(heading, motion_dir):
    """
    Ingombro del corpo rispetto alla direzione di marcia: (meta' lungo la marcia,
    meta' di traverso). heading = orientamento del corpo, motion_dir = direzione del
    moto, in radianti. Con heading == motion_dir: (0.55, 0.25); di lato: (0.25, 0.55).
    """
    a = motion_dir - heading
    c, s = abs(np.cos(a)), abs(np.sin(a))
    along = ROBOT_HALF_LENGTH_M * c + ROBOT_HALF_WIDTH_M * s
    across = ROBOT_HALF_LENGTH_M * s + ROBOT_HALF_WIDTH_M * c
    return float(along), float(across)
FRONTIER_SAMPLE_STEP_M = 0.03        # un campione per cella
FRONTIER_ROBOT_RADIUS_M = 0.60
FRONTIER_SLOPE_MIN_LENGTH_M = 0.50   # sotto questa lunghezza la pendenza non e' stimabile

# Passo angolare con cui si campiona la rotazione in rotation_is_clear(). 5 gradi: a
# raggio 0.60 m sono 5 cm di arco, cioe' meno di due celle, quindi fra due pose campionate
# non puo' nascondersi un ostacolo che il corpo attraverserebbe.
ROTATION_SWEEP_STEP_DEG = 5.0


def rotation_is_clear(obstacle_dist, origin_x, origin_y, cell_size, robot_x, robot_y,
                      robot_yaw, dyaw, half_along=ROBOT_HALF_LENGTH_M,
                      half_across=ROBOT_HALF_WIDTH_M, air=ROTATION_CLEARANCE_AIR_M,
                      step_deg=ROTATION_SWEEP_STEP_DEG):
    """
    Il robot puo' ruotare sul posto di `dyaw` radianti senza toccare niente?

    Controlla le celle che il corpo spazza DAVVERO, invece di chiedere che sia libero
    tutto il cerchio circoscritto. ROTATION_CLEARANCE_M = 0.65 risponde alla domanda
    "e' libero il cerchio di raggio 0.60 attorno al robot?", cioe' il caso peggiore su
    TUTTE le direzioni e TUTTI gli angoli; ruotando di 39 gradi il corpo spazza invece
    un settore, e gli ostacoli stanno dove stanno.

    Misurato sulla missione del 2026-10-07 (iterazione 2, cella (1,0)): il controllo
    scalare ha rifiutato 9 rotazioni su 10, e di quelle 9 il controllo sul settore dice
    che 8 erano sicure. In tre casi (scansioni 8, 9, 10) il rifiuto ha prodotto un blocco
    su un percorso che col muso in avanti era libero per 1.39 m: serviva ruotare di 39
    gradi e li' se ne potevano fare 68.

    Il rettangolo e' GIA' gonfiato dell'aria (`air`), quindi "margine >= 0" significa che
    il corpo vero passa con almeno `air` metri di gioco. Si lavora su una finestra
    ritagliata attorno al robot: costo misurato 0.26 ms per un controllo normale,
    0.90 ms per una rotazione di 180 gradi.

    Args:
        obstacle_dist: griglia 2D (num_y, num_x) di obstacle_distance, in metri.
        origin_x, origin_y, cell_size: geometria della griglia nel frame VISION.
        robot_x, robot_y, robot_yaw: posa attuale del robot.
        dyaw: rotazione richiesta, in radianti (segno compreso).
        half_along, half_across: meta' ingombro del corpo (default Spot: 0.55 x 0.25).
        air: aria lasciata attorno al corpo.
        step_deg: passo angolare di campionamento.

    Returns:
        (libero: bool, margine: float) -- margine e' il minimo di obstacle_distance sulle
        celle spazzate, oppure +inf se la finestra cade fuori dalla griglia (nel qual caso
        non sappiamo nulla e si risponde True, come fa il resto del codice per i dati
        mancanti: a decidere restera' il fronte sicuro).
    """
    if obstacle_dist is None or cell_size is None:
        return True, float('inf')
    reach = float(np.hypot(half_along, half_across)) + air
    num_rows, num_cols = obstacle_dist.shape
    c0 = max(int(np.floor((robot_x - reach - origin_x) / cell_size)), 0)
    c1 = min(int(np.ceil((robot_x + reach - origin_x) / cell_size)), num_cols - 1)
    r0 = max(int(np.floor((robot_y - reach - origin_y) / cell_size)), 0)
    r1 = min(int(np.ceil((robot_y + reach - origin_y) / cell_size)), num_rows - 1)
    if c1 <= c0 or r1 <= r0:
        return True, float('inf')

    rows, cols = np.mgrid[r0:r1 + 1, c0:c1 + 1]
    dx = origin_x + (cols + 0.5) * cell_size - robot_x
    dy = origin_y + (rows + 0.5) * cell_size - robot_y
    window = obstacle_dist[r0:r1 + 1, c0:c1 + 1]

    num_steps = max(2, int(abs(np.degrees(dyaw)) / step_deg) + 2)
    worst = float('inf')
    for t in np.linspace(0.0, float(dyaw), num_steps):
        a = robot_yaw + t
        ca, sa = np.cos(a), np.sin(a)
        in_body = ((np.abs(dx * ca + dy * sa) <= half_along + air) &
                   (np.abs(-dx * sa + dy * ca) <= half_across + air))
        if in_body.any():
            worst = min(worst, float(window[in_body].min()))
    return worst >= 0.0, worst

# Margine con cui il PRM scarta gli archi che passano vicino a celle occupate nella mappa
# globale. Le celle occupate sono quelle con obstacle_distance <= OBSTACLE_THRESHOLD,
# quindi un arco ammesso passa ad almeno OBSTACLE_THRESHOLD + margine dall'ostacolo.
# Deve dare la STESSA distanza che pretende il fronte, altrimenti il pianificatore
# disegna percorsi che il fronte poi rifiuta, e il robot si blocca su piani "validi".
# Era 0.05 (cioe' 0.20 m dall'ostacolo).
PRM_EDGE_SAFETY_MARGIN_M = FRONTIER_CLEARANCE_M - OBSTACLE_THRESHOLD   # = 0.15

# =========================================================================
# MAPPA GLOBALE DEGLI OSTACOLI (2026-10-06) -- vedi GlobalGrid.update().
# Una cella diventa occupata solo dopo GLOBAL_OCC_CONFIRM_OBS osservazioni
# consecutive concordi. Prima bastava UNA sola osservazione, partendo da
# qualunque stato -- anche da "vista libera quattro volte". Con il fronte
# sicuro questa mappa e' cio' che ricorda gli ostacoli fra un tentativo e
# l'altro, quindi non puo' credere a una scansione isolata.
# =========================================================================
GLOBAL_OCC_CONFIRM_OBS = 2
GLOBAL_OCC_MAX_COUNT = 4
GLOBAL_OCC_COARSE_CELLS = 8          # lato del blocco dell'indice a grana grossa (8 x 3 cm = 24 cm)

# =========================================================================
# GAIT SELECTION THRESHOLDS -- three-tier terrain difficulty, well below the
# OBSTACLE thresholds above (those mean "basically impassable"; these mean
# "how carefully should Spot walk through terrain we've already decided is
# passable"). A segment's 90th-percentile gradient/roughness (see
# prm_graph.PRM.sample_terrain_between_points) is classified PLAIN /
# MODERATE / HARD using the WORSE (max) of the two metrics -- either one
# alone can push a segment into a more cautious tier, same OR logic as the
# obstacle veto in fuse_obstacle_mask.
# Starting points only -- retune against how Spot actually behaves on your
# terrain; nothing here has been validated on hardware.
# =========================================================================
GAIT_PLAIN_SLOPE_MAX = 0.15     # below this on BOTH -> PLAIN tier (flat ground)
GAIT_PLAIN_ROUGH_MAX = 0.05
GAIT_MODERATE_SLOPE_MAX = 0.35  # below this on BOTH -> MODERATE tier; at/above -> HARD tier
GAIT_MODERATE_ROUGH_MAX = 0.10

# ground_mu_hint per tier (bosdyn.api.spot.TerrainParams): BD's own docs suggest 0.4-0.8,
# 0.8 being the robot's default (assume good grip); lower values tell Spot to expect less
# friction and plan foot placement more conservatively.
GAIT_MU_PLAIN = 0.8
GAIT_MU_MODERATE = 0.6
GAIT_MU_HARD = 0.4

# Velocita' massime per livello (m/s lineare, rad/s rotazione) -- 2026-10-06.
# Prima non esisteva alcun limite: Spot camminava fino al proprio massimo (~1.6 m/s).
# Valori pensati per l'ESTERNO, che e' il campo d'uso: anche un prato "in piano" non e'
# un pavimento. Vedi movements.DEFAULT_MAX_LINEAR_VEL_MPS per l'avvertenza su min/max.
GAIT_VEL_PLAIN = 0.5
GAIT_VEL_MODERATE = 0.35
GAIT_VEL_HARD = 0.25
GAIT_ANG_PLAIN = 0.6
GAIT_ANG_MODERATE = 0.5
GAIT_ANG_HARD = 0.4

# =========================================================================
# PRM EDGE-COST DIRECTIONAL WEIGHTING -- quanto deve costare IN PIU', nella
# funzione di costo usata da Dijkstra per scegliere tra archi gia' ammessi,
# un traverso laterale (sidehill/rollio) rispetto a una salita/discesa
# longitudinale (beccheggio) della STESSA grandezza. Un arco sotto soglia
# non viene mai scartato solo per questo -- il veto vero (arco ammesso o
# no) resta SLOPE_THRESHOLD in entrambe le direzioni, applicato in
# build_graph()/refresh_local_edge_weights()/arcVerification.py, del tutto
# indipendente da questo moltiplicatore. Questo serve solo a far preferire,
# tra due archi ENTRAMBI ammessi, quello meno esposto al rollio -- un
# quadrupede tollera peggio l'inclinazione laterale di quella in avanti a
# parita' di pendenza. Punto di partenza, da ritarare su hardware reale.
#
# VALORE PER IL PRIMO TEST: 1.0 (neutro -- laterale e longitudinale pesano
# uguale nel costo, la direzione resta comunque distinta dal vecchio campo
# isotropo). Era 1.5, ma era una supposizione non osservata su Spot, e la
# stima di pendenza per arco sovrastimava le componenti piccole fino a ~1.9x
# (vedi CONTEXT.md, punto A dei problemi aperti): moltiplicarla ancora per 1.5
# avrebbe amplificato l'errore. Con il profilo d'arco corretto (base fisica +
# campionamento NaN-aware, vedi compute_arc_slope_profile) quella sovrastima e'
# rientrata, quindi un peso > 1.0 e' ora valutabile sui dati reali; il CSV per
# segmento registra i termini di costo.
# =========================================================================
LATERAL_SLOPE_COST_MULTIPLIER = 1.0

# =========================================================================
# SOGLIA DIAGNOSTICA -- quanto vicino a SLOPE_THRESHOLD conta come "quasi al
# limite" per i log/salvataggi di debug (flip del veto tra due refresh, .npy
# dei casi limite in easy_walk.py). Puramente informativa: non influisce su
# NESSUNA decisione del robot, solo su cosa finisce nei log per la revisione
# post-missione.
# =========================================================================
NEAR_THRESHOLD_MARGIN_FRACTION = 0.15

# =========================================================================
# PROFILO DI PENDENZA PER ARCO -- parametri fisici (vedi compute_arc_slope_profile).
#
# SLOPE_BASELINE_M: la stessa base fisica gia' usata da
#   compute_gradient_and_roughness() per il campo di gradiente per-cella. Il profilo
#   d'arco usava invece il passo nativo della cella (~3 cm): su quella base la faccia
#   verticale di un armadio da 80 cm produce 0.80 / (2 * 0.03) = 13 m/m, e un'anta da
#   1.45 m produce 24 -- esattamente i massimi comparsi nei log delle missioni del
#   2026-10-05. Misurando il dislivello su ~25 cm (l'ordine di un passo di Spot) un
#   gradino isolato viene diluito, mentre una salita vera e prolungata resta intatta.
#   E' la stessa correzione gia' applicata al campo per-cella, che non era mai stata
#   estesa al profilo per-arco.
#   PRECISAZIONE (revisione del 2026-10-06 sera): la differenza e' CENTRATA, k celle avanti
#   e k indietro con k = round(0.25 / 0.03) = 8, quindi il dislivello si misura su 2k = 16
#   celle, cioe' ~0.48 m, non 0.25. Vale identico per il PRM, il fronte e il campo per-cella,
#   quindi sono coerenti fra loro; ma un gradino isolato e' diluito il doppio di quanto
#   dicevano i commenti: un gradino di 25 cm da' ~0.52 di pendenza (sotto la soglia 0.6) e
#   ~0.125 di rugosita' (sotto 0.15). Spot sale gradini fino a ~30 cm, quindi non e' un
#   rischio in se'; e' da ricordare se si ritarano le soglie. Non cambiato.
#
# LATERAL_SLICE_HALF_WIDTH_M: meta' larghezza della fetta perpendicolare usata per il
#   rollio. Prima la fetta era lunga quanto l'ARCO (+/- dist/2), quindi un arco da 2 m
#   raccoglieva terreno fino a 1 m per lato: qualunque mobile entro un metro generava
#   una "pendenza laterale" enorme e vetava l'arco, e la portata del veto cambiava con
#   la lunghezza del passo (lo stesso punto risultava percorribile con un arco da 1.5 m
#   e vietato con uno da 2 m). Il rollio e' per definizione l'inclinazione TRA LE
#   TRACCE DEI PIEDI: oltre quella fascia non si misura rollio, si misura paesaggio.
#   0.30 m deriva dai 0.50 m di larghezza di Spot (vedi compute_robot_footprint_mask)
#   piu' margine per la variabilita' di appoggio.
#
# MIN_ARC_SAMPLE_FRACTION: frazione minima di campioni che deve cadere davvero dentro
#   la griglia perche' has_data sia True (vedi compute_arc_slope_profile).
# =========================================================================
SLOPE_BASELINE_M = 0.25
LATERAL_SLICE_HALF_WIDTH_M = 0.30
MIN_ARC_SAMPLE_FRACTION = 0.5


def fill_invalid_nearest(values_2d, invalid_mask):
    """
    Extend each valid cell's value into the nearest invalid (masked) cells.

    Used to patch both sensor-invalid terrain cells (BD's own terrain_valid grid
    flags a cell invalid when its height estimate is too extreme to trust) and
    the robot's own footprint (self-occluded, never a real reading). In both
    cases the honest fix is the same: borrow the nearest trustworthy neighbor's
    value rather than leaving raw/garbage data or a NaN hole in the terrain layer.

    Args:
        values_2d: 2D array of raw values (e.g. terrain height)
        invalid_mask: 2D bool array, True where values_2d should be considered
                      untrustworthy and filled in from the nearest False cell

    Returns:
        2D array, same shape as values_2d, with invalid_mask cells replaced by
        their nearest valid neighbor's value. If everything is valid, or
        everything is invalid, returns values_2d unchanged (nothing sensible to
        borrow from in the second case).
    """
    if not np.any(invalid_mask) or not np.any(~invalid_mask):
        return values_2d.copy()
    nearest_indices = distance_transform_edt(invalid_mask, return_distances=False, return_indices=True)
    return values_2d[tuple(nearest_indices)]


def _gather_heights(terrain_2d, grid_origin_x, grid_origin_y, cell_size, xs, ys, valid_2d=None):
    """
    Campionamento vettoriale delle quote da una griglia densa.

    Restituisce (quote, dentro_griglia): le posizioni fuori griglia valgono NaN e
    NON vengono compattate via. Prima venivano semplicemente scartate dalla lista
    dei campioni, e np.gradient trattava i superstiti come contigui a cell_size --
    inventando pendenze inesistenti ogni volta che un arco sfiorava il bordo della
    finestra scansionata.

    Usa np.floor, non int(): int() tronca verso lo zero e quindi sbaglia cella per
    coordinate negative. Il frame VISION di questa missione ha y sempre negative.

    valid_2d (2026-10-06): maschera delle celle attendibili (is_valid di correct_terrain).
    Se data, i campioni su celle NON attendibili (riempite, filtrate, mai scritte, sotto
    l'impronta) valgono NaN come quelli fuori griglia, e il secondo valore restituito
    diventa "dentro E attendibile".
    """
    num_rows, num_cols = terrain_2d.shape
    cols = np.floor((xs - grid_origin_x) / cell_size).astype(np.int64)
    rows = np.floor((ys - grid_origin_y) / cell_size).astype(np.int64)
    inside = (rows >= 0) & (rows < num_rows) & (cols >= 0) & (cols < num_cols)
    out = np.full(np.shape(xs), np.nan, dtype=np.float64)
    if valid_2d is not None:
        ok = inside.copy()
        ok[inside] = np.asarray(valid_2d, dtype=bool)[rows[inside], cols[inside]]
        inside = ok
    if np.any(inside):
        out[inside] = terrain_2d[rows[inside], cols[inside]]
    return out, inside


def _baseline_gradient(heights, step, baseline_m):
    """
    Differenza centrata su una base FISICA (baseline_m metri) invece che sul passo
    nativo di campionamento -- vedi SLOPE_BASELINE_M per il perche'.

    Opera sull'ultimo asse, quindi accetta sia un profilo 1D (longitudinale) sia una
    pila di fette 2D (laterale), mettendo in comune tutti i campioni. NaN-aware: dove
    manca un campione il risultato e' NaN e viene scartato, invece di produrre una
    pendenza finta tra due punti non adiacenti.

    Returns:
        array 1D dei moduli di pendenza validi (eventualmente vuoto).
    """
    n = heights.shape[-1]
    k = max(1, int(round(baseline_m / step)))
    if n <= 2 * k:
        k = max(1, (n - 1) // 2)
    if n <= 2 * k:
        return np.empty(0)
    g = np.abs(heights[..., 2 * k:] - heights[..., :-2 * k]) / (2 * k * step)
    return g[np.isfinite(g)]


def compute_arc_slope_profile(terrain_2d, grid_origin_x, grid_origin_y, cell_size,
                              x1, y1, x2, y2, perp_sample_spacing_m=0.10,
                              baseline_m=SLOPE_BASELINE_M,
                              lateral_half_width_m=LATERAL_SLICE_HALF_WIDTH_M,
                              min_sample_fraction=MIN_ARC_SAMPLE_FRACTION, valid_2d=None,
                              obstacle_2d=None):
    """
    Compute the slope Spot would actually experience walking a specific arc, in BOTH
    directions, sampled directly from real terrain heights (not a precomputed gradient
    field). A pure function of the terrain data explicitly passed in -- not tied to any
    particular object's stored state -- so both the PRM graph (build time, main thread)
    and the arc-verification background thread (real time, its own freshly-fetched scan)
    can each call this with their own current data, with no staleness or cross-thread
    coupling between them.

      - LONGITUDINAL (pitch-relevant): sample terrain height along the arc's centerline
        and differentiate it over baseline_m -- the real height profile the robot would
        walk, measured over a stride-sized baseline rather than over a single ~3cm cell.

      - LATERAL (roll-relevant): every perp_sample_spacing_m along the arc, drop a
        perpendicular slice +/- lateral_half_width_m wide (the width of the robot, NOT
        the length of the arc), sample heights across it and differentiate over the same
        baseline. All lateral samples from every slice are pooled before taking one
        percentile, so a sidehill traverse gets caught even if the along-arc profile
        alone looks flat.

    Both directions are summarized with the 90th percentile -- robust to isolated
    noisy cells, but not diluted by averaging over the whole arc.

    has_data is True only if at least min_sample_fraction of the samples actually fell
    inside the grid, in BOTH directions. The previous version returned True
    unconditionally as soon as a grid object existed: an arc entirely outside the
    currently-scanned window was reported as a LIVE measurement of 0.0 slope -- unknown
    ground presented as known-flat -- and the global-map fallback in
    PRM.sample_terrain_between_points, which is gated on `not has_data`, was therefore
    unreachable (hence global_map: 0.0% in every mission summary).

    valid_2d (2026-10-06, rete di sicurezza): maschera is_valid di correct_terrain. Le
    quote riempite (buchi del sensore, celle filtrate dal controllo sul suolo, celle mai
    scritte) non entrano piu' nelle differenze: prima la pendenza si misurava anche sul
    terreno riempito, che e' una stima. Con la maschera anche has_data conta solo i
    campioni attendibili: un arco visto per meno di meta' torna "senza dati" e il PRM
    passa alla mappa globale, invece di dichiarare piatto cio' che non ha visto.
    Senza maschera: comportamento precedente.

    obstacle_2d (2026-10-07, questione aperta B): maschera delle celle che
    `obstacle_distance` chiama GIA' ostacolo. Vengono escluse dalle differenze come le
    celle non attendibili. Motivo: la faccia verticale di un mobile non e' terreno in
    pendenza, e' un ostacolo -- ed e' gia' gestita come tale, due volte: il veto di
    occupazione del PRM e il fronte sicuro, che si ferma a FRONTIER_CLEARANCE_M da
    qualunque cella con obstacle_distance bassa. Contarla ANCHE come pendenza del
    terreno la conta la terza volta, e con il peggiore degli effetti: non ferma il
    robot davanti al mobile (ci pensa il fronte), scollega il grafo PRM in tutta la
    stanza.

    Misurato sulla missione del 2026-10-07 13:58, stanza con mobili: 200 archi distinti
    vetati per pendenza, mediana longitudinale 1.487 contro soglia 0.600, massimo 2.092
    -- valori impossibili su un pavimento piano, e la ripianificazione falliva da OGNI
    nodo (225, 227, 228) nonostante 22-28 collegamenti ciascuno, quindi la ritirata era
    l'unico esito possibile. Escludendo queste celle quegli archi non diventano
    "liberi": diventano "senza dati di pendenza" (has_data False), e un arco senza dati
    NON viene vetato ne' in build_graph ne' in _try_add_edge -- entra nel grafo con un
    costo di ripiego, e chi lo percorre lo verifica metro per metro col fronte sicuro.
    E' la scelta esplicita del 2026-10-07: la sicurezza su quegli archi la dà il fronte,
    non una pendenza misurata su un armadio.

    Senza maschera: comportamento precedente.

    Returns:
        (longitudinal_slope, lateral_slope, has_data)
    """
    if terrain_2d is None or grid_origin_x is None or cell_size is None:
        return 0.0, 0.0, False

    dist = float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
    if dist < 1e-6:
        return 0.0, 0.0, False

    # Uscita rapida (2026-10-06 sera): se l'arco, allargato della fetta laterale, non tocca
    # nemmeno la griglia, nessun campione vi cade dentro -> has_data False con pendenze 0,
    # identico al calcolo completo. In build_graph la grande maggioranza degli archi sta
    # fuori dalla finestra di ~3.8 m: prima ognuno pagava il campionamento intero (~0.2 ms,
    # due volte per arco -> ~30 s per costruzione).
    num_rows, num_cols = terrain_2d.shape
    pad = lateral_half_width_m + cell_size
    if (max(x1, x2) + pad < grid_origin_x or min(x1, x2) - pad > grid_origin_x + num_cols * cell_size or
            max(y1, y2) + pad < grid_origin_y or min(y1, y2) - pad > grid_origin_y + num_rows * cell_size):
        return 0.0, 0.0, False

    dir_x, dir_y = (x2 - x1) / dist, (y2 - y1) / dist   # along-arc unit vector
    perp_x, perp_y = -dir_y, dir_x                      # across-arc unit vector

    # Celle attendibili E non-ostacolo: una sola maschera, usata per entrambe le direzioni
    # (vedi obstacle_2d nella docstring). Se nessuna delle due e' data, resta None e il
    # campionamento si comporta esattamente come prima.
    usable_2d = valid_2d
    if obstacle_2d is not None:
        obs = np.asarray(obstacle_2d, dtype=bool)
        usable_2d = (~obs) if valid_2d is None else (np.asarray(valid_2d, dtype=bool) & ~obs)

    # --- LONGITUDINAL: along the centerline, differentiated over baseline_m ---
    num_along = max(2, int(dist / cell_size) + 1)
    t = np.linspace(0.0, 1.0, num_along)
    step_along = dist / (num_along - 1)
    h_along, inside_along = _gather_heights(
        terrain_2d, grid_origin_x, grid_origin_y, cell_size,
        x1 + t * (x2 - x1), y1 + t * (y2 - y1), valid_2d=usable_2d
    )
    grad_long = _baseline_gradient(h_along, step_along, baseline_m)
    longitudinal_slope = float(np.percentile(grad_long, 90)) if grad_long.size else 0.0

    # --- LATERAL: perpendicular slices as wide as the ROBOT, not as the arc ---
    num_slices = max(1, int(dist / perp_sample_spacing_m) + 1)
    num_perp = max(2, int(2 * lateral_half_width_m / cell_size) + 1)
    step_perp = 2 * lateral_half_width_m / (num_perp - 1)
    u = np.linspace(-lateral_half_width_m, lateral_half_width_m, num_perp)
    ts = np.linspace(0.0, 1.0, num_slices) if num_slices > 1 else np.array([0.5])
    xs_perp = (x1 + ts * (x2 - x1))[:, None] + u[None, :] * perp_x
    ys_perp = (y1 + ts * (y2 - y1))[:, None] + u[None, :] * perp_y
    h_perp, inside_perp = _gather_heights(
        terrain_2d, grid_origin_x, grid_origin_y, cell_size, xs_perp, ys_perp, valid_2d=usable_2d
    )
    grad_lat = _baseline_gradient(h_perp, step_perp, baseline_m)
    lateral_slope = float(np.percentile(grad_lat, 90)) if grad_lat.size else 0.0

    has_data = (float(inside_along.mean()) >= min_sample_fraction
                and float(inside_perp.mean()) >= min_sample_fraction
                and grad_long.size > 0 and grad_lat.size > 0)

    return longitudinal_slope, lateral_slope, has_data


def _bilinear_height(terrain_2d, grid_origin_x, grid_origin_y, cell_size, px, py):
    """
    Quota del terreno nel punto (px, py) per interpolazione bilineare tra i centri delle
    4 celle vicine (il centro della cella (r, c) sta a origine + (c + 0.5) * cell_size,
    coerente con offset_grid_pixels). None se uno dei 4 vicini cade fuori griglia o non e'
    finito. SOLO DIAGNOSTICA: usata da compute_arc_slope_profile_interp().
    """
    num_rows, num_cols = terrain_2d.shape
    fx = (px - grid_origin_x) / cell_size - 0.5
    fy = (py - grid_origin_y) / cell_size - 0.5
    c0 = int(np.floor(fx))
    r0 = int(np.floor(fy))
    if r0 < 0 or c0 < 0 or r0 + 1 >= num_rows or c0 + 1 >= num_cols:
        return None
    wx = fx - c0
    wy = fy - r0
    h = (terrain_2d[r0, c0] * (1 - wx) * (1 - wy) + terrain_2d[r0, c0 + 1] * wx * (1 - wy) +
         terrain_2d[r0 + 1, c0] * (1 - wx) * wy + terrain_2d[r0 + 1, c0 + 1] * wx * wy)
    return float(h) if np.isfinite(h) else None


def compute_arc_slope_profile_interp(terrain_2d, grid_origin_x, grid_origin_y, cell_size,
                                     x1, y1, x2, y2, perp_sample_spacing_m=0.10):
    """
    STIMA DIAGNOSTICA, NON USATA PER DECIDERE NULLA. Variante interpolata bilinearmente,
    tenuta per il confronto loggato in `[SLOPE-CHECK]` e nelle colonne long_interp /
    lat_interp del CSV per segmento.

    NOTA (2026-10-05): questa funzione nacque per correggere la lettura "a scalini" di
    compute_arc_slope_profile (CONTEXT.md, punto A). Quel problema e' ora risolto nella
    funzione principale per altra via -- base fisica SLOPE_BASELINE_M invece del passo
    nativo, piu' campionamento NaN-aware -- quindi questa resta solo come riferimento
    indipendente. Attenzione nel confronto: usa ancora la fetta laterale lunga quanto
    l'arco e la differenziazione sul passo di campionamento, quindi sui mobili restituira'
    valori molto piu' alti della stima in uso. E' atteso, non e' un disaccordo da indagare.

    Returns:
        (longitudinal_slope, lateral_slope, has_data)
    """
    if terrain_2d is None or grid_origin_x is None or cell_size is None:
        return 0.0, 0.0, False
    dist = float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
    if dist < 1e-6:
        return 0.0, 0.0, False

    dir_x, dir_y = (x2 - x1) / dist, (y2 - y1) / dist
    perp_x, perp_y = -dir_y, dir_x

    num_along = max(2, int(dist / cell_size) + 1)
    step_along = dist / (num_along - 1)
    heights_along = []
    for i in range(num_along):
        t = i / (num_along - 1)
        h = _bilinear_height(terrain_2d, grid_origin_x, grid_origin_y, cell_size,
                             x1 + t * (x2 - x1), y1 + t * (y2 - y1))
        if h is not None:
            heights_along.append(h)
    if len(heights_along) >= 2:
        # Se qualche campione e' caduto fuori griglia la serie e' accorciata ma resta contigua
        # per costruzione (si perdono solo gli estremi), quindi il passo costante vale ancora.
        longitudinal_slope = float(np.percentile(
            np.abs(np.gradient(np.array(heights_along), step_along)), 90))
    else:
        longitudinal_slope = 0.0

    num_slices = max(1, int(dist / perp_sample_spacing_m) + 1)
    num_perp = max(2, int(dist / cell_size) + 1)
    step_perp = dist / (num_perp - 1)
    pooled = []
    for s in range(num_slices):
        t = s / max(1, num_slices - 1) if num_slices > 1 else 0.5
        cx = x1 + t * (x2 - x1)
        cy = y1 + t * (y2 - y1)
        heights_perp = []
        for j in range(num_perp):
            u = (j / (num_perp - 1) - 0.5) * dist
            h = _bilinear_height(terrain_2d, grid_origin_x, grid_origin_y, cell_size,
                                 cx + u * perp_x, cy + u * perp_y)
            if h is not None:
                heights_perp.append(h)
        if len(heights_perp) >= 2:
            pooled.extend(np.abs(np.gradient(np.array(heights_perp), step_perp)).tolist())
    lateral_slope = float(np.percentile(pooled, 90)) if pooled else 0.0

    has_data = len(heights_along) >= 2 and len(pooled) > 0
    return longitudinal_slope, lateral_slope, has_data


def compute_arc_sample_coverage(terrain_2d, grid_origin_x, grid_origin_y, cell_size,
                                x1, y1, x2, y2, perp_sample_spacing_m=0.10,
                                lateral_half_width_m=LATERAL_SLICE_HALF_WIDTH_M):
    """
    SOLO DIAGNOSTICA. Frazione dei campioni di un arco che cadono davvero dentro la griglia
    locale: (frazione_lungo_l_arco, frazione_nelle_fette_laterali), entrambe in [0, 1].
    Un valore basso qui dice "fidati meno di questo numero".

    Allineata alla geometria usata da compute_arc_slope_profile: la fetta laterale e'
    +/- lateral_half_width_m, non +/- dist/2. Da quando has_data e' onesto (vedi
    compute_arc_slope_profile) questa misura e' ridondante per DECIDERE, ma resta utile
    nel CSV per capire quanto margine c'era sotto la soglia di min_sample_fraction.
    """
    if terrain_2d is None or grid_origin_x is None or cell_size is None:
        return 0.0, 0.0
    dist = float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
    if dist < 1e-6:
        return 0.0, 0.0
    dir_x, dir_y = (x2 - x1) / dist, (y2 - y1) / dist
    perp_x, perp_y = -dir_y, dir_x

    num_along = max(2, int(dist / cell_size) + 1)
    t = np.linspace(0.0, 1.0, num_along)
    _, inside_along = _gather_heights(terrain_2d, grid_origin_x, grid_origin_y, cell_size,
                                      x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    num_slices = max(1, int(dist / perp_sample_spacing_m) + 1)
    num_perp = max(2, int(2 * lateral_half_width_m / cell_size) + 1)
    u = np.linspace(-lateral_half_width_m, lateral_half_width_m, num_perp)
    ts = np.linspace(0.0, 1.0, num_slices) if num_slices > 1 else np.array([0.5])
    xs_perp = (x1 + ts * (x2 - x1))[:, None] + u[None, :] * perp_x
    ys_perp = (y1 + ts * (y2 - y1))[:, None] + u[None, :] * perp_y
    _, inside_perp = _gather_heights(terrain_2d, grid_origin_x, grid_origin_y, cell_size,
                                     xs_perp, ys_perp)

    return float(inside_along.mean()), float(inside_perp.mean())


class LocalGrid:
    def __init__(self, robot):
        self.local_grid_client = robot.ensure_client(LocalGridClient.default_service_name)
        self.footprint_correction_pending = True
        # Esito dell'ultimo filtro sull'altezza del suolo (vedi _cells_above_ground).
        self.last_ground_filter = {'ground_z': None, 'n_above': 0, 'max_above': 0.0,
                                   'source': 'non ancora eseguito', 'n_ref': 0}

    def create_vtk_no_step_grid(self, proto, robot_state_client):
        """Generate VTK polydata for the no step grid from the local grid response.
        Questa funzione:

        Cerca la local grid "no_step" nel messaggio ricevuto.

        Decodifica i valori (raw o RLE).

        Costruisce una griglia di punti (x,y,z) nel frame VISION.

        Colora le celle (rosso = non steppable, blu = steppable).

        Restituisce:

            pts -> punti 3D nel frame VISION

            cells_no_step -> valori della grid

            color -> colori RGB"""
        local_grid_proto = None
        cell_size = 0.0
        for local_grid_found in proto:
            if local_grid_found.local_grid_type_name == 'no_step':
                local_grid_proto = local_grid_found
                cell_size = local_grid_found.local_grid.extent.cell_size

        # If no relevant local grid found, return empty arrays (caller can handle)
        if local_grid_proto is None:
            return np.empty((0, 3), dtype=np.float32), np.array([], dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

        # Unpack the data field for the local grid.
        cells_no_step = self.unpack_grid(local_grid_proto).astype(np.float32)
        # Populate the x,y values with a complete combination of all possible pairs for the dimensions in the grid extent.
        ys, xs = np.mgrid[0:local_grid_proto.local_grid.extent.num_cells_x,
                          0:local_grid_proto.local_grid.extent.num_cells_y]
        # Get the estimated height (z value) of the ground in the vision frame as if the robot was standing.
        transforms_snapshot = local_grid_proto.local_grid.transforms_snapshot
        vision_tform_body = get_a_tform_b(transforms_snapshot, VISION_FRAME_NAME, BODY_FRAME_NAME)
        z_ground_in_vision_frame = self.compute_ground_height_in_vision_frame(robot_state_client)
        # Numpy vstack makes it so that each column is (x,y,z) for a single no step grid point. The height values come
        # from the estimated height of the ground plane.
        cell_count = local_grid_proto.local_grid.extent.num_cells_x * local_grid_proto.local_grid.extent.num_cells_y
        cells_est_height = np.ones(cell_count) * z_ground_in_vision_frame
        pts = np.vstack(
            [np.ravel(xs).astype(np.float32),
             np.ravel(ys).astype(np.float32), cells_est_height]).T
        pts[:, [0, 1]] *= (local_grid_proto.local_grid.extent.cell_size,
                           local_grid_proto.local_grid.extent.cell_size)
        # Determine the coloration based on whether or not the region is steppable. The regions that Spot considers it
        # cannot safely step are colored red, and the regions that are considered safe to step are colored blue.
        color = np.zeros([cell_count, 3], dtype=np.uint8)
        color[:, 0] = (cells_no_step <= 0.0)
        color[:, 2] = (cells_no_step > 0.0)
        color *= 255
        # Offset the grid points to be in the vision frame instead of the local grid frame.
        vision_tform_local_grid = get_a_tform_b(transforms_snapshot, VISION_FRAME_NAME,
                                                local_grid_proto.local_grid.frame_name_local_grid_data)
        pts = self.offset_grid_pixels(pts, vision_tform_local_grid, cell_size)

        return pts, cells_no_step, color


    def create_vtk_obstacle_grid(self, proto, robot_state_client):
        """Generate points, cell values and colors for the obstacle distance grid.

        The obstacle_distance grid encodes the signed distance (in metres) from each
        cell to the nearest detected obstacle:
            dist < 0   -> strictly inside an obstacle  -> blocked  (red)
            dist >= 0  -> border or free space          -> passable (blue)

        Zero-padding policy: no safety margin is added around obstacles.
        A cell is passable as soon as its distance is >= 0.
        Unlike the no_step grid, grass is NOT classified as an obstacle here,
        making this grid more suitable for outdoor environments.

        Returns:
            pts (np.ndarray, shape (N,3)): cell positions in the VISION frame
            cells_obstacle_dist (np.ndarray, shape (N,)): raw signed-distance values
            color (np.ndarray, shape (N,3) uint8): RGB colours per cell
        """
        local_grid_proto = None
        cell_size = 0.0
        for local_grid_found in proto:
            if local_grid_found.local_grid_type_name == 'obstacle_distance':
                local_grid_proto = local_grid_found
                cell_size = local_grid_found.local_grid.extent.cell_size

        # If no relevant local grid found, return empty arrays (caller can handle)
        if local_grid_proto is None:
            return np.empty((0, 3), dtype=np.float32), np.array([], dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

        # Unpack the raw distance values.
        cells_obstacle_dist = self.unpack_grid(local_grid_proto).astype(np.float32)

        # Build (x, y) grid coordinates.
        ys, xs = np.mgrid[0:local_grid_proto.local_grid.extent.num_cells_x,
                          0:local_grid_proto.local_grid.extent.num_cells_y]

        # Use ground-plane height for the z coordinate.
        transforms_snapshot = local_grid_proto.local_grid.transforms_snapshot
        z_ground_in_vision_frame = self.compute_ground_height_in_vision_frame(robot_state_client)
        cell_count = local_grid_proto.local_grid.extent.num_cells_x * local_grid_proto.local_grid.extent.num_cells_y
        z = np.ones(cell_count, dtype=np.float32) * z_ground_in_vision_frame

        pts = np.vstack([np.ravel(xs).astype(np.float32),
                         np.ravel(ys).astype(np.float32), z]).T
        pts[:, [0, 1]] *= (local_grid_proto.local_grid.extent.cell_size,
                           local_grid_proto.local_grid.extent.cell_size)

        # Colour coding (zero-padding - no border zone):
        #   red  -> strictly inside obstacle  (dist < 0)
        #   blue -> passable: border or free  (dist >= 0)
        color = np.zeros([cell_count, 3], dtype=np.uint8)
        color[:, 0] = (cells_obstacle_dist < 0.0)   # red  = blocked
        color[:, 2] = (cells_obstacle_dist >= 0.0)  # blue = passable
        color *= 255

        # Offset to VISION frame.
        vision_tform_local_grid = get_a_tform_b(transforms_snapshot, VISION_FRAME_NAME,
                                                local_grid_proto.local_grid.frame_name_local_grid_data)
        pts = self.offset_grid_pixels(pts, vision_tform_local_grid, cell_size)

        return pts, cells_obstacle_dist, color


    def compute_ground_height_in_vision_frame(self, robot_state_client):
        """Get the z-height of the ground plane in vision frame from the current robot state."""
        robot_state = robot_state_client.get_robot_state()
        vision_tform_ground_plane = get_a_tform_b(robot_state.kinematic_state.transforms_snapshot,
                                                  VISION_FRAME_NAME, GROUND_PLANE_FRAME_NAME)
        return vision_tform_ground_plane.position.z

    def offset_grid_pixels(self, pts, vision_tform_local_grid, cell_size):
        """
        [FIX ROTOTRASLAZIONE SE3]
        Ruota e trasla la griglia locale nel frame VISION considerando
        sia la posizione sia il quaternione di rotazione.
        """
        pts_centered = pts.copy()
        pts_centered[:, 0] += cell_size * 0.5
        pts_centered[:, 1] += cell_size * 0.5

        # Se l'oggetto della trasformazione supporta transform_cloud, usalo direttamente
        if hasattr(vision_tform_local_grid, 'transform_cloud'):
            return vision_tform_local_grid.transform_cloud(pts_centered)
        else:
            # Fallback manuale tramite matrice di rotazione R e traslazione T
            rot_mat = vision_tform_local_grid.rotation.to_matrix()
            trans = np.array([
                vision_tform_local_grid.position.x,
                vision_tform_local_grid.position.y,
                vision_tform_local_grid.position.z
            ], dtype=np.float32)
            return (pts_centered @ rot_mat.T) + trans


    def unpack_grid(self, local_grid_proto):
        """Unpack the local grid proto."""
        # Determine the data type for the bytes data.
        data_type = self.get_numpy_data_type(local_grid_proto.local_grid)
        if data_type is None:
            print('Cannot determine the dataformat for the local grid.')
            return None
        # Decode the local grid.
        if local_grid_proto.local_grid.encoding == local_grid_pb2.LocalGrid.ENCODING_RAW:
            full_grid = np.frombuffer(local_grid_proto.local_grid.data, dtype=data_type)
        elif local_grid_proto.local_grid.encoding == local_grid_pb2.LocalGrid.ENCODING_RLE:
            full_grid = self.expand_data_by_rle_count(local_grid_proto, data_type=data_type)
        else:
            # Return nothing if there is no encoding type set.
            return None
        # Apply the offset and scaling to the local grid.
        if local_grid_proto.local_grid.cell_value_scale == 0:
            return full_grid
        full_grid_float = full_grid.astype(np.float64)
        full_grid_float *= local_grid_proto.local_grid.cell_value_scale
        full_grid_float += local_grid_proto.local_grid.cell_value_offset
        return full_grid_float


    def get_numpy_data_type(self, local_grid_proto):
        """Convert the cell format of the local grid proto to a numpy data type."""
        if local_grid_proto.cell_format == local_grid_pb2.LocalGrid.CELL_FORMAT_UINT16:
            return np.uint16
        elif local_grid_proto.cell_format == local_grid_pb2.LocalGrid.CELL_FORMAT_INT16:
            return np.int16
        elif local_grid_proto.cell_format == local_grid_pb2.LocalGrid.CELL_FORMAT_UINT8:
            return np.uint8
        elif local_grid_proto.cell_format == local_grid_pb2.LocalGrid.CELL_FORMAT_INT8:
            return np.int8
        elif local_grid_proto.cell_format == local_grid_pb2.LocalGrid.CELL_FORMAT_FLOAT64:
            return np.float64
        elif local_grid_proto.cell_format == local_grid_pb2.LocalGrid.CELL_FORMAT_FLOAT32:
            return np.float32
        else:
            return None


    def expand_data_by_rle_count(self, local_grid_proto, data_type=np.int16):
        """Expand local grid data to full bytes data using the RLE count."""
        cells_pz = np.frombuffer(local_grid_proto.local_grid.data, dtype=data_type)
        cells_pz_full = []
        # For each value of rle_counts, we expand the cell data at the matching index
        # to have that many repeated, consecutive values.
        for i in range(0, len(local_grid_proto.local_grid.rle_counts)):
            for j in range(0, local_grid_proto.local_grid.rle_counts[i]):
                cells_pz_full.append(cells_pz[i])
        return np.array(cells_pz_full)

    def _fetch_layers(self, types_of_grid):
        """
        Scarica i layer richiesti. Prima prova UNA sola richiesta per tutti (stessa fotografia
        per tutti i layer); se fallisce (vecchi SDK / bug protobuf) ripiega su una richiesta
        per layer, come prima.
        """
        if getattr(self, '_multi_request_ok', True):
            try:
                out = list(self.local_grid_client.get_local_grids(list(types_of_grid)))
                self._multi_request_fails = 0
                return out
            except Exception as e:
                # Dopo 3 fallimenti di fila non la si riprova piu' (una richiesta che fallisce
                # sempre costerebbe un giro di rete a ogni scansione); un singolo errore di
                # rete invece non la spegne.
                self._multi_request_fails = getattr(self, '_multi_request_fails', 0) + 1
                if self._multi_request_fails >= 3:
                    print(f"[WARNING] Richiesta unica dei layer non riuscita 3 volte di fila ({e}): da ora "
                          f"li chiedo uno alla volta (con controllo di allineamento).")
                    self._multi_request_ok = False
        p = []
        for t in types_of_grid:
            try:
                p.extend(self.local_grid_client.get_local_grids([t]))
            except Exception as e:
                print(f"[WARNING] Fallito recupero layer '{t}': {e}")
        return p

    @staticmethod
    def _layer_geometry(lg):
        """(origine x, origine y, yaw in gradi, nx, ny, cella) del layer nel frame VISION."""
        g = lg.local_grid
        tf = get_a_tform_b(g.transforms_snapshot, VISION_FRAME_NAME, g.frame_name_local_grid_data)
        q = tf.rotation
        yaw = np.degrees(np.arctan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y ** 2 + q.z ** 2)))
        return (tf.position.x, tf.position.y, float(yaw), g.extent.num_cells_x, g.extent.num_cells_y,
                g.extent.cell_size)

    def return_local_grid(self, types_of_grid, robot_state_client, max_attempts=3):
        """
        Scarica e decodifica i layer richiesti.

        2026-10-06 sera (dopo la revisione), due controlli in piu'. Restituisce
        (None, None, None) -- che tutti i chiamanti gestiscono gia' -- se dopo max_attempts
        tentativi:
          - manca anche un solo layer richiesto. Prima si restituiva il dizionario senza quel
            layer: i chiamanti controllavano solo 'obstacle_distance' e poi leggevano
            grids_data['terrain'] -> KeyError, missione terminata per un errore di rete;
          - i layer non sono allineati fra loro (origine, dimensioni, cella diverse). Con una
            richiesta per layer, se il robot si muove fra una richiesta e l'altra la griglia
            puo' ricentrarsi: i valori di terrain finirebbero indicizzati con l'origine di
            obstacle_distance, spostati di qualche cella.
        Avvisa una volta se la griglia risulta ruotata rispetto a VISION: tutto il codice
        assume che non lo sia (indici di cella calcolati con la sola traslazione).
        """
        if isinstance(types_of_grid, str):
            types_of_grid = [types_of_grid]

        p, problem = [], None
        for attempt in range(1, max_attempts + 1):
            if attempt > 1:
                time.sleep(0.05)
            p = self._fetch_layers(types_of_grid)
            names = [lg.local_grid_type_name for lg in p]
            missing = [t for t in types_of_grid if t not in names]
            if missing:
                problem = f"layer mancanti {missing}"
                continue
            try:
                geoms = {lg.local_grid_type_name: self._layer_geometry(lg) for lg in p}
            except Exception as e:
                problem = f"geometria dei layer non leggibile ({e})"
                continue
            ref = geoms[types_of_grid[0]]
            bad = [n for n, gm in geoms.items()
                   if gm[3:5] != ref[3:5] or abs(gm[5] - ref[5]) > 1e-6
                   or np.hypot(gm[0] - ref[0], gm[1] - ref[1]) > 0.5 * ref[5]]
            if bad:
                problem = f"layer non allineati a {types_of_grid[0]}: {bad}"
                continue
            if abs(ref[2]) > 0.5 and not getattr(self, '_warned_grid_yaw', False):
                print(f"[WARNING] La griglia locale e' ruotata di {ref[2]:.2f} gradi rispetto a VISION: "
                      f"gli indici di cella assumono rotazione nulla. Da segnalare.")
                self._warned_grid_yaw = True
            problem = None
            break

        if problem is not None or not p:
            self.rejected_scans = getattr(self, 'rejected_scans', 0) + 1
            print(f"[ERROR] Griglie locali non utilizzabili dopo {max_attempts} tentativi: "
                  f"{problem or 'nessun layer scaricato'} (scansioni scartate finora: {self.rejected_scans}).")
            return None, None, None

        grids_data = {}
        # main_proto = il primo layer richiesto (obstacle_distance in tutti i chiamanti):
        # la sua origine e' quella con cui si indicizzano TUTTI i layer.
        main_proto = next(lg for lg in p if lg.local_grid_type_name == types_of_grid[0])

        for lg in p:
            grid_type = lg.local_grid_type_name
            if grid_type == 'obstacle_distance':
                pts, vals, color = self.create_vtk_obstacle_grid(p, robot_state_client)
                grids_data[grid_type] = {'pts': pts, 'values': vals, 'color': color}
            elif grid_type == 'no_step':
                pts, vals, color = self.create_vtk_no_step_grid(p, robot_state_client)
                grids_data[grid_type] = {'pts': pts, 'values': vals, 'color': color}
            elif grid_type in ['terrain', 'intensity', 'terrain_valid']:
                pts, vals, color = self.create_vtk_terrain_grid(p, robot_state_client, layer_name=grid_type)
                grids_data[grid_type] = {'pts': pts, 'values': vals, 'color': color}
                if grid_type == 'terrain':
                    # Passo di quantizzazione delle quote di QUESTA scansione: serve come
                    # tolleranza in _cells_never_written. 'raw_zero' e' tenuto per
                    # compatibilita' con i chiamanti e vale solo da interruttore della
                    # regola -- NON e' piu' il riferimento, vedi _cells_never_written.
                    grids_data[grid_type]['raw_zero'] = float(np.float32(lg.local_grid.cell_value_offset))
                    grids_data[grid_type]['scale'] = float(lg.local_grid.cell_value_scale)

        return grids_data, main_proto, p

    def create_vtk_terrain_grid(self, proto, robot_state_client, layer_name='terrain'):
        """Generate VTK polydata for any terrain-based grid from the local grid response."""
        local_grid_proto = None
        cell_size = 0.0

        for local_grid_found in proto:
            # Check against the dynamic layer_name instead of a hardcoded string
            if local_grid_found.local_grid_type_name == layer_name:
                local_grid_proto = local_grid_found
                cell_size = local_grid_found.local_grid.extent.cell_size

        if local_grid_proto is None:
            return np.empty((0, 3), dtype=np.float32), np.array([], dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

        # Decode values
        terrain_values = self.unpack_grid(local_grid_proto).astype(np.float32)

        # Build XY grid
        ys, xs = np.mgrid[
                 0:local_grid_proto.local_grid.extent.num_cells_x,
                 0:local_grid_proto.local_grid.extent.num_cells_y
                 ]

        z_vals = np.ravel(terrain_values)
        pts = np.vstack([
            np.ravel(xs).astype(np.float32),
            np.ravel(ys).astype(np.float32),
            z_vals
        ]).T

        pts[:, [0, 1]] *= (cell_size, cell_size)

        transforms_snapshot = local_grid_proto.local_grid.transforms_snapshot
        vision_tform_local_grid = get_a_tform_b(
            transforms_snapshot,
            VISION_FRAME_NAME,
            local_grid_proto.local_grid.frame_name_local_grid_data
        )

        pts = self.offset_grid_pixels(pts, vision_tform_local_grid, cell_size)

        # BUG FIX: `terrain_values` above is the raw height in the local grid's own
        # (untransformed) frame, which can sit at an arbitrary/drifting z-reference.
        # offset_grid_pixels already applied the full rotation+translation to `pts`,
        # so pts[:, 2] holds the correct absolute height in the VISION frame -- use that
        # as the returned terrain value instead of the raw, frame-relative one.
        terrain_values_vision = pts[:, 2].astype(np.float32)

        color = np.zeros([len(z_vals), 3], dtype=np.uint8)
        color[:, 1] = 255  # green

        return pts, terrain_values_vision, color

    def correct_terrain(self, terrain_values, terrain_valid_values, num_cells_x, num_cells_y,
                        robot_footprint_mask=None, cell_size=0.03, robot_rc=None,
                        unwritten_value=None, unwritten_scale=None, return_unwritten=False):
        """
        Single source of terrain correction. Fills BOTH sensor-invalid cells and
        the robot's own footprint with their nearest valid neighbor's height via
        fill_invalid_nearest(). This corrected surface is what everything
        downstream should consume -- compute_gradient_and_roughness(), the
        terrain_real used for visualization/PRM node elevation, and the is_valid
        gate that keeps fuse_obstacle_mask() from ever vetoing based on an
        extrapolated/borrowed value -- instead of each place doing its own
        ad-hoc fill or footprint patch as before.

        Args:
            terrain_values: raw terrain heights (flat or (num_cells_y, num_cells_x))
            terrain_valid_values: BD's terrain_valid grid (>= TERRAIN_VALID_MIN means
                trustworthy -- NON '> 0.0', vedi la costante per il perche')
            robot_footprint_mask: optional 2D bool mask, True under the robot's body
            unwritten_value: interruttore della regola delle celle mai scritte
                (grids_data['terrain']['raw_zero']). None = regola disattivata. NON e'
                piu' il riferimento della quota, vedi _cells_never_written.
            unwritten_scale: passo di quantizzazione (grids_data['terrain']['scale']).
            return_unwritten: se True restituisce anche la maschera delle celle mai scritte.

        Returns:
            terrain_corrected: 1D float array, filled terrain heights
            is_valid: 1D bool array -- True where the ORIGINAL sensor reading was
                      trustworthy (not footprint, not sensor-invalid, not filtered)
            unwritten: (solo con return_unwritten) 1D bool array, celle mai scritte:
                      da trattare come NON VISTE (vedi make_grid_snapshot)
        """
        terrain_2d = terrain_values.reshape((num_cells_y, num_cells_x)).astype(np.float32)

        if terrain_valid_values is not None and terrain_valid_values.size > 0:
            valid_2d = (terrain_valid_values.reshape((num_cells_y, num_cells_x)) >= TERRAIN_VALID_MIN)
        else:
            valid_2d = np.ones((num_cells_y, num_cells_x), dtype=bool)

        sensor_invalid_2d = ~valid_2d
        fill_mask = sensor_invalid_2d.copy()
        if robot_footprint_mask is not None:
            fill_mask = fill_mask | robot_footprint_mask

        # Filtro sull'altezza del suolo -- vedi GROUND_HEIGHT_TOLERANCE_M.
        above_ground = self._cells_above_ground(terrain_2d, ~fill_mask, cell_size, robot_rc)
        # Celle mai scritte: la quota si RICAVA dai dati, non si prevede piu'.
        # Si passa valid_2d (la bandiera di Spot), non ~fill_mask: vedi _cells_never_written.
        unwritten = self._cells_never_written(terrain_2d, valid_2d, unwritten_value, unwritten_scale)
        fill_mask = fill_mask | above_ground | unwritten

        terrain_corrected = fill_invalid_nearest(terrain_2d, fill_mask)
        is_valid = ~fill_mask

        if return_unwritten:
            return terrain_corrected.ravel(), is_valid.ravel(), unwritten.ravel()
        return terrain_corrected.ravel(), is_valid.ravel()

    def _cells_never_written(self, terrain_2d, sensor_valid_2d, unwritten_value, unwritten_scale):
        """
        Celle che i sensori non hanno mai scritto: restano tutte alla STESSA quota esatta.

        RISCRITTA il 2026-10-07. Prima la quota si PREVEDEVA uguale a cell_value_offset
        ("il valore grezzo 0"). Due errori indipendenti, entrambi fatali:
          1. cell_value_offset e' nel frame della GRIGLIA LOCALE, mentre le quote di
             `terrain` sono in VISION (create_vtk_terrain_grid restituisce pts[:, 2]).
             Misurato sulle scansioni del 2026-10-07: differenza 0.259-0.350 m contro una
             tolleranza di 0.25 mm.
          2. quelle celle non stanno al grezzo 0, ma a un intero qualunque (350, 273,
             269 ... diverso a ogni scansione, perche' Spot sposta l'offset).
        Risultato: la regola non e' MAI scattata -- zero celle in tutte e 16 le scansioni
        registrate, mentre il plateau era di 1845-3710 celle (11-23% della griglia).

        Ora la quota non si prevede, si RICAVA: e' la quota esatta piu' frequente fra le
        celle che SPOT STESSO dichiara non attendibili. Se almeno UNWRITTEN_MIN_CELLS la
        condividono, ogni cella a quella quota e' mai scritta -- comprese le poche che il
        layer di validita' marca buone per errore (misurato: lo 0-1.8%). Nessuna costante
        magica e nessun confronto fra frame diversi.

        Niente piu' condizione sulla distanza dal suolo: il plateau sta allo zero verticale
        della griglia locale, che con il robot in piedi cade ~0.93 m sopra il pavimento --
        sotto GROUND_HEIGHT_TOLERANCE_M = 1.2 m, quindi quella condizione non poteva
        comunque scattare nel caso normale. Il caso del 2026-10-05 (robot acceso un piano
        piu' su, 3.226 m) era l'eccezione.

        Serve ancora, adesso che TERRAIN_VALID_MIN e' corretto? Si', come rete: il
        98.2-100% di queste celle Spot le marca gia' non valide, il resto passerebbe.

        unwritten_value non e' piu' il riferimento: resta solo da interruttore,
        None = regola spenta, come prima.

        Aggiorna self.last_ground_filter['n_unwritten'] e ['n_unwritten_below'].
        """
        none = np.zeros_like(sensor_valid_2d, dtype=bool)
        lgf = self.last_ground_filter
        lgf['n_unwritten'] = 0
        lgf['n_unwritten_below'] = 0
        if unwritten_value is None:
            return none

        tol = (UNWRITTEN_MATCH_FRACTION * abs(float(unwritten_scale)) if unwritten_scale
               else UNWRITTEN_MATCH_FALLBACK_M)
        tol = max(tol, 1e-9)

        finite = np.isfinite(terrain_2d)
        cand = terrain_2d[(~sensor_valid_2d) & finite]
        if cand.size < UNWRITTEN_MIN_CELLS:
            return none
        keys, counts = np.unique(np.round(cand / tol).astype(np.int64), return_counts=True)
        i = int(np.argmax(counts))
        if counts[i] < UNWRITTEN_MIN_CELLS:
            return none
        quota = float(keys[i]) * tol

        unwritten = finite & (np.abs(terrain_2d - quota) <= tol)
        # Un plateau vero e' fatto di celle che Spot stesso dichiara non attendibili. Se
        # quella quota coincidesse con una superficie REALE (il piano di un tavolo
        # esattamente a quell'altezza) le celle sarebbero in gran parte valide: non si tocca.
        if not unwritten.any() or \
                float(np.mean(~sensor_valid_2d[unwritten])) < UNWRITTEN_MIN_SPOT_INVALID:
            return none

        lgf['n_unwritten'] = int(unwritten.sum())
        lgf['quota_unwritten'] = quota
        ground_z = lgf.get('ground_z')
        if ground_z is not None:
            lgf['n_unwritten_below'] = int((unwritten & (terrain_2d < ground_z)).sum())
        return unwritten

    def _cells_above_ground(self, terrain_2d, trusted_2d, cell_size, robot_rc=None):
        """
        Celle che stanno piu' di GROUND_HEIGHT_TOLERANCE_M SOPRA il suolo su cui il robot
        e' appoggiato. Solo sopra: vedi il commento alla costante per il perche'.

        Il suolo e' la mediana delle celle attendibili entro GROUND_REF_RADIUS_M dal robot.
        La griglia locale di Spot e' centrata sul robot (verificato sulle figure del
        2026-10-05: centro della finestra e posizione del robot coincidono entro ~5 cm),
        quindi di default il robot e' al centro; robot_rc=(riga, colonna) lo sovrascrive.

        Registra l'esito in self.last_ground_filter per i log e il salvataggio dati.
        Restituisce una maschera 2D booleana (tutta False se il suolo non e' stimabile).
        """
        num_rows, num_cols = terrain_2d.shape
        none = np.zeros_like(trusted_2d, dtype=bool)
        if robot_rc is None:
            cr, cc = (num_rows - 1) / 2.0, (num_cols - 1) / 2.0
        else:
            cr, cc = robot_rc

        rr, cc_grid = np.ogrid[:num_rows, :num_cols]
        dist = np.hypot((rr - cr) * cell_size, (cc_grid - cc) * cell_size)
        finite = np.isfinite(terrain_2d)

        source = 'sotto il robot'
        ref = trusted_2d & finite & (dist <= GROUND_REF_RADIUS_M)
        if ref.sum() < GROUND_REF_MIN_CELLS:
            source = 'anello esteso'
            ref = trusted_2d & finite & (dist <= GROUND_REF_FALLBACK_RADIUS_M)
        if ref.sum() < GROUND_REF_MIN_CELLS:
            self.last_ground_filter = {'ground_z': None, 'n_above': 0, 'max_above': 0.0,
                                       'source': 'non stimabile', 'n_ref': int(ref.sum())}
            return none

        ground_z = float(np.median(terrain_2d[ref]))
        height_above = terrain_2d - ground_z
        above = trusted_2d & finite & (height_above > GROUND_HEIGHT_TOLERANCE_M)

        self.last_ground_filter = {
            'ground_z': ground_z,
            'n_above': int(above.sum()),
            'max_above': float(height_above[above].max()) if above.any() else 0.0,
            'source': source,
            'n_ref': int(ref.sum()),
        }
        return above

    def compute_gradient_and_roughness(self, terrain_corrected, terrain_valid_values, num_cells_x, num_cells_y, cell_size,
                                       slope_baseline_m=SLOPE_BASELINE_M, is_valid=None):
        """
        Genera layer di pendenza e rugosita' a partire da un terreno GIA' corretto
        (vedi correct_terrain) -- niente piu' fill qui dentro, solo smoothing e
        differenziazione. Le zone realmente cieche (sensore, non impronta) vengono
        azzerate nell'output finale per pulizia visiva/diagnostica.

        La PENDENZA viene calcolata su una base fisica (slope_baseline_m, default
        ~25cm -- l'ordine di un passo di Spot) invece che sulla spaziatura nativa
        della cella (circa 3cm). Con una base cosi' stretta, un piccolo gradino o
        bordo (es. 8cm) produce lo stesso valore numerico di una salita continua e
        genuinamente pericolosa -- np.gradient(mean_t, cell_size) misura letteralmente
        "quanto sale da una cella alla prossima", non "che pendenza sentirebbe Spot
        camminando qui". Misurare il dislivello su una distanza comparabile a un
        passo diluisce i gradini isolati mantenendo intatta la pendenza di una vera
        salita prolungata.

        La RUGOSITA' resta sulla finestra corta originale (window_size=5): cattura
        un pericolo diverso (terreno irregolare/sassoso sotto il piede), per cui la
        scala corta e' quella corretta -- allargarla comincerebbe a nascondere
        proprio l'irregolarita' fine che deve individuare.
        """
        terrain_2d = terrain_corrected.reshape((num_cells_y, num_cells_x)).astype(np.float64)

        if terrain_valid_values is not None and terrain_valid_values.size > 0:
            sensor_invalid_mask = ~(terrain_valid_values.reshape((num_cells_y, num_cells_x)) >= TERRAIN_VALID_MIN)
        else:
            sensor_invalid_mask = np.zeros((num_cells_y, num_cells_x), dtype=bool)

        window_size = 5

        # Celle che possono entrare nelle statistiche di vicinato. Con is_valid (uscita di
        # correct_terrain) restano fuori anche le celle riempite e quelle scartate dal
        # filtro sull'altezza del suolo; senza, si torna al comportamento precedente.
        #
        # Perche' conta (2026-10-05, iterazione 6): fuse_obstacle_mask() impediva gia' a
        # una cella non attendibile di generare un veto SU SE STESSA, ma non impediva alle
        # sue VICINE valide di assorbirne il valore dentro la finestra 5x5. 29 celle mai
        # scritte hanno cosi' prodotto 83 falsi ostacoli attorno a se'. Con le statistiche
        # calcolate solo su celle valide quei veti scendono a 0 e la rugosita' massima da
        # 1.585 a 0.029, mentre i 54 veti veri dell'iterazione 2 restano tutti.
        if is_valid is not None:
            weight_2d = np.asarray(is_valid).reshape((num_cells_y, num_cells_x)).astype(np.float64)
        else:
            weight_2d = (~sensor_invalid_mask).astype(np.float64)

        # 1. Smoothing "normalizzato": media delle sole celle valide della finestra.
        #    Dove nella finestra non c'e' nessuna cella valida si tiene il terreno corretto.
        n_valid = uniform_filter(weight_2d, size=window_size)
        sum_t = uniform_filter(terrain_2d * weight_2d, size=window_size)
        sum_t2 = uniform_filter((terrain_2d ** 2) * weight_2d, size=window_size)
        has_support = n_valid > (0.5 / (window_size * window_size))
        with np.errstate(invalid='ignore', divide='ignore'):
            mean_t = np.where(has_support, sum_t / n_valid, terrain_2d)
            mean_t2 = np.where(has_support, sum_t2 / n_valid, terrain_2d ** 2)

        # 2. Calcolo Gradiente (Pendenza) su una base fisica, non sulla spaziatura nativa della cella
        k = max(1, round(slope_baseline_m / cell_size))  # es. 0.25m / 0.03m =~ 8 celle

        def _centered_diff(arr, k, axis):
            # Differenza centrata a distanza k celle, con bordo "clampato" (edge padding)
            # cosi' non introduciamo pendenze finte ai margini della griglia locale.
            pad = [(0, 0), (0, 0)]
            pad[axis] = (k, k)
            padded = np.pad(arr, pad, mode='edge')
            sl_ahead = [slice(None), slice(None)]
            sl_ahead[axis] = slice(2 * k, 2 * k + arr.shape[axis])
            sl_behind = [slice(None), slice(None)]
            sl_behind[axis] = slice(0, arr.shape[axis])
            return padded[tuple(sl_ahead)] - padded[tuple(sl_behind)]

        grad_x = _centered_diff(mean_t, k, axis=1) / (2 * k * cell_size)
        grad_y = _centered_diff(mean_t, k, axis=0) / (2 * k * cell_size)
        gradient_2d = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # 3. Calcolo Rugosita' (Deviazione Standard locale, solo celle valide) -- finestra corta invariata
        variance = mean_t2 - (mean_t ** 2)
        roughness_2d = np.sqrt(np.maximum(0.0, variance))
        # Con meno di 3 celle valide nella finestra la deviazione standard non significa nulla.
        roughness_2d[n_valid * window_size * window_size < 3] = 0.0

        # Azzeriamo SOLO le zone realmente cieche nell'output finale.
        # L'impronta del robot mantiene invece la stima interpolata realistica.
        gradient_2d[sensor_invalid_mask] = 0.0
        roughness_2d[sensor_invalid_mask] = 0.0

        return gradient_2d.ravel(), roughness_2d.ravel()


    def fuse_obstacle_mask(self, cells_obstacle_dist,
                        rough_values,
                        is_valid,
                        obstacle_threshold=OBSTACLE_THRESHOLD,
                        rough_threshold=ROUGH_THRESHOLD):
        """
        Pura logica di veto. La correzione del terreno (fill + impronta robot) e'
        gia' avvenuta a monte in correct_terrain(); questa funzione decide soltanto,
        cella per cella, se obstacle_distance / rugosita' superano le soglie --
        filtrato da is_valid cosi' che un valore estrapolato/riempito non possa mai
        da solo generare un blocco.

        NOTA: il gradiente NON fa piu' parte di questo veto per-cella. La pendenza e'
        ora valutata esclusivamente per-ARCO, su tutta la sua lunghezza reale e in
        entrambe le direzioni (longitudinale e laterale), in
        prm_graph.PRM.compute_arc_slope_profile() -- usata per scartare l'arco
        direttamente in build_graph()/refresh_local_edge_weights(), non per marcare
        singole celle come ostacolo. Gli ostacoli discreti restano di competenza
        esclusiva di obstacle_distance; la rugosita' resta un veto per-cella (una
        zona sassosa/irregolare e' comunque una proprieta' locale, non qualcosa da
        mediare sull'intera lunghezza di un arco).
        """
        cells_obstacle_dist = cells_obstacle_dist.ravel()
        rough_values = rough_values.ravel()
        is_valid = is_valid.ravel()

        assert cells_obstacle_dist.shape == rough_values.shape == is_valid.shape, \
            "Errore: I layer ambientali hanno dimensioni disallineate!"

        # Calcolo dei VETI applicato SOLO alle zone con dati validi
        # Se una zona NON e' valida (punto cieco o impronta robot), non deve attivare il veto di ostacolo
        v1 = (cells_obstacle_dist <= obstacle_threshold) & is_valid
        v3 = (rough_values > rough_threshold) & is_valid

        # Maschera binaria finale degli ostacoli
        obstacle_mask = v1 | v3
        obstacle_mask = np.where(obstacle_mask, -1.0, 1.0)

        return obstacle_mask

    def compute_robot_footprint_mask(self, pts, robot_x, robot_y, robot_yaw, num_cells_x, num_cells_y, margin=0.05,
                                     diagnostic_only=False,
                                     body_length=1.10, body_width=0.50):
        """
        Returns the footprint mask only once per mission (the very first call).
        Every call after that returns None, so downstream functions skip footprint
        masking entirely and rely on Spot's own terrain_valid_values from then on --
        the original reasoning was that the footprint gap is only ever real on the
        very first scan.

        diagnostic_only=True computes and returns the mask WITHOUT consuming the
        one-shot flag and WITHOUT changing any behaviour -- for measuring how many
        cells under the robot's body would be flagged as obstacles on later scans,
        before deciding whether to apply the mask every scan. See the FOOTPRINT-DIAG
        block in easy_walk.py.

        body_length / body_width: dimensioni del corpo (m) prima del margine. I default
        sono le dimensioni reali di Spot e NON vanno cambiati nell'uso normale. Servono
        alla diagnostica per confrontare il rettangolo intero con il solo NUCLEO (la
        fascia entro la carreggiata delle zampe): un ostacolo dentro il nucleo
        significherebbe che il robot ci sta gia' sopra, quindi mascherare li' non puo'
        nascondere nulla di raggiungibile, mentre mascherare l'intero rettangolo in un
        corridoio stretto puo' cancellare un muro vero.

        ATTENZIONE (2026-10-05): il thread di arcVerification.py chiama anche lui
        questa funzione, quindi in condizioni di corsa puo' essere IL BACKGROUND a
        consumare l'unica maschera prevista, lasciando il ciclo principale senza.
        Un motivo in piu' per misurare prima di decidere.
        """
        if not diagnostic_only:
            if not self.footprint_correction_pending:
                return None
            self.footprint_correction_pending = False

        dx = pts[:, 0] - robot_x
        dy = pts[:, 1] - robot_y
        cos_yaw, sin_yaw = np.cos(robot_yaw), np.sin(robot_yaw)
        x_body = dx * cos_yaw + dy * sin_yaw
        y_body = -dx * sin_yaw + dy * cos_yaw

        half_length = (body_length / 2.0) + margin
        half_width = (body_width / 2.0) + margin
        mask_flat = (np.abs(x_body) <= half_length) & (np.abs(y_body) <= half_width)
        return mask_flat.reshape((num_cells_y, num_cells_x))


class GlobalGrid:
    def __init__(self, resolution=0.03):
        self.resolution = resolution
        # Dizionario globale: la chiave e' (grid_x, grid_y), il valore e' lo stato:
        #  1.0 -> Libero/Sicuro
        # -1.0 -> Occupato/Ostacolo
        self.grid = {}
        self.obs_count = {}
        self._cached_margin = None
        self._cached_offsets = []
        self.global_occupancy_map = None
        self.global_terrain_map = None  # spotGrid.GlobalTerrainGrid, lazy-init altrove
        # Indice a grana grossa: blocco (GLOBAL_OCC_COARSE_CELLS celle di lato) -> numero di
        # celle occupate al suo interno. Serve a is_occupied() per rispondere "libero" senza
        # guardare cella per cella quando nell'intorno non c'e' nessun blocco con ostacoli,
        # che e' il caso di gran lunga piu' comune (vedi PRM_EDGE_SAFETY_MARGIN_M).
        self._coarse_occ = {}
        # Celle dei dischi pericolosi (mark_hazard): restano occupate per tutta la missione.
        self.hazard_cells = set()
        self.hazards = []

    def _coarse_key(self, gx, gy):
        c = GLOBAL_OCC_COARSE_CELLS
        return (gx // c, gy // c)

    def _set_state(self, key, new_state):
        """Unico punto in cui cambia lo stato di una cella: tiene allineato l'indice grosso."""
        old_state = self.grid.get(key)
        if old_state == new_state:
            return
        self.grid[key] = new_state
        ck = self._coarse_key(*key)
        if new_state == -1.0:
            self._coarse_occ[ck] = self._coarse_occ.get(ck, 0) + 1
        elif old_state == -1.0:
            n = self._coarse_occ.get(ck, 0) - 1
            if n > 0:
                self._coarse_occ[ck] = n
            else:
                self._coarse_occ.pop(ck, None)

    def _rebuild_coarse_index(self):
        self._coarse_occ = {}
        for key, state in self.grid.items():
            if state == -1.0:
                ck = self._coarse_key(*key)
                self._coarse_occ[ck] = self._coarse_occ.get(ck, 0) + 1


    def update(self, pts, obstacle_mask):
        """
        [FIX AGGIORNAMENTO PROBABILISTICO]
        Se una cella inizialmente vista occupata viene vista libera per
        piu' scansioni consecutive, torna ad essere considerata libera (1.0).
        """
        pts_flat = pts.reshape(-1, 3)
        mask_flat = obstacle_mask.ravel()

        # np.floor, non np.round -- stesso identico motivo spiegato in
        # GlobalTerrainGrid.update(): `pts` sono i CENTRI delle celle, cioe'
        # (i + 0.5) * resolution, e np.round di un mezzo punto esatto arrotonda al pari.
        # Le chiavi scritte risultavano 0, 2, 2, 4, 5, 5, 7, 7 ...: celle diverse che
        # collassano sulla stessa chiave, e chiavi intermedie (1, 3, 6 ...) mai scritte.
        # is_occupied() interroga invece con coordinate arbitrarie lungo un arco, che si
        # distribuiscono su TUTTE le chiavi: quelle mai scritte cadevano nel default
        # self.grid.get(key, 1.0), cioe' "libero". Una cella realmente occupata poteva
        # quindi essere letta come libera. In pratica l'effetto era in larga parte
        # mascherato dal safety_margin (controlla un intorno di ~13 celle e quasi sempre
        # ne becca una scritta), ma restava un buco sul percorso che decide se un arco e'
        # ostruito. Con floor su entrambi i lati la corrispondenza e' esatta e 1:1.
        gx_array = np.floor(pts_flat[:, 0] / self.resolution).astype(int)
        gy_array = np.floor(pts_flat[:, 1] / self.resolution).astype(int)

        # Isteresi (2026-10-06, vedi GLOBAL_OCC_CONFIRM_OBS):
        #   - occupata solo dopo GLOBAL_OCC_CONFIRM_OBS osservazioni di ostacolo consecutive
        #     (prima: UNA sola, da qualunque stato -- anche da "vista libera 4 volte");
        #   - libera quando il contatore torna a 0 o sotto, come prima;
        #   - nel mezzo la cella mantiene lo stato che aveva.
        # Una cella mai vista e osservata libera viene scritta come libera (serve a is_known).
        #
        # 2026-10-06 sera (dopo la revisione):
        #   - state == 0 vuol dire "nessuna informazione" (cella non attendibile in questa
        #     scansione: buco del sensore, scartata dal filtro sul suolo, mai scritta). Prima
        #     queste celle arrivavano come +1, cioe' "vista libera": l'interno di un tronco o
        #     di un muro (scartato dal filtro perche' sta sopra il suolo) risultava terreno
        #     noto e libero, e una cella che alternava ostacolo / non attendibile non veniva
        #     mai confermata occupata. Ora si salta.
        #   - le celle di un disco pericoloso (caduta) non vengono mai riportate a libere:
        #     prima bastavano GLOBAL_OCC_MAX_COUNT = 4 scansioni normali, e il punto di una
        #     caduta di solito sembra normale ai sensori (verificato: spariva alla 4a).
        hazard_cells = self.hazard_cells
        for gx, gy, state in zip(gx_array, gy_array, mask_flat):
            if state == 0:
                continue
            key = (int(gx), int(gy))
            if key in hazard_cells:
                continue
            curr_count = self.obs_count.get(key, 0)

            if state == -1.0:
                new_count = min(curr_count + 1 if curr_count >= 0 else 1, GLOBAL_OCC_MAX_COUNT)
                self.obs_count[key] = new_count
                if new_count >= GLOBAL_OCC_CONFIRM_OBS:
                    self._set_state(key, -1.0)
                elif key not in self.grid:
                    self._set_state(key, 1.0)    # vista, ma non ancora confermata occupata
            else:
                new_count = max(curr_count - 1, -GLOBAL_OCC_MAX_COUNT)
                self.obs_count[key] = new_count
                if new_count <= 0:
                    self._set_state(key, 1.0)
                elif key not in self.grid:
                    self._set_state(key, 1.0)

    def is_occupied(self, x, y, safety_margin=0.0):
        """
        Verifica se una coordinata reale (x, y) cade in una cella occupata.
        Opzionalmente controlla anche un intorno (safety_margin in metri).
        Utilizza offset pre-calcolati per evitare cicli nidificati ripetitivi.
        """
        gx_center = int(np.floor(x / self.resolution))
        gy_center = int(np.floor(y / self.resolution))

        if safety_margin <= 0.0:
            return self.grid.get((gx_center, gy_center), 1.0) == -1.0

        # Prefiltro a grana grossa: se nessun blocco che interseca l'intorno contiene celle
        # occupate, la risposta e' "libero" senza guardare le singole celle. Con il margine
        # portato a 0.15 m l'intorno e' di 81 celle; senza prefiltro la costruzione del grafo
        # passerebbe da ~2 s a ~12 s. Risultato identico al controllo completo.
        steps = int(np.ceil(safety_margin / self.resolution))
        c = GLOBAL_OCC_COARSE_CELLS
        any_coarse = False
        for cx in range((gx_center - steps) // c, (gx_center + steps) // c + 1):
            for cy in range((gy_center - steps) // c, (gy_center + steps) // c + 1):
                if (cx, cy) in self._coarse_occ:
                    any_coarse = True
                    break
            if any_coarse:
                break
        if not any_coarse:
            return False

        # Rigenera la maschera di vicinato solo se il margine cambia
        if self._cached_margin != safety_margin:
            self._cached_margin = safety_margin
            steps = int(np.ceil(safety_margin / self.resolution))
            offsets = []
            for dx in range(-steps, steps + 1):
                for dy in range(-steps, steps + 1):
                    if dx ** 2 + dy ** 2 <= steps ** 2:
                        offsets.append((dx, dy))
            self._cached_offsets = offsets

        for dx, dy in self._cached_offsets:
            if self.grid.get((gx_center + dx, gy_center + dy), 1.0) == -1.0:
                return True

        return False

    def mark_hazard(self, x, y, radius=0.5):
        """
        Segna come occupato (con il contatore al massimo) un disco attorno a un punto in cui
        e' successo qualcosa di pericoloso -- oggi: una CADUTA (2026-10-06). Il terreno li'
        puo' sembrare normale ai sensori (scivoloso, cedevole, una buca coperta d'erba), quindi
        non si aspetta che le scansioni lo confermino: il pianificatore lo evita da subito.
        Dal 2026-10-06 sera le scansioni successive NON lo liberano piu' (vedi update): resta
        occupato per tutta la missione. Restituisce il numero di celle segnate.
        """
        if not hasattr(self, 'hazards'):
            self.hazards = []
        if not hasattr(self, 'hazard_cells'):
            self.hazard_cells = set()
        self.hazards.append((float(x), float(y), float(radius)))
        r = self.resolution
        steps = int(np.ceil(radius / r))
        gx0, gy0 = int(np.floor(x / r)), int(np.floor(y / r))
        n = 0
        for dx in range(-steps, steps + 1):
            for dy in range(-steps, steps + 1):
                if dx * dx + dy * dy <= steps * steps:
                    key = (gx0 + dx, gy0 + dy)
                    self.obs_count[key] = GLOBAL_OCC_MAX_COUNT
                    self._set_state(key, -1.0)
                    self.hazard_cells.add(key)
                    n += 1
        return n

    def segment_hits_hazard(self, x1, y1, x2, y2, margin=0.0):
        """True se il segmento passa entro (raggio + margin) dal centro di un disco pericoloso."""
        for hx, hy, hr in getattr(self, 'hazards', []):
            dx, dy = x2 - x1, y2 - y1
            L2 = dx * dx + dy * dy
            t = 0.0 if L2 < 1e-12 else max(0.0, min(1.0, ((hx - x1) * dx + (hy - y1) * dy) / L2))
            if np.hypot(x1 + t * dx - hx, y1 + t * dy - hy) <= hr + margin:
                return True
        return False

    def is_known(self, x, y):
        """
        True se questa posizione e' gia' stata OSSERVATA almeno una volta, a prescindere
        dall'esito (libera o occupata).

        Serve a distinguere "l'ho vista ed e' libera" da "non l'ho mai vista", che
        is_occupied() da sola non sa fare: per una cella mai osservata il default di
        self.grid.get(key, 1.0) e' 1.0, cioe' "libera", quindi l'ignoto si traveste da
        spazio libero. Per muoversi va benissimo (si procede con prudenza e si verifica
        strada facendo), ma per SCEGLIERE un punto obiettivo no: a parita' di distanza
        dal centro della cella e' molto meglio un punto che sappiamo libero di uno che
        non abbiamo mai guardato. Vedi find_best_point_in_cell in easy_walk.py.
        """
        gx = int(np.floor(x / self.resolution))
        gy = int(np.floor(y / self.resolution))
        return (gx, gy) in self.grid

    def save_map(self, file_path):
        """Salva la mappa globale di occupazione in un file pickle."""
        with open(file_path, 'wb') as f:
            pickle.dump({
                'resolution': self.resolution,
                'grid': self.grid,
                'hazards': list(getattr(self, 'hazards', [])),   # dischi delle cadute (x, y, raggio)
            }, f)
        print(f"[MAP] Mappa globale salvata correttamente in {file_path}")

    def load_map(self, file_path):
        """Ricarica una mappa precedentemente salvata."""
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                self.resolution = data['resolution']
                self.grid = data['grid']
                self.hazards = list(data.get('hazards', []))
            self.hazard_cells = set()
            for hx, hy, hr in self.hazards:          # i dischi delle cadute restano permanenti
                st = int(np.ceil(hr / self.resolution))
                gx0, gy0 = int(np.floor(hx / self.resolution)), int(np.floor(hy / self.resolution))
                self.hazard_cells.update((gx0 + dx, gy0 + dy) for dx in range(-st, st + 1)
                                         for dy in range(-st, st + 1) if dx * dx + dy * dy <= st * st)
            self._rebuild_coarse_index()
            print(f"[MAP] Mappa globale caricata con {len(self.grid)} celle occupate/libere.")
        else:
            print(f"[WARNING] File {file_path} non trovato. Inizializzo mappa vuota.")


def _slope_profile_from_sampler(sample_height, x1, y1, x2, y2, cell_size,
                                perp_sample_spacing_m=0.10,
                                baseline_m=SLOPE_BASELINE_M,
                                lateral_half_width_m=LATERAL_SLICE_HALF_WIDTH_M,
                                min_sample_fraction=MIN_ARC_SAMPLE_FRACTION):
    """
    Nucleo di calcolo condiviso della pendenza longitudinale/laterale, indipendente
    dalla FONTE dei dati di altezza -- qui usato dalla mappa globale sparsa (vedi
    GlobalTerrainGrid.compute_arc_slope_profile). sample_height(px, py) deve restituire
    l'altezza del terreno in quel punto, o None se non disponibile.

    IMPORTANTE: usa esattamente la stessa geometria e la stessa base fisica di
    compute_arc_slope_profile (base SLOPE_BASELINE_M, fetta laterale larga quanto il
    robot). Prima differiva -- passo nativo e fetta lunga quanto l'arco -- il che non
    si notava solo perche' questo percorso era di fatto irraggiungibile (has_data era
    sempre True nel percorso live). Ora che il fallback globale funziona davvero, le
    due fonti DEVONO misurare la stessa grandezza, altrimenti lo stesso arco cambierebbe
    pendenza a seconda che sia dentro o fuori dalla finestra scansionata.

    has_data riflette se sono stati trovati ABBASTANZA campioni REALI -- per una mappa
    sparsa "esiste la mappa" non significa "ho dati in questo punto".

    Returns:
        (longitudinal_slope, lateral_slope, has_data)
    """
    dist = float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
    if dist < 1e-6:
        return 0.0, 0.0, False

    dir_x, dir_y = (x2 - x1) / dist, (y2 - y1) / dist
    perp_x, perp_y = -dir_y, dir_x

    # --- USCITA ANTICIPATA: la mappa ha dati da queste parti? -------------------
    # Questa funzione campiona una mappa SPARSA, un punto alla volta, in Python. Un
    # arco da 2 m costa ~500 lookup: ~70 lungo la mezzeria piu' ~440 nelle fette
    # laterali. In una missione reale la stragrande maggioranza degli archi del PRM
    # sta FUORI da qualunque area gia' vista (la finestra scansionata e' ~3 m, il
    # grafo copre decine di metri): nella missione del 2026-10-05 13:12 il 98% delle
    # query finiva qui e non trovava nulla, dopo aver pero' pagato tutti i ~500
    # lookup per scoprirlo -- circa 50 s di puro spreco per ogni costruzione del
    # grafo, e la costruzione avviene piu' volte per missione.
    # Tre lookup bastano a escludere il caso comune. Non e' una euristica sul
    # RISULTATO: se almeno un punto risponde si prosegue con il calcolo completo e
    # la decisione finale resta quella di min_sample_fraction, identica a prima.
    if (sample_height(x1, y1) is None
            and sample_height(0.5 * (x1 + x2), 0.5 * (y1 + y2)) is None
            and sample_height(x2, y2) is None):
        return 0.0, 0.0, False

    def _sample_array(xs, ys):
        """Campiona un array di posizioni, NaN dove la mappa non ha dati."""
        flat_x = np.ravel(xs)
        flat_y = np.ravel(ys)
        out = np.empty(flat_x.size, dtype=np.float64)
        for i in range(flat_x.size):
            h = sample_height(flat_x[i], flat_y[i])
            out[i] = np.nan if h is None else float(h)
        return out.reshape(np.shape(xs))

    # --- LONGITUDINALE ---
    num_along = max(2, int(dist / cell_size) + 1)
    t = np.linspace(0.0, 1.0, num_along)
    step_along = dist / (num_along - 1)
    h_along = _sample_array(x1 + t * (x2 - x1), y1 + t * (y2 - y1))
    cov_along = float(np.isfinite(h_along).mean())

    # Il laterale costa ~6 volte il longitudinale (una fetta ogni 10 cm, ciascuna
    # larga quanto il robot). Se la copertura longitudinale e' gia' insufficiente,
    # has_data sarebbe False comunque: inutile pagarlo.
    if cov_along < min_sample_fraction:
        return 0.0, 0.0, False

    grad_long = _baseline_gradient(h_along, step_along, baseline_m)
    longitudinal_slope = float(np.percentile(grad_long, 90)) if grad_long.size else 0.0

    # --- LATERALE ---
    num_slices = max(1, int(dist / perp_sample_spacing_m) + 1)
    num_perp = max(2, int(2 * lateral_half_width_m / cell_size) + 1)
    step_perp = 2 * lateral_half_width_m / (num_perp - 1)
    u = np.linspace(-lateral_half_width_m, lateral_half_width_m, num_perp)
    ts = np.linspace(0.0, 1.0, num_slices) if num_slices > 1 else np.array([0.5])
    xs_perp = (x1 + ts * (x2 - x1))[:, None] + u[None, :] * perp_x
    ys_perp = (y1 + ts * (y2 - y1))[:, None] + u[None, :] * perp_y
    h_perp = _sample_array(xs_perp, ys_perp)
    cov_lat = float(np.isfinite(h_perp).mean())
    grad_lat = _baseline_gradient(h_perp, step_perp, baseline_m)
    lateral_slope = float(np.percentile(grad_lat, 90)) if grad_lat.size else 0.0

    has_data = (cov_along >= min_sample_fraction and cov_lat >= min_sample_fraction
                and grad_long.size > 0 and grad_lat.size > 0)

    return longitudinal_slope, lateral_slope, has_data


class GlobalTerrainGrid:
    """
    Mappa globale, persistente, dell'ALTEZZA del terreno -- non della pendenza. La
    pendenza e' direzionale (dipende da come si percorre un arco, vedi
    compute_arc_slope_profile) e va quindi ricalcolata al volo; l'altezza invece e'
    uno scalare per punto, esattamente come l'occupazione di GlobalGrid, e puo' essere
    accumulata nel tempo con lo stesso schema a chiave (gx, gy).

    Risoluzione nativa (stessa cell_size della griglia locale di Spot, non ridotta) --
    su missioni molto lunghe questo puo' crescere parecchio in memoria; se mai diventasse
    un problema, la prima leva e' aumentare `resolution` qui, non altrove.

    Alimentata SOLO con altezze gia' corrette (vedi LocalGrid.correct_terrain) e filtrate
    da is_valid -- un valore riempito/estrapolato (impronta robot, punto cieco sensore)
    non deve mai contaminare la media, esattamente come per il veto in fuse_obstacle_mask.

    Nota sui limiti: la deriva dell'odometria visiva di Spot (frame VISION) nel tempo puo'
    far si' che due letture della "stessa" cella globale provengano in realta' da punti
    fisici leggermente diversi, sporcando silenziosamente la media -- stesso rischio gia'
    presente in GlobalGrid per l'occupazione, qui piu' insidioso perche' il dato e'
    continuo (un errore non e' vistoso come una cella libera diventata "occupata").
    """
    def __init__(self, resolution=0.03):
        self.resolution = resolution
        self.height_sum = {}    # (gx, gy) -> somma delle altezze valide osservate
        self.height_count = {}  # (gx, gy) -> numero di osservazioni valide
        # Rettangoli (xmin, xmax, ymin, ymax) aggiornati, uno per chiamata a update: servono
        # al PRM per sapere quali pendenze in cache rifare (prm_graph._sync_global_slope_cache).
        self.update_rects = []

    def update(self, pts, heights, is_valid):
        """
        Args:
            pts: (N,3) o (N,2) -- solo le colonne x,y vengono usate (posizione mondo)
            heights: array 1D di N altezze GIA' corrette (vedi correct_terrain)
            is_valid: array 1D booleano di N elementi -- True solo dove il sensore ha
                      dato una lettura affidabile (non impronta robot, non punto cieco)
        """
        pts_flat = np.asarray(pts).reshape(-1, pts.shape[-1] if pts.ndim > 1 else 2)
        x_array = pts_flat[:, 0]
        y_array = pts_flat[:, 1]
        heights_flat = np.asarray(heights).ravel()
        valid_flat = np.asarray(is_valid).ravel()

        # np.floor, non np.round. `pts` sono i CENTRI delle celle, cioe' (i + 0.5) *
        # resolution: np.round di un mezzo punto esatto arrotonda al pari, quindi le
        # chiavi scritte risultavano 0, 2, 2, 4, 5, 5, 7, 7 ... -- celle diverse che
        # collassano sulla stessa chiave e, soprattutto, chiavi intermedie che non
        # vengono MAI scritte. get_height() interroga invece con coordinate arbitrarie,
        # che si distribuiscono su tutte le chiavi 1:1, e cadeva quindi nei buchi.
        # Con floor su entrambi i lati la corrispondenza e' esatta.
        # Il bug non si vedeva finche' questo percorso era di fatto irraggiungibile
        # (has_data sempre True nel ramo live, vedi compute_arc_slope_profile).
        gx_array = np.floor(x_array / self.resolution).astype(int)
        gy_array = np.floor(y_array / self.resolution).astype(int)

        if valid_flat.any():
            xv, yv = x_array[valid_flat.astype(bool)], y_array[valid_flat.astype(bool)]
            if not hasattr(self, 'update_rects'):
                self.update_rects = []
            self.update_rects.append((float(xv.min()), float(xv.max()), float(yv.min()), float(yv.max())))

        for gx, gy, h, v in zip(gx_array, gy_array, heights_flat, valid_flat):
            if not v:
                continue
            key = (gx, gy)
            self.height_sum[key] = self.height_sum.get(key, 0.0) + float(h)
            self.height_count[key] = self.height_count.get(key, 0) + 1

    def get_height(self, x, y):
        """Altezza media osservata in quella posizione, o None se mai vista."""
        gx = int(np.floor(x / self.resolution))
        gy = int(np.floor(y / self.resolution))
        key = (gx, gy)
        count = self.height_count.get(key, 0)
        if count == 0:
            return None
        return self.height_sum[key] / count

    def compute_arc_slope_profile(self, x1, y1, x2, y2, perp_sample_spacing_m=0.10):
        """
        Come spotGrid.compute_arc_slope_profile, ma campionando dalla mappa globale
        accumulata invece che dalla griglia locale live -- usata come fallback quando
        un arco e' fuori dalla finestra scansionata IN QUESTO momento, ma l'area era
        gia' stata vista in una scansione precedente. has_data e' True solo se sono
        stati trovati abbastanza campioni reali lungo l'arco, non solo perche' la mappa
        esiste (vedi _slope_profile_from_sampler).

        Returns:
            (longitudinal_slope, lateral_slope, has_data)
        """
        rects = getattr(self, 'update_rects', None)
        if rects is not None:
            if not rects:
                return 0.0, 0.0, False
            # Uscita rapida: arco tutto fuori dal rettangolo che contiene tutto cio' che e'
            # stato visto -> nessun campione puo' avere dati (stesso risultato, senza leggere).
            if not hasattr(self, '_seen_bbox') or self._seen_bbox[4] != len(rects):
                r = np.array(rects)
                self._seen_bbox = (r[:, 0].min(), r[:, 1].max(), r[:, 2].min(), r[:, 3].max(), len(rects))
            bxmin, bxmax, bymin, bymax, _ = self._seen_bbox
            pad = LATERAL_SLICE_HALF_WIDTH_M + 2 * self.resolution
            if (max(x1, x2) + pad < bxmin or min(x1, x2) - pad > bxmax or
                    max(y1, y2) + pad < bymin or min(y1, y2) - pad > bymax):
                return 0.0, 0.0, False
        return _slope_profile_from_sampler(
            self.get_height, x1, y1, x2, y2, self.resolution, perp_sample_spacing_m
        )

    def save_map(self, file_path):
        """Salva la mappa globale di altezza in un file pickle."""
        with open(file_path, 'wb') as f:
            pickle.dump({
                'resolution': self.resolution,
                'height_sum': self.height_sum,
                'height_count': self.height_count,
                # quote nel frame VISION (rispetto all'accensione); sottrarre questo valore
                # per averle rispetto al suolo all'avvio missione. None se non noto.
                'mission_z0': getattr(self, 'mission_z0', None),
            }, f)
        print(f"[MAP] Mappa globale del terreno salvata correttamente in {file_path}")

    def load_map(self, file_path):
        """Ricarica una mappa di altezza precedentemente salvata."""
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                self.resolution = data['resolution']
                self.height_sum = data['height_sum']
                self.height_count = data['height_count']
                self.mission_z0 = data.get('mission_z0')
                # zone aggiornate sconosciute per una mappa ricaricata: niente uscite rapide
                self.update_rects = None
            print(f"[MAP] Mappa globale del terreno caricata con {len(self.height_count)} celle osservate.")
        else:
            print(f"[WARNING] File {file_path} non trovato. Inizializzo mappa terreno vuota.")