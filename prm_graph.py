"""
Probabilistic Road Map (PRM) for Spot autonomous mission.

Builds a graph connecting sampled waypoints with collision-free edges.
Edges are weighted with costs based on distance and terrain slope/gradient.
"""

import numpy as np
from typing import List, Tuple, Dict, Set, Optional
import heapq
import threading
import spotGrid
try:
    from scipy.spatial import cKDTree
except Exception:          # scipy c'e' gia' (spotGrid usa scipy.ndimage); ripiego per sicurezza
    cKDTree = None


# Densità di campionamento dell'occupazione lungo un arco (campioni per metro,
# minimo 5). Prima build_graph() usava 3/m e refresh_local_edge_weights() 5/m:
# lo stesso arco poteva essere ammesso in costruzione e rifiutato al primo
# refresh per la sola densità di campionamento, senza che nulla fosse cambiato
# nel mondo. Un'unica costante elimina quella finta instabilità.
#
# 2026-10-06: da 5/m a 10/m. Ogni campione controlla un disco di raggio pari al margine
# (spotGrid.PRM_EDGE_SAFETY_MARGIN_M = 0.15): con campioni ogni 20 cm, a meta' fra due
# campioni il disco copre solo sqrt(0.15^2 - 0.10^2) = 0.11 m, quindi uno spigolo poteva
# arrivare a 0.26 m dalla linea centrale -- meno dei 0.30 che pretende il fronte sicuro.
# Il pianificatore approvava archi che il fronte poi rifiutava, e il robot si bloccava su
# piani "validi". Con 10 cm a meta' si copre 0.14 m: 0.29 m, coerente entro una cella.
# Il costo in piu' e' assorbito dal prefiltro a grana grossa di GlobalGrid.is_occupied.
OCCUPANCY_SAMPLES_PER_M = 10
OCCUPANCY_SAMPLES_MIN = 5

# Lunghezza minima degli archi che collegano un nodo di sosta ai vicini (vedi add_stop_node).
STOP_NODE_MIN_EDGE_M = 0.20

# Attorno al nodo in cui il robot si trova ADESSO, la mappa globale degli ostacoli non si usa
# per i primi STOP_NODE_MAP_IGNORE_M di ogni arco: li' giudica il fronte sicuro sui dati dal
# vivo (che i muri e gli alberi li vede comunque). Senza questo, un robot rialzato dopo una
# caduta sta DENTRO il disco pericoloso segnato attorno al punto della caduta, ogni arco in
# uscita viene scartato e il robot resta isolato nel grafo (test_e2e_loop.py, scenario S6).
# 0.70 > 0.50 (raggio del disco) + 0.15 (margine): il robot esce in qualunque direzione, ma
# andare dritto attraverso il punto della caduta resta vietato se il disco si estende oltre.
# Vale SOLO per il nodo corrente (set_robot_node): quando il robot si sposta, il permesso
# sparisce e gli archi vengono rivalutati senza.
STOP_NODE_MAP_IGNORE_M = 0.70


class PRM:
    """
    Probabilistic Road Map for local path planning.

    Connects pre-sampled points (from global sampler) using collision-free edges.
    Uses Dijkstra's algorithm for pathfinding with slope and distance weights.
    """

    def __init__(self, max_edge_length: float = 2.0, connection_radius: float = 3.0,
                 min_edge_length: float = 0.5, alpha: float = 1.0, beta: float = 2.0, gamma: float = 0.0,
                 beta_lateral: float = None):
        # gamma al momento 0 per non dare peso alla roughness, da testare
        """
        Initialize the PRM.

        Args:
            max_edge_length: Maximum length of an edge in the graph (m)
            connection_radius: Only consider points within this radius for connection (m)
            alpha: Peso associato alla componente di distanza pura nella funzione di costo.
            beta: Peso associato alla componente di pendenza LONGITUDINALE (in avanti,
                beccheggio) nella funzione di costo.
            gamma: Peso associato alla componente di rugosità nella funzione di costo.
            beta_lateral: Peso associato alla componente di pendenza LATERALE (traverso,
                rollio) nella funzione di costo -- tipicamente maggiore di beta, perche'
                a parita' di pendenza il rollio e' piu' rischioso/scomodo per un
                quadrupede del beccheggio. Se None (default), viene derivato da beta
                tramite spotGrid.LATERAL_SLOPE_COST_MULTIPLIER.
        """
        self.max_edge_length = max_edge_length
        self.connection_radius = connection_radius
        self.min_edge_length = min_edge_length

        # Parametri della funzione di costo
        self.alpha = alpha  # Peso distanza
        self.beta = beta  # Peso pendenza longitudinale (beccheggio)
        self.gamma = gamma  # Peso rugosità (roughness)
        self.beta_lateral = (beta_lateral if beta_lateral is not None
                             else beta * spotGrid.LATERAL_SLOPE_COST_MULTIPLIER)  # Peso pendenza laterale (rollio)

        self.nodes = {}  # {point_idx: (x, y)}
        self.node_gradients = {}  # {point_idx: slope_value} <-- Memorizza la pendenza del nodo
        self.node_roughness = {}  # <-- Memorizza la rugosità del nodo
        self.edges = {}  # {point_idx: [(neighbor_idx, weight), ...]}
        self.edge_validity = {}  # {(idx1, idx2): bool} - caches edge validity

        # Variabili di appoggio per la griglia locale live
        self.current_grad_2d = None
        self.current_rough_2d = None
        self.current_terrain_2d = None  # raw (corrected) terrain heights -- for per-arc slope profiles
        self.grid_origin_x = None
        self.grid_origin_y = None
        self.cell_size = None

        # Mappa globale persistente delle altezze (spotGrid.GlobalTerrainGrid), opzionale --
        # usata come fallback per la pendenza quando un arco e' fuori dalla finestra
        # scansionata ORA ma era gia' stato visto in una scansione precedente. None finche'
        # non viene iniettata da fuori (vedi set_global_terrain_map).
        self.global_terrain_map = None
        # Cache della pendenza letta dalla mappa globale, per arco (2026-10-06 sera): era il
        # costo dominante di build_graph (~80 mila archi x ~500 letture di dizionario).
        # Una voce resta valida finche' la mappa non viene aggiornata vicino all'arco: vedi
        # GlobalTerrainGrid.update_rects e _sync_global_slope_cache().
        self._gslope_cache = {}
        self._gslope_map_id = None
        self._gslope_seen_updates = 0

        self.built = False

        self.tracker_blocked_edges = set()   # <-- ADD, alongside self.edge_validity = {}
        # Nodi creati durante la missione nella posizione reale del robot (o dell'obiettivo),
        # vedi add_stop_node(). id -> etichetta.
        self.stop_nodes = {}
        # Archi effettivamente percorsi dal robot (liberi per costruzione).
        self.traversed_edges = set()
        # Nodo in cui si trova il robot adesso, e raggio entro cui la mappa globale non si usa
        # per gli archi che ne partono (vedi STOP_NODE_MAP_IGNORE_M).
        self.robot_node = None
        self.robot_node_ignore_m = 0.0

        # ==================================================================
        # STATISTICHE DI MISSIONE -- solo per debug/log, mai usate per decidere
        # comportamento del robot. Contatori cumulativi per tutta la vita del PRM
        # (creato una volta per missione), pensati per un riepilogo di fine
        # missione (vedi print_mission_summary) invece di dover leggere migliaia
        # di righe di log.
        # ==================================================================
        self.slope_source_counts = {'live': 0, 'global_map': 0, 'fallback_nodes': 0}
        self.last_slope_source = None  # ultima fonte usata, letta da easy_walk.py per il log per-arco
        # NB: questi tre contano VALUTAZIONI di veto, non archi distinti -- refresh_local_edge_weights()
        # ricalcola lo stesso arco a ogni passo, quindi lo stesso arco ripidamente vetato viene
        # contato molte volte. Per il numero di archi DIVERSI vedi i set *_arcs qui sotto.
        self.veto_count_longitudinal = 0  # valutazioni di veto per pendenza longitudinale
        self.veto_count_lateral = 0       # valutazioni di veto per pendenza laterale
        self.veto_count_occupancy = 0     # valutazioni di veto per occupazione globale (non pendenza)
        self.vetoed_arcs_longitudinal = set()  # chiavi (min_id, max_id) di archi DISTINTI vetati per long.
        self.vetoed_arcs_lateral = set()       # idem per laterale
        self.vetoed_arcs_occupancy = set()     # idem per occupazione
        self.max_slope_seen = {'long': 0.0, 'lat': 0.0}  # massimi misurati su qualunque arco valutato
        self.slope_flip_count = 0         # volte in cui un arco vicino soglia ha cambiato stato tra due refresh
        self.edge_slope_info = {}         # {(idx1,idx2): (long_slope, lat_slope)} -- per visualizzazione, niente ricalcolo

    def add_node(self, point_idx: int, x: float, y: float, gradient: float = 0.0, roughness: float = 0.0):
        """Add a node (waypoint) to the PRM along with its terrain gradient."""
        self.nodes[point_idx] = (x, y)
        self.node_gradients[point_idx] = gradient
        self.node_roughness[point_idx] = roughness
        self.edges[point_idx] = []

    def add_nodes_from_sampler(self, global_sampler, env=None, grad_values=None, rough_values=None):
        """
        Populate PRM with nodes from global sampler and map their gradients.

        Args:
            global_sampler: GlobalSampler instance with sampled points
            env: L'istanza dell'ambiente (EnvironmentMap) per convertire le coordinate in celle della griglia
            grad_values: La griglia 2D o array flat contenente i gradienti calcolati da spotGrid
            rough_values: La griglia 2D o array flat contenente la rugosità calcolata da spotGrid
        """

        # 1. Check della forma degli array PRIMA del loop
        is_grad_2d = grad_values is not None and len(grad_values.shape) == 2
        is_rough_2d = rough_values is not None and len(rough_values.shape) == 2

        points = global_sampler.get_all_points()
        for idx, (x, y) in enumerate(points):
            node_grad = 0.0
            node_rough = 0.0

            if env is not None:
                cell = env.get_cell_from_world(x, y)
                if cell is not None:
                    r, c = cell

                    # 2. Estrazione ottimizzata
                    if grad_values is not None:
                        if is_grad_2d:
                            node_grad = grad_values[r, c]
                        else:
                            idx_flat = r * env.cols + c
                            if idx_flat < grad_values.size:
                                node_grad = grad_values[idx_flat]

                    if rough_values is not None:
                        if is_rough_2d:
                            node_rough = rough_values[r, c]
                        else:
                            idx_flat = r * env.cols + c
                            if idx_flat < rough_values.size:
                                node_rough = rough_values[idx_flat]

            self.add_node(idx, x, y, gradient=node_grad, roughness=node_rough)

        print(f"[PRM] Added {len(self.nodes)} nodes from global sampler with gradient and roughness mapping")

    def set_global_terrain_map(self, global_terrain_map):
        """
        Inietta la mappa globale persistente delle altezze (spotGrid.GlobalTerrainGrid).
        Va richiamata una volta che l'oggetto esiste (tipicamente subito dopo averlo
        creato/recuperato in easy_walk.py) -- puo' essere richiamata piu' volte, e'
        idempotente, serve solo ad aggiornare il riferimento.
        """
        self.global_terrain_map = global_terrain_map

    def update_local_grid_data(self, grad_values_2d, rough_values_2d, grid_origin_x, grid_origin_y, cell_size,
                               terrain_values_2d=None, valid_2d=None):
        """
        Salva temporaneamente la mappa locale corrente dei gradienti, delle rugosità e
        (opzionale) delle altezze del terreno grezze (già corrette, vedi correct_terrain)
        per permettere il campionamento accurato degli archi -- incluso il profilo di
        pendenza denso per-arco calcolato direttamente dalle altezze reali.
        """
        self.current_grad_2d = grad_values_2d
        self.current_rough_2d = rough_values_2d
        self.current_terrain_2d = terrain_values_2d
        # celle attendibili (is_valid di correct_terrain): le quote riempite non entrano
        # nel profilo di pendenza -- vedi spotGrid.compute_arc_slope_profile
        self.current_valid_2d = valid_2d
        self.grid_origin_x = grid_origin_x
        self.grid_origin_y = grid_origin_y
        self.cell_size = cell_size

    def refresh_local_edge_weights(self, global_map=None, edge_safety_margin: float = 0.05):
        """
        Fully re-derives all candidate edges among nodes currently inside the local
        grid window -- geometry, occupancy, and weight -- rather than only updating
        weights of edges that already survived a previous pass. This lets an edge
        wrongly invalidated by a transient/noisy occupancy reading become valid
        again once fresher data shows it's actually clear, instead of staying
        permanently removed for the rest of the attempt.
        Bounded to local_nodes x local_nodes, so still much cheaper than build_graph().

        2026-10-06: rivaluta anche gli archi che hanno UN SOLO estremo nella finestra. Prima
        solo locale x locale: un arco che partiva da un nodo vicino e attraversava un muro
        appena osservato verso un nodo lontano non veniva mai riesaminato fino alla
        ricostruzione completa, e Dijkstra continuava a proporlo. Con il fronte sicuro questo
        produceva blocchi a ripetizione sullo stesso muro (vedi test_e2e_loop.py, scenario S2).
        Gli archi percorsi dal robot (traversed_edges) non si toccano: sono liberi per
        costruzione.
        """
        if self.current_grad_2d is None or self.grid_origin_x is None or self.cell_size is None:
            return set()

        num_rows, num_cols = self.current_grad_2d.shape
        local_nodes = []
        for idx, (x, y) in self.nodes.items():
            col_idx = int(np.floor((x - self.grid_origin_x) / self.cell_size))
            row_idx = int(np.floor((y - self.grid_origin_y) / self.cell_size))
            if 0 <= row_idx < num_rows and 0 <= col_idx < num_cols:
                local_nodes.append(idx)
        local_set = set(local_nodes)

        all_ids = list(self.nodes.keys())
        all_xy = np.array([self.nodes[i] for i in all_ids], dtype=np.float64) if all_ids else np.empty((0, 2))
        reach = min(self.connection_radius, self.max_edge_length)
        # Gli archi dei nodi di sosta partono da STOP_NODE_MIN_EDGE_M (vedi add_stop_node):
        # vanno rivalutati anche loro, altrimenti un arco corto di una sosta resta valido anche
        # dopo che un ostacolo e' comparso sopra.
        is_stop = np.array([i in self.stop_nodes for i in all_ids], dtype=bool)
        self._sync_global_slope_cache()

        pairs = []
        for idx1 in local_nodes:
            x1, y1 = self.nodes[idx1]
            d_all = np.hypot(all_xy[:, 0] - x1, all_xy[:, 1] - y1)
            dmin = np.where(is_stop | (idx1 in self.stop_nodes), STOP_NODE_MIN_EDGE_M, self.min_edge_length)
            for j in np.nonzero((d_all >= dmin) & (d_all <= reach))[0]:
                idx2 = all_ids[j]
                if idx2 == idx1 or (idx2 in local_set and idx2 < idx1):
                    continue           # coppia locale-locale gia' considerata dall'altro estremo
                pairs.append((idx1, idx2, float(d_all[j])))

        for idx1, idx2, dist in pairs:
            x1, y1 = self.nodes[idx1]
            x2, y2 = self.nodes[idx2]
            key = (min(idx1, idx2), max(idx1, idx2))
            if key in self.tracker_blocked_edges:
                continue  # blocco confermato da vicino (fronte sicuro) -- mai ricreato
            if key in self.traversed_edges:
                if self._traversed_edge_ok(idx1, idx2, global_map):
                    continue  # percorso dal robot: libero per costruzione
                self.traversed_edges.discard(key)   # ora attraversa il punto di una caduta

            old_valid = self.edge_validity.get(key)  # None se l'arco non era mai stato valutato prima

            # Inizializzate qui e non dentro `if not blocked:` perche' vengono
            # rilette piu' sotto per passare il profilo gia' calcolato a
            # _compute_edge_weight (vedi sample_terrain_between_points).
            long_slope, lat_slope, has_slope_data = 0.0, 0.0, False

            blocked = False
            if global_map is not None and edge_safety_margin > 0.0:
                if self._occupancy_blocked(idx1, idx2, dist, global_map, edge_safety_margin):
                    blocked = True
                    self._record_occupancy_veto(key)

            if not blocked:
                long_slope, lat_slope, has_slope_data = self.compute_arc_slope_profile(x1, y1, x2, y2)
                if has_slope_data:
                    self.edge_slope_info[key] = (long_slope, lat_slope)  # per visualizzazione, no ricalcolo
                    if self._record_slope_veto(key, long_slope, lat_slope):
                        blocked = True

                    # --- Solo diagnostico: un arco vicino soglia che cambia stato tra due
                    # refresh consecutivi puo' indicare oscillazione dovuta a rumore del
                    # sensore piu' che a un cambiamento reale del terreno -- non influisce
                    # su nessuna decisione, serve solo a farlo emergere nei log.
                    near_thresh = spotGrid.SLOPE_THRESHOLD * (1.0 - spotGrid.NEAR_THRESHOLD_MARGIN_FRACTION)
                    is_near_threshold = (long_slope >= near_thresh) or (lat_slope >= near_thresh)
                    new_valid = not blocked
                    if is_near_threshold and old_valid is not None and old_valid != new_valid:
                        self.slope_flip_count += 1
                        print(f"[SLOPE-FLIP] Arco {idx1}-{idx2}: stato cambiato "
                              f"{old_valid}->{new_valid} vicino soglia "
                              f"(long={long_slope:.3f}, lat={lat_slope:.3f}, "
                              f"soglia={spotGrid.SLOPE_THRESHOLD:.3f})")

            self.edges[idx1] = [(n, w) for n, w in self.edges.get(idx1, []) if n != idx2]
            self.edges[idx2] = [(n, w) for n, w in self.edges.get(idx2, []) if n != idx1]

            if blocked:
                self.edge_validity[key] = False
            else:
                # Il profilo dal vivo e' gia' calcolato (anche quando dice "nessun dato"):
                # non lo si rifa' dentro il calcolo del peso. Prima con has_slope_data False
                # si passava None e il profilo veniva ricalcolato da capo.
                weight = self._compute_edge_weight(
                    idx1, idx2, dist, precomputed_slope=(long_slope, lat_slope, has_slope_data))
                self.edges[idx1].append((idx2, weight))
                self.edges[idx2].append((idx1, weight))
                self.edge_validity[key] = True

        return set(local_nodes)

    def compute_arc_slope_profile(self, x1: float, y1: float, x2: float, y2: float,
                                   perp_sample_spacing_m: float = 0.10):
        """
        Thin wrapper around spotGrid.compute_arc_slope_profile() using this PRM's own
        currently-stored live terrain grid (see update_local_grid_data). See that
        function's docstring for the full explanation of the longitudinal/lateral
        computation. Returns (longitudinal_slope, lateral_slope, has_data).
        """
        return spotGrid.compute_arc_slope_profile(
            self.current_terrain_2d, self.grid_origin_x, self.grid_origin_y, self.cell_size,
            x1, y1, x2, y2, perp_sample_spacing_m=perp_sample_spacing_m,
            valid_2d=getattr(self, 'current_valid_2d', None)
        )

    def sample_terrain_between_points(self, x1: float, y1: float, x2: float, y2: float,
                                       node_idx1: Optional[int] = None, node_idx2: Optional[int] = None,
                                       precomputed_slope: Optional[Tuple[float, float, bool]] = None):
        """
        Sample terrain difficulty along the straight segment between two points, for use
        in edge costing.

        La pendenza NON è più un singolo valore isotropo: viene scomposta in longitudinale
        (beccheggio, lungo la direzione di marcia) e laterale (rollio, perpendicolare alla
        direzione di marcia) via compute_arc_slope_profile() -- la stessa funzione già usata
        per il veto di build_graph()/arcVerification, qui riusata per il COSTO. Questo
        distingue un arco che sale/scende dritto da un arco che attraversa di lato lo stesso
        pendio (sidehill): un profilo isotropo puntuale non sa in che direzione si sta
        viaggiando e li confonde, mentre longitudinale/laterale sono per costruzione
        direzionali. La rugosità resta isotropa (proprietà locale del terreno, non della
        direzione di marcia) e viene campionata come prima: ~5 campioni/metro (minimo 5),
        90° percentile.

        node_idx1/node_idx2 sono id nodo PRM opzionali; usati come ultimo fallback statico.

        precomputed_slope: (long, lat, has_data) già calcolato dal chiamante per il veto.
        Il profilo d'arco è la singola voce di costo più pesante di tutta la costruzione
        del grafo, e prima veniva calcolato DUE volte per ogni arco -- una per il veto in
        build_graph()/refresh_local_edge_weights(), e una identica qui dentro, con gli
        stessi identici argomenti. Passandolo si dimezza il lavoro senza cambiare di una
        virgola il risultato. Se è None il profilo viene calcolato qui, come prima.

        Catena di fallback per la pendenza, dal dato migliore al più grezzo:
          1. Griglia locale live (self.current_terrain_2d) -- dati reali di QUESTO ciclo
          2. Mappa globale persistente (self.global_terrain_map, se iniettata via
             set_global_terrain_map) -- dati reali di una scansione PRECEDENTE, per un
             arco già visto ma ora fuori dalla finestra scansionata
          3. Media statica dei nodi -- nessun dato reale disponibile da nessuna fonte;
             la pendenza non è decomponibile per direzione, quindi il valore medio viene
             assegnato interamente alla componente longitudinale (laterale = 0.0), una
             stima conservativa ma non doppiamente pesata.

        Returns:
            (longitudinal_slope, lateral_slope, actual_roughness) -- floats.
        """
        dist = float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
        num_samples = max(OCCUPANCY_SAMPLES_MIN, int(dist * OCCUPANCY_SAMPLES_PER_M))

        # --- Rugosità: isotropa, campionata come prima dal campo 2D pre-calcolato ---
        sampled_roughness = []
        has_rough_grid = (self.current_rough_2d is not None and self.grid_origin_x is not None
                          and self.cell_size is not None)
        if has_rough_grid:
            num_rows, num_cols = self.current_rough_2d.shape
            for i in range(num_samples):
                t = i / (num_samples - 1)
                cx = x1 + t * (x2 - x1)
                cy = y1 + t * (y2 - y1)

                dx = cx - self.grid_origin_x
                dy = cy - self.grid_origin_y
                col_idx = int(np.floor(dx / self.cell_size))   # floor: int() sbaglia sotto zero
                row_idx = int(np.floor(dy / self.cell_size))

                if 0 <= row_idx < num_rows and 0 <= col_idx < num_cols:
                    sampled_roughness.append(self.current_rough_2d[row_idx, col_idx])

        if sampled_roughness:
            actual_roughness = float(np.percentile(sampled_roughness, 90))
        else:
            r1 = self.node_roughness.get(node_idx1, 0.0)
            r2 = self.node_roughness.get(node_idx2, 0.0)
            actual_roughness = float((r1 + r2) / 2.0)

        # --- Pendenza: direzionale, via compute_arc_slope_profile sul terreno reale ---
        if precomputed_slope is not None:
            longitudinal_slope, lateral_slope, has_slope_data = precomputed_slope
        else:
            longitudinal_slope, lateral_slope, has_slope_data = self.compute_arc_slope_profile(x1, y1, x2, y2)
        source = 'live'

        if not has_slope_data and self.global_terrain_map is not None:
            # Fuori dalla griglia locale live di QUESTO ciclo -- prima di arrendersi alla
            # media statica, proviamo la mappa globale accumulata: se quest'area era già
            # stata scansionata in un ciclo precedente, otteniamo comunque una pendenza
            # reale invece di una stima grezza.
            g_long, g_lat, g_has_data = self._global_slope(x1, y1, x2, y2, node_idx1, node_idx2)
            if g_has_data:
                longitudinal_slope, lateral_slope, has_slope_data = g_long, g_lat, True
                source = 'global_map'

        if not has_slope_data:
            # Nessun dato reale da nessuna fonte -- fallback statico sulla media dei nodi,
            # assegnata interamente al longitudinale (non sappiamo scomporla per direzione).
            grad1 = self.node_gradients.get(node_idx1, 0.0)
            grad2 = self.node_gradients.get(node_idx2, 0.0)
            longitudinal_slope = float((grad1 + grad2) / 2.0)
            lateral_slope = 0.0
            source = 'fallback_nodes'

        # Statistiche di debug -- non influenzano il comportamento, solo il log/riepilogo.
        self.slope_source_counts[source] += 1
        self.last_slope_source = source

        return float(longitudinal_slope), float(lateral_slope), actual_roughness

    def _record_slope_veto(self, key, long_slope: float, lat_slope: float) -> bool:
        """
        Applica il veto di pendenza (soglia unica spotGrid.SLOPE_THRESHOLD, su entrambe le
        direzioni) e aggiorna le statistiche di debug. Restituisce True se l'arco va scartato.
        Il comportamento di veto e' identico a prima: qui si e' solo centralizzato il
        conteggio. La prima volta che un arco DISTINTO viene vetato per una direzione
        stampa una riga con i valori, cosi' nei log si vede quali archi e con che numeri.
        """
        self.max_slope_seen['long'] = max(self.max_slope_seen['long'], long_slope)
        self.max_slope_seen['lat'] = max(self.max_slope_seen['lat'], lat_slope)

        vetoed = False
        if long_slope > spotGrid.SLOPE_THRESHOLD:
            vetoed = True
            self.veto_count_longitudinal += 1
            if key not in self.vetoed_arcs_longitudinal:
                self.vetoed_arcs_longitudinal.add(key)
                print(f"[SLOPE-VETO] Arco {key[0]}-{key[1]}: LONGITUDINALE {long_slope:.3f} > "
                      f"soglia {spotGrid.SLOPE_THRESHOLD:.3f} (lat={lat_slope:.3f})")
        if lat_slope > spotGrid.SLOPE_THRESHOLD:
            vetoed = True
            self.veto_count_lateral += 1
            if key not in self.vetoed_arcs_lateral:
                self.vetoed_arcs_lateral.add(key)
                print(f"[SLOPE-VETO] Arco {key[0]}-{key[1]}: LATERALE {lat_slope:.3f} > "
                      f"soglia {spotGrid.SLOPE_THRESHOLD:.3f} (long={long_slope:.3f})")
        return vetoed

    def _record_occupancy_veto(self, key):
        """Conteggio di debug dei veti per occupazione globale (comportamento invariato)."""
        self.veto_count_occupancy += 1
        self.vetoed_arcs_occupancy.add(key)

    def get_mission_summary_dict(self) -> dict:
        """Le stesse statistiche di print_mission_summary, in forma serializzabile (JSON)."""
        return {
            'slope_source_counts': dict(self.slope_source_counts),
            'veto_evaluations': {
                'longitudinal': self.veto_count_longitudinal,
                'lateral': self.veto_count_lateral,
                'occupancy': self.veto_count_occupancy,
            },
            'veto_distinct_arcs': {
                'longitudinal': len(self.vetoed_arcs_longitudinal),
                'lateral': len(self.vetoed_arcs_lateral),
                'occupancy': len(self.vetoed_arcs_occupancy),
            },
            'max_slope_seen': dict(self.max_slope_seen),
            'slope_flip_count': self.slope_flip_count,
            'slope_threshold': spotGrid.SLOPE_THRESHOLD,
            'slope_baseline_m': spotGrid.SLOPE_BASELINE_M,
            'lateral_slice_half_width_m': spotGrid.LATERAL_SLICE_HALF_WIDTH_M,
            'beta': self.beta,
            'beta_lateral': self.beta_lateral,
            'gamma': self.gamma,
            'alpha': self.alpha,
        }

    def print_mission_summary(self):
        """
        Stampa un riepilogo cumulativo per tutta la missione -- pensato per essere
        richiamato una volta a fine missione, invece di dover leggere migliaia di righe
        di log per capire quanto spesso ciascun meccanismo è intervenuto davvero.
        Puramente diagnostico, non influisce sul comportamento del robot.
        """
        total_sources = sum(self.slope_source_counts.values())
        print("\n================ [PRM MISSION SUMMARY] ================")
        print(f"Fonte pendenza usata nel costo degli archi (totale {total_sources} query):")
        for src, count in self.slope_source_counts.items():
            pct = (100.0 * count / total_sources) if total_sources > 0 else 0.0
            print(f"  {src:15s}: {count:6d}  ({pct:5.1f}%)")
        print("Veti -- archi DISTINTI (e, tra parentesi, valutazioni totali, che contano lo stesso arco "
              "a ogni refresh):")
        print(f"  longitudinale: {len(self.vetoed_arcs_longitudinal)} ({self.veto_count_longitudinal})  "
              f"laterale: {len(self.vetoed_arcs_lateral)} ({self.veto_count_lateral})  "
              f"occupazione: {len(self.vetoed_arcs_occupancy)} ({self.veto_count_occupancy})")
        print(f"Pendenza massima misurata su un arco -- longitudinale: {self.max_slope_seen['long']:.3f}, "
              f"laterale: {self.max_slope_seen['lat']:.3f} (soglia veto {spotGrid.SLOPE_THRESHOLD:.3f})")
        print(f"Geometria del profilo d'arco -- base {spotGrid.SLOPE_BASELINE_M:.2f} m, "
              f"fetta laterale +/-{spotGrid.LATERAL_SLICE_HALF_WIDTH_M:.2f} m")
        print(f"Pesi del costo -- alpha={self.alpha}, beta(long)={self.beta}, "
              f"beta_lateral={self.beta_lateral}, gamma={self.gamma}")
        print(f"Flip del veto pendenza vicino soglia (possibile oscillazione): {self.slope_flip_count}")
        print("=========================================================\n")

    def _compute_edge_weight(self, idx1: int, idx2: int, dist: float,
                             precomputed_slope: Optional[Tuple[float, float, bool]] = None) -> float:
        """
        Funzione di costo multi-layer che campiona l'arco in più punti per rilevare
        pendenza (longitudinale e laterale separate) o rugosità nascoste nel mezzo.

        La componente laterale (rollio/sidehill) è pesata con self.beta_lateral, tipicamente
        maggiore di self.beta (longitudinale) -- vedi spotGrid.LATERAL_SLOPE_COST_MULTIPLIER.
        Questo fa sì che, tra due archi ENTRAMBI ammessi (nessuno sopra SLOPE_THRESHOLD in
        nessuna direzione), il pianificatore preferisca quello meno esposto al traverso
        laterale a parità di pendenza complessiva -- prima di questa modifica un arco in
        sidehill sotto soglia costava esattamente quanto un arco piano con lo stesso valore
        di gradiente isotropo, perché la funzione di costo non sapeva distinguere le due
        direzioni.

        precomputed_slope viene inoltrato a sample_terrain_between_points per evitare di
        ricalcolare il profilo d'arco che il chiamante ha già calcolato per il veto.
        """
        x1, y1 = self.nodes[idx1]
        x2, y2 = self.nodes[idx2]

        longitudinal_slope, lateral_slope, actual_roughness = self.sample_terrain_between_points(
            x1, y1, x2, y2, node_idx1=idx1, node_idx2=idx2,
            precomputed_slope=precomputed_slope
        )

        # Calcolo costo esteso
        weight = ((self.alpha * dist) + (self.beta * longitudinal_slope) +
                 (self.beta_lateral * lateral_slope) + (self.gamma * actual_roughness))
        return float(weight)

    def _global_slope(self, x1, y1, x2, y2, idx1=None, idx2=None):
        """
        Pendenza dalla mappa globale, con cache per arco. La cache si usa solo quando i punti
        sono davvero i due nodi (non per un tratto parziale verso un punto intermedio).
        """
        tm = self.global_terrain_map
        cacheable = (idx1 is not None and idx2 is not None and idx1 in self.nodes and idx2 in self.nodes
                     and self.nodes[idx1] == (x1, y1) and self.nodes[idx2] == (x2, y2))
        if not cacheable:
            return tm.compute_arc_slope_profile(x1, y1, x2, y2)
        key = (min(idx1, idx2), max(idx1, idx2))
        hit = self._gslope_cache.get(key)
        if hit is None:
            hit = tm.compute_arc_slope_profile(x1, y1, x2, y2)
            self._gslope_cache[key] = hit
        return hit

    def _sync_global_slope_cache(self):
        """
        Butta le voci della cache su cui la mappa globale e' cambiata: gli archi il cui
        rettangolo (allargato della fetta laterale) interseca una zona aggiornata dopo il
        calcolo. Se la mappa non registra le zone aggiornate, svuota tutto (sempre corretto).
        """
        tm = self.global_terrain_map
        if tm is None:
            self._gslope_cache.clear()
            return
        if id(tm) != self._gslope_map_id:
            self._gslope_cache.clear()
            self._gslope_map_id = id(tm)
            self._gslope_seen_updates = len(getattr(tm, 'update_rects', None) or [])
            return
        rects = getattr(tm, 'update_rects', None)
        if rects is None:
            self._gslope_cache.clear()
            return
        new = rects[self._gslope_seen_updates:]
        self._gslope_seen_updates = len(rects)
        if not new or not self._gslope_cache:
            return
        keys = [k for k in self._gslope_cache if k[0] in self.nodes and k[1] in self.nodes]
        if len(keys) != len(self._gslope_cache):
            self._gslope_cache = {k: self._gslope_cache[k] for k in keys}
        if not keys:
            return
        a = np.array([self.nodes[k[0]] for k in keys]); b = np.array([self.nodes[k[1]] for k in keys])
        m = spotGrid.LATERAL_SLICE_HALF_WIDTH_M + 2 * tm.resolution
        exmin, exmax = np.minimum(a[:, 0], b[:, 0]) - m, np.maximum(a[:, 0], b[:, 0]) + m
        eymin, eymax = np.minimum(a[:, 1], b[:, 1]) - m, np.maximum(a[:, 1], b[:, 1]) + m
        stale = np.zeros(len(keys), dtype=bool)
        for (rxmin, rxmax, rymin, rymax) in new:
            stale |= (exmax >= rxmin) & (exmin <= rxmax) & (eymax >= rymin) & (eymin <= rymax)
        for j in np.nonzero(stale)[0]:
            del self._gslope_cache[keys[j]]

    def _traversed_edge_ok(self, a, b, global_map) -> bool:
        """
        Un arco percorso dal robot resta nel grafo senza controlli (e' libero per costruzione),
        TRANNE se e' stato bloccato dopo (blocco confermato o fallimento fisico) o se ora passa
        sopra il punto di una caduta. Prima veniva rimesso sempre, a forza: un arco percorso
        all'andata e poi bloccato tornava nel grafo alla ricostruzione successiva (verificato).
        """
        key = (min(a, b), max(a, b))
        if key in self.tracker_blocked_edges:
            return False
        if global_map is not None and hasattr(global_map, 'segment_hits_hazard'):
            (xa, ya), (xb, yb) = self.nodes[a], self.nodes[b]
            if global_map.segment_hits_hazard(xa, ya, xb, yb, spotGrid.PRM_EDGE_SAFETY_MARGIN_M):
                return False
        return True

    def get_nearest_node(self, x: float, y: float) -> Optional[int]:
        """Find the ID of the nearest node in the PRM to the given coordinates."""
        if not self.nodes:
            return None
        min_dist = float('inf')
        nearest_idx = None
        for idx, (nx, ny) in self.nodes.items():
            dist = np.sqrt((nx - x)**2 + (ny - y)**2)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = idx
        return nearest_idx

    def set_robot_node(self, node_id, ignore_map_within_m: float = STOP_NODE_MAP_IGNORE_M):
        """Registra il nodo in cui si trova il robot adesso (vedi STOP_NODE_MAP_IGNORE_M)."""
        self.robot_node = node_id
        self.robot_node_ignore_m = float(ignore_map_within_m)

    def _occupancy_blocked(self, idx1, idx2, dist, global_map, edge_safety_margin) -> bool:
        """
        True se l'arco passa vicino a celle occupate nella mappa globale. Unico punto usato da
        _evaluate_and_add_edge e da refresh_local_edge_weights. I campioni entro
        robot_node_ignore_m dal nodo in cui si trova il robot non contano.
        """
        x1, y1 = self.nodes[idx1]
        x2, y2 = self.nodes[idx2]
        ignore_xy = None
        if self.robot_node is not None and self.robot_node in (idx1, idx2) and self.robot_node_ignore_m > 0:
            ignore_xy = self.nodes[self.robot_node]
        # ceil + 1: passo fra campioni <= 1/OCCUPANCY_SAMPLES_PER_M (prima int() dava fino a
        # 0.147 m fra due campioni invece di 0.10, vedi revisione 2026-10-06 sera).
        num_samples = max(OCCUPANCY_SAMPLES_MIN, int(np.ceil(dist * OCCUPANCY_SAMPLES_PER_M)) + 1)
        for k in range(num_samples):
            t = k / (num_samples - 1)
            sx = x1 + t * (x2 - x1)
            sy = y1 + t * (y2 - y1)
            if ignore_xy is not None and np.hypot(sx - ignore_xy[0], sy - ignore_xy[1]) <= self.robot_node_ignore_m:
                continue
            if global_map.is_occupied(sx, sy, safety_margin=edge_safety_margin):
                return True
        return False

    def _evaluate_and_add_edge(self, idx1: int, idx2: int, dist: float,
                               global_map=None, edge_safety_margin: float = 0.0,
                               force: bool = False) -> bool:
        """
        Decide se l'arco idx1-idx2 e' ammesso e, se si', lo aggiunge con il suo peso.
        Estratta da build_graph() il 2026-10-06 perche' serve identica anche per collegare
        i nodi di sosta (connect_node): due copie della stessa logica divergerebbero.

        Ordine dei controlli invariato: archi bloccati in questo tentativo, occupazione nella
        mappa globale, pendenza (longitudinale e laterale), poi il peso.

        force=True salta i veti e aggiunge comunque l'arco: serve SOLO per il tratto che il
        robot ha appena percorso con le proprie zampe, che e' libero per costruzione.
        Restituisce True se l'arco e' stato aggiunto.
        """
        x1, y1 = self.nodes[idx1]
        x2, y2 = self.nodes[idx2]
        key = (min(idx1, idx2), max(idx1, idx2))

        long_slope, lat_slope, has_slope_data = 0.0, 0.0, False
        if not force:
            if key in self.tracker_blocked_edges:
                self.edge_validity[key] = False
                return False

            # Se abbiamo una mappa globale, controlliamo che l'arco non passi in zone occupate.
            if global_map is not None and edge_safety_margin > 0.0:
                if self._occupancy_blocked(idx1, idx2, dist, global_map, edge_safety_margin):
                    self._record_occupancy_veto(key)
                    self.edge_validity[key] = False
                    return False

            # Scartiamo l'arco se troppo ripido in AVANTI (longitudinale) o di LATO
            # (laterale, sidehill) -- vedi compute_arc_slope_profile(). Registrare
            # esplicitamente l'invalidità serve a contare correttamente i [SLOPE-FLIP].
            long_slope, lat_slope, has_slope_data = self.compute_arc_slope_profile(x1, y1, x2, y2)
            if has_slope_data:
                self.edge_slope_info[key] = (long_slope, lat_slope)  # per visualizzazione
                if self._record_slope_veto(key, long_slope, lat_slope):
                    self.edge_validity[key] = False
                    return False

        weight = self._compute_edge_weight(
            idx1, idx2, dist,
            precomputed_slope=(None if force else (long_slope, lat_slope, has_slope_data)))

        self.edges[idx1] = [(n, w) for n, w in self.edges.get(idx1, []) if n != idx2]
        self.edges[idx2] = [(n, w) for n, w in self.edges.get(idx2, []) if n != idx1]
        self.edges[idx1].append((idx2, weight))
        self.edges[idx2].append((idx1, weight))
        self.edge_validity[key] = True
        return True

    def connect_node(self, idx: int, global_map=None, edge_safety_margin: float = 0.0,
                     min_edge_length: Optional[float] = None) -> int:
        """
        Collega UN nodo ai vicini entro connection_radius / max_edge_length, con gli stessi
        controlli di build_graph(). Lineare nel numero di nodi, invece che quadratico: si
        puo' chiamare a ogni sosta senza ricostruire il grafo. Restituisce quanti archi ha
        aggiunto.
        """
        if min_edge_length is None:
            min_edge_length = self.min_edge_length
        x, y = self.nodes[idx]
        added = 0
        for other, (ox, oy) in list(self.nodes.items()):
            if other == idx:
                continue
            dist = float(np.hypot(ox - x, oy - y))
            if min_edge_length <= dist <= min(self.connection_radius, self.max_edge_length):
                if self._evaluate_and_add_edge(idx, other, dist, global_map, edge_safety_margin):
                    added += 1
        return added

    def add_stop_node(self, x: float, y: float, global_map=None, edge_safety_margin: float = 0.0,
                      came_from: Optional[int] = None, label: str = "sosta",
                      is_robot: bool = True) -> int:
        """
        Crea SEMPRE un nodo nuovo nella posizione indicata e lo collega ai vicini.

        Decisione dell'utente (2026-10-06): il robot non viene MAI agganciato al nodo
        esistente piu' vicino. Dopo un arresto a meta' arco il nodo piu' vicino puo' essere
        quello alle sue spalle, o uno raggiungibile solo attraversando cio' che lo ha fatto
        fermare: pianificare da li' vuol dire pianificare da un punto in cui il robot non e'.

        came_from: il nodo da cui il robot e' appena arrivato camminando. Quell'arco viene
        aggiunto comunque (force=True), a qualunque lunghezza: e' stato appena percorso,
        quindi e' libero per costruzione -- l'unico arco del grafo verificato con le zampe.

        I collegamenti agli altri vicini partono da STOP_NODE_MIN_EDGE_M invece che da
        min_edge_length: una sosta cade spesso a pochi decimetri da un nodo campionato, ed
        escluderlo lascerebbe la sosta poco collegata proprio verso dove si trova.
        """
        new_id = max(self.nodes.keys(), default=-1) + 1
        self.add_node(new_id, x, y)
        self.stop_nodes[new_id] = label
        if is_robot:
            # il robot e' qui: i primi metri li giudica il fronte, non la mappa
            self.set_robot_node(new_id)
        n = self.connect_node(new_id, global_map, edge_safety_margin,
                              min_edge_length=STOP_NODE_MIN_EDGE_M)
        if came_from is not None and came_from in self.nodes and came_from != new_id:
            fx, fy = self.nodes[came_from]
            already = any(nb == came_from for nb, _ in self.edges[new_id])
            self._evaluate_and_add_edge(new_id, came_from, float(np.hypot(fx - x, fy - y)), force=True)
            self.traversed_edges.add((min(new_id, came_from), max(new_id, came_from)))
            n += 0 if already else 1
        print(f"[NODO] {label}: nuovo nodo {new_id} in ({x:.2f}, {y:.2f}), {n} collegamenti"
              + (f" (incluso l'arco appena percorso da {came_from})" if came_from is not None else ""))
        return new_id

    def build_graph(self, global_map=None, edge_safety_margin: float = 0.0):
        """
        Build the PRM graph by connecting nearby nodes using custom edge weights.
        Se global_map è fornita, scarta gli archi che attraversano celle globalmente occupate.
        """
        print(f"[PRM] Building graph with {len(self.nodes)} nodes...")

        # Reset archi
        for idx in self.nodes.keys():
            self.edges[idx] = []
        self.edge_validity = {}
        # Il permesso di ignorare la mappa vicino al robot vale per il nodo in cui il robot si
        # trova ADESSO: quello del tentativo precedente non c'entra piu' (lo reimposta
        # add_stop_node subito dopo).
        self.robot_node = None
        self._sync_global_slope_cache()

        node_ids = list(self.nodes.keys())
        reach = min(self.connection_radius, self.max_edge_length)

        # Coppie candidate. Prima un doppio ciclo Python su tutte le coppie (~5 milioni con
        # 3150 nodi, ~4.6 s solo per le distanze); ora un albero k-d, stesse coppie. Ordinate,
        # cosi' l'ordine di valutazione e' deterministico come prima.
        if cKDTree is not None and len(node_ids) > 1:
            xy = np.array([self.nodes[i] for i in node_ids], dtype=np.float64)
            pairs = sorted(cKDTree(xy).query_pairs(r=reach + 1e-9))
            for i, j in pairs:
                dist = float(np.hypot(xy[j, 0] - xy[i, 0], xy[j, 1] - xy[i, 1]))
                if self.connection_radius >= dist >= self.min_edge_length and dist <= self.max_edge_length:
                    self._evaluate_and_add_edge(node_ids[i], node_ids[j], dist, global_map, edge_safety_margin)
        else:
            for i, idx1 in enumerate(node_ids):
                x1, y1 = self.nodes[idx1]
                for idx2 in node_ids[i + 1:]:
                    x2, y2 = self.nodes[idx2]
                    dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    if self.connection_radius >= dist >= self.min_edge_length and dist <= self.max_edge_length:
                        self._evaluate_and_add_edge(idx1, idx2, dist, global_map, edge_safety_margin)

        # Gli archi percorsi dal robot sono liberi per costruzione: la ricostruzione li ha
        # azzerati insieme a tutti gli altri, e i nodi di sosta spesso stanno a meno di
        # min_edge_length dal nodo precedente, quindi il ciclo sopra non li ricreerebbe.
        # Eccezioni: bloccati dopo, o sopra il punto di una caduta (_traversed_edge_ok).
        for a, b in list(self.traversed_edges):
            if a in self.nodes and b in self.nodes:
                if not self._traversed_edge_ok(a, b, global_map):
                    self.traversed_edges.discard((a, b))
                    continue
                (xa, ya), (xb, yb) = self.nodes[a], self.nodes[b]
                self._evaluate_and_add_edge(a, b, float(np.hypot(xb - xa, yb - ya)), force=True)

        self.built = True

        total_edges = sum(len(neighbors) for neighbors in self.edges.values()) // 2
        print(f"[PRM] Built graph with {total_edges} edges using Slope-Distance cost function")

    def mark_edge_invalid(self, idx1: int, idx2: int):
        """
        Segna un arco come bloccato: non viene piu' ricreato ne' da refresh_local_edge_weights
        ne' (dal 2026-10-06) da build_graph / connect_node.

        Dal 2026-10-06 la chiama SOLO il ciclo principale, dopo un blocco confermato DA
        VICINO dal fronte sicuro (il robot non riesce ad avanzare per piu' scansioni
        consecutive). Prima la chiamava il vecchio verificatore su verdetti presi da lontano
        e parziali, che e' cio' che ha causato le ritirate da terreno libero del 2026-10-05.
        """
        key = (min(idx1, idx2), max(idx1, idx2))
        self.edge_validity[key] = False
        self.tracker_blocked_edges.add(key)   # <-- ADD: permanent, never re-added by refresh_local_edge_weights
        self.traversed_edges.discard(key)     # percorso in passato non vuol dire percorribile ora

        if idx2 in [n[0] for n in self.edges.get(idx1, [])]:
            self.edges[idx1] = [(n, w) for n, w in self.edges[idx1] if n != idx2]
        if idx1 in [n[0] for n in self.edges.get(idx2, [])]:
            self.edges[idx2] = [(n, w) for n, w in self.edges[idx2] if n != idx1]

    def is_edge_valid(self, idx1: int, idx2: int) -> bool:
        """Check if an edge is assumed valid (not yet marked as blocked)."""
        key = (min(idx1, idx2), max(idx1, idx2))
        return self.edge_validity.get(key, False)

    def compute_path_cost(self, path_ids, allow_direct_first_hop: bool = False):
        """
        Sum of edge weights along a sequence of node ids, using the graph's current weights.

        allow_direct_first_hop (2026-10-06 sera): il primo tratto, dal robot al primo
        waypoint, puo' non essere un arco del grafo -- succede dopo una SCORCIATOIA, quando il
        robot va dritto verso un nodo piu' avanti. Quel tratto e' confermato libero dal fronte;
        lo si conta come alpha * distanza. Prima risultava "piano rotto" (costo infinito): si
        ripianificava a ogni scansione, il conteggio di conferma di un blocco ripartiva da zero
        e un blocco vero non veniva mai dichiarato.
        """
        if path_ids is None or len(path_ids) < 2:
            return 0.0
        total = 0.0
        for i in range(len(path_ids) - 1):
            a, b = path_ids[i], path_ids[i + 1]
            weight = None
            for n, w in self.edges.get(a, []):
                if n == b:
                    weight = w
                    break
            key = (min(a, b), max(a, b))
            if (weight is None and i == 0 and allow_direct_first_hop and a in self.nodes and b in self.nodes
                    and key not in self.tracker_blocked_edges and self.edge_validity.get(key) is not False):
                # Solo per un tratto che NON e' mai stato scartato: un arco appena tolto dal
                # grafo (occupazione, pendenza, pericolo, blocco) deve far ripianificare.
                (xa, ya), (xb, yb) = self.nodes[a], self.nodes[b]
                weight = self.alpha * float(np.hypot(xb - xa, yb - ya))
            if weight is None:
                return float('inf')  # edge no longer exists (e.g. invalidated)
            total += weight
        return total

    def find_path_dijkstra(self, start_idx: int, goal_idx: int,
                            current_path_ids: Optional[List[int]] = None,
                            margin: float = 0.0) -> Optional[List[int]]:
        """
        Find shortest path between two nodes using Dijkstra's algorithm.

        If current_path_ids is given (a plan already being followed, starting at start_idx)
        and margin > 0, the newly-computed path is only returned if it's at least `margin`
        fraction cheaper than sticking with current_path_ids -- this prevents oscillation
        between near-equal routes caused by noisy per-step weight updates. Otherwise
        (current_path_ids=None, the default), behaves exactly as a plain shortest-path search.

        NB per chi legge i log: quando la stabilità tiene il piano vecchio, questa funzione
        restituisce current_path_ids invariato. Il chiamante in easy_walk.py deve
        distinguere quel caso (piano STABILE, normale) dal ritorno None (nessun percorso
        trovato, anomalia vera) -- vedi il messaggio [REPLAN WARNING], che nelle missioni
        del 2026-10-05 etichettava il primo caso come se fosse il secondo.
        """
        if start_idx not in self.nodes or goal_idx not in self.nodes:
            return None

        distances = {node: float('inf') for node in self.nodes.keys()}
        distances[start_idx] = 0
        parents = {node: None for node in self.nodes.keys()}

        pq = [(0, start_idx)]
        visited = set()
        new_path = None

        while pq:
            current_dist, current = heapq.heappop(pq)
            if current in visited:
                continue
            visited.add(current)

            if current == goal_idx:
                path = []
                node = goal_idx
                while node is not None:
                    path.append(node)
                    node = parents[node]
                new_path = path[::-1]
                break

            for neighbor, weight in self.edges.get(current, []):
                if neighbor not in visited:
                    new_dist = current_dist + weight
                    if new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        parents[neighbor] = current
                        heapq.heappush(pq, (new_dist, neighbor))

        # --- Plain behavior: no stability check requested ---
        if current_path_ids is None or margin <= 0.0:
            return new_path

        # --- Stability check ---
        current_cost = self.compute_path_cost(current_path_ids, allow_direct_first_hop=True)

        if new_path is None:
            return current_path_ids if current_cost != float('inf') else None

        if current_cost == float('inf'):
            return new_path  # current plan is broken (an edge got invalidated) -> must switch

        new_cost = self.compute_path_cost(new_path)
        if new_cost < current_cost * (1.0 - margin):
            return new_path

        return current_path_ids

    def get_node_position(self, node_idx: int) -> Optional[Tuple[float, float]]:
        """Get the (x, y) position of a node."""
        return self.nodes.get(node_idx)

    def get_node_neighbors(self, node_idx: int) -> List[int]:
        """Get indices of neighboring nodes."""
        return [idx for idx, _ in self.edges.get(node_idx, [])]

    def add_temporary_node(self, x: float, y: float, current_gradient: float = 0.0, current_roughness: float = 0.0) -> int:
        """
        Add a temporary node (e.g., current robot position) for pathfinding.

        Returns:
        Temporary node index (negative value)
        """
        temp_idx = -(len(self.nodes) + 1)
        self.nodes[temp_idx] = (x, y)
        self.node_gradients[temp_idx] = current_gradient  # Salva gradiente temporaneo
        self.node_roughness[temp_idx] = current_roughness
        self.edges[temp_idx] = []

        # Connect to nearby permanent nodes
        for perm_idx, (px, py) in self.nodes.items():
            if perm_idx >= 0: # Only permanent nodes
                dist = np.sqrt((px - x)**2 + (py - y)**2)
                if dist <= self.connection_radius:
                    # Calcolo dinamico del costo anche per i nodi temporanei inseriti al volo
                    weight = self._compute_edge_weight(temp_idx, perm_idx, dist)
                    self.edges[temp_idx].append((perm_idx, weight))
                    self.edges[perm_idx].append((temp_idx, weight))
                    key = (min(temp_idx, perm_idx), max(temp_idx, perm_idx))
                    self.edge_validity[key] = True

        return temp_idx

    def remove_temporary_node(self, temp_idx: int):
        """Remove a temporary node and its edges."""
        if temp_idx in self.nodes:

            # 1. Rimuoviamo gli archi di ritorno (Unpacking corretto della tupla!)
            for neighbor_idx, weight in list(self.edges.get(temp_idx, [])):
                if neighbor_idx in self.edges:
                    self.edges[neighbor_idx] = [(n, w) for n, w in self.edges[neighbor_idx] if n != temp_idx]

                key = (min(temp_idx, neighbor_idx), max(temp_idx, neighbor_idx))
                self.edge_validity.pop(key, None)

            # 2. Rimuoviamo le coordinate fisiche
            del self.nodes[temp_idx]

            # 3. Rimuoviamo il nodo dalla lista degli archi
            if temp_idx in self.edges:
                del self.edges[temp_idx]

            # 4. Puliamo i dati ambientali legati al nodo in modo sicuro
            if temp_idx in self.node_gradients:
                del self.node_gradients[temp_idx]
            if temp_idx in self.node_roughness:
                del self.node_roughness[temp_idx]