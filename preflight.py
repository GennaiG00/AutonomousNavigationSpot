#!/usr/bin/env python3
"""
Controllo pre-volo: verifica che i file sulla macchina del robot siano allineati fra
loro PRIMA di lanciare una missione.

Motivo: il 2026-10-05 alle 15:55 una missione si e' interrotta subito dopo la
costruzione del grafo con

    AttributeError: 'GlobalGrid' object has no attribute 'is_known'

perche' `easy_walk.py` era aggiornato ma `spotGrid.py` era una copia precedente. Il
robot era gia' acceso e in piedi. Non c'e' stato alcun rischio -- il crash e' avvenuto
prima di qualunque movimento -- ma e' tempo perso, e in esterno sarebbe stato tempo
perso a batteria accesa in mezzo a un campo.

Il controllo e' PURAMENTE STATICO: legge i file, non importa nulla, non tocca il robot,
non serve l'SDK bosdyn. Si puo' lanciare in qualunque momento.

    python3 preflight.py                 # nella cartella del progetto
    python3 preflight.py /percorso/cartella

Uscita 0 = tutto allineato. Uscita 1 = manca qualcosa, NON lanciare la missione.
"""

import ast
import os
import sys

# Per ogni file: i simboli che DEVONO esistere perche' gli altri file li usano.
# Aggiornare questa tabella quando si aggiunge una dipendenza fra moduli.
REQUIRED = {
    "spotGrid.py": {
        "costanti": [
            "OBSTACLE_THRESHOLD", "SLOPE_THRESHOLD", "ROUGH_THRESHOLD",
            "SLOPE_BASELINE_M", "LATERAL_SLICE_HALF_WIDTH_M", "MIN_ARC_SAMPLE_FRACTION",
            "NEAR_THRESHOLD_MARGIN_FRACTION", "LATERAL_SLOPE_COST_MULTIPLIER",
            "GROUND_HEIGHT_TOLERANCE_M", "GROUND_REF_RADIUS_M",
            "GROUND_REF_FALLBACK_RADIUS_M", "GROUND_REF_MIN_CELLS",
            "GAIT_VEL_PLAIN", "GAIT_VEL_MODERATE", "GAIT_VEL_HARD",          # passo 2
            "GAIT_ANG_PLAIN", "GAIT_ANG_MODERATE", "GAIT_ANG_HARD",
            "FRONTIER_CLEARANCE_M", "FRONTIER_MIN_CLEARANCE_M", "FRONTIER_ESCAPE_DIST_M",   # passo 3
            "FRONTIER_BODY_HALF_LENGTH_M", "FRONTIER_MARGIN_M", "FRONTIER_SAMPLE_STEP_M",
            "FRONTIER_ROBOT_RADIUS_M", "FRONTIER_SLOPE_MIN_LENGTH_M", "PRM_EDGE_SAFETY_MARGIN_M",
            "GLOBAL_OCC_CONFIRM_OBS", "GLOBAL_OCC_MAX_COUNT", "GLOBAL_OCC_COARSE_CELLS",       # passo 6
            "ROBOT_HALF_LENGTH_M", "ROBOT_HALF_WIDTH_M", "ROBOT_CLEARANCE_AIR_M",               # passo 4
            "ROTATION_CLEARANCE_M", "ROTATION_CHECK_MIN_DYAW_DEG",
            "UNWRITTEN_MATCH_FRACTION", "UNWRITTEN_MATCH_FALLBACK_M",                          # celle mai scritte
        ],
        "funzioni": ["body_extents",
            "compute_arc_slope_profile", "compute_arc_sample_coverage",
            "_gather_heights", "_baseline_gradient", "_slope_profile_from_sampler",
        ],
        "metodi": [
            ("GlobalGrid", "is_occupied"),
            ("GlobalGrid", "is_known"),            # usato da find_best_point_in_cell
            ("GlobalGrid", "_set_state"),          # passo 6
            ("GlobalGrid", "_rebuild_coarse_index"),
            ("GlobalGrid", "mark_hazard"),         # cadute
            ("GlobalTerrainGrid", "compute_arc_slope_profile"),
            ("LocalGrid", "compute_robot_footprint_mask"),
            ("LocalGrid", "correct_terrain"),
            ("LocalGrid", "_cells_above_ground"),    # filtro altezza suolo (passo 1)
            ("LocalGrid", "_cells_never_written"),   # celle mai scritte (sopra o sotto il suolo)
            ("LocalGrid", "_fetch_layers"),          # revisione 2026-10-06 sera
            ("LocalGrid", "_layer_geometry"),
            ("GlobalGrid", "segment_hits_hazard"),
            ("LocalGrid", "fuse_obstacle_mask"),
        ],
    },
    "prm_graph.py": {
        "costanti": ["OCCUPANCY_SAMPLES_PER_M", "OCCUPANCY_SAMPLES_MIN", "STOP_NODE_MIN_EDGE_M",
                     "STOP_NODE_MAP_IGNORE_M"],
        "funzioni": [],
        "metodi": [
            ("PRM", "build_graph"),
            ("PRM", "refresh_local_edge_weights"),
            ("PRM", "sample_terrain_between_points"),
            ("PRM", "_compute_edge_weight"),
            ("PRM", "set_global_terrain_map"),
            ("PRM", "add_stop_node"),             # passo 5
            ("PRM", "connect_node"),
            ("PRM", "_evaluate_and_add_edge"),
            ("PRM", "set_robot_node"),            # cadute
            ("PRM", "_occupancy_blocked"),
            ("PRM", "_global_slope"),             # revisione 2026-10-06 sera
            ("PRM", "_sync_global_slope_cache"),
            ("PRM", "_traversed_edge_ok"),
        ],
    },
    "arcVerification.py": {
        "costanti": [],
        "funzioni": ["arc_visible_fraction", "is_arc_in_fov", "verify_arc_safety",
                     "make_grid_snapshot", "compute_safe_frontier", "allowed_advance",   # passo 3
                     "point_along", "describe_frontier"],
        "metodi": [("ArcVerificationTracker", "update_path"),
                   ("ArcVerificationTracker", "get_frontier"),
                   ("ArcVerificationTracker", "clear_path")],
    },
    "movements.py": {
        "costanti": ["DISABLE_FOOT_OBSTACLE_AVOIDANCE_DEFAULT",
                     "DEFAULT_MAX_LINEAR_VEL_MPS", "DEFAULT_MAX_ANGULAR_VEL_RPS",   # passo 2
                     "RETREAT_MAX_LINEAR_VEL_MPS",
                     "FALL_TILT_DEG", "UPRIGHT_TILT_DEG", "SELF_RIGHT_TIMEOUT_S",       # cadute
                     "STAND_TIMEOUT_S", "SELF_RIGHT_MAX_ATTEMPTS",
                     "MOVE_TIMEOUT_FACTOR", "MOVE_TIMEOUT_MARGIN_S", "MOVE_MAX_RPC_ERRORS"],
        "funzioni": ["relative_move", "stop_robot", "move_backward",
                     "move_to_world_point_without_turning", "_build_mobility_params",
                     "_velocity_limit", "check_fall", "recover_from_fall", "_roll_pitch_deg"],
        "metodi": [],
    },
    "easy_walk.py": {
        "costanti": ["MIN_PARTIAL_STEP_M", "FRONTIER_BLOCK_CONFIRM_SCANS", "FRONTIER_RESCAN_WAIT_S",
                     "MAX_BLOCK_REPLANS", "FRONTIER_ABORT_CONFIRM", "FRONTIER_ABORT_TOLERANCE_M",
                     "GOAL_REACHED_TOLERANCE_M", "MAX_LOOP_ITERATIONS",
                     "SAVE_FIGURES_DURING_MISSION", "SAVE_SCAN_BUNDLES",                 # passi 7-8
                     "ROTATION_ROOM_SEARCH_M", "ROTATION_ROOM_DIRECTIONS_DEG",           # passo 4
                     "SHORTCUT_MAX_DIST_M", "MAX_MANEUVERS_SAME_EDGE",
                     "FALL_HAZARD_RADIUS_M",                                             # cadute
                     "RETREAT_MAX_SEGMENTS", "RETREAT_MIN_SEGMENT_M", "RETREAT_AFTER_ABORT_M",
                     "RETREAT_CONTINUITY_M", "RETREAT_MAX_SEGMENT_M",                    # revisione
                     "GRAPHNAV_TIMEOUT_S", "GRAPHNAV_MAX_LINEAR_VEL_MPS", "GRAPHNAV_MAX_ANGULAR_VEL_RPS",
                     "MAX_FALLS_PER_MISSION"],
        "funzioni": ["find_best_point_in_cell", "retreat_along_traveled_path",
                     "navigate_to", "attempt_enter_cell_from_position",
                     "_log_footprint_diagnostic", "_log_ground_filter", "decide_next_move",
                     "rotation_plan", "find_rotation_room", "shortcut_index",
                     "_check_and_recover_fall", "_raw_zero", "_raw_scale",
                     "_mission_z0", "_set_mission_z0", "_mask_for_global_map",
                     "_graphnav_navigate_to", "_graphnav_return_to_start", "_append_csv"],
        "metodi": [],
    },
}

# Firme che devono combaciare fra chiamante e definizione: (file, funzione, parametro).
REQUIRED_PARAMS = [
    ("movements.py", "relative_move", "should_abort"),
    ("movements.py", "relative_move", "poll_period_s"),
    ("movements.py", "relative_move", "disable_foot_obstacle_avoidance"),
    ("movements.py", "relative_move", "max_linear_vel"),                # passo 2
    ("movements.py", "_build_mobility_params", "max_linear_vel"),       # passo 2
    ("easy_walk.py", "navigate_to", "should_abort"),
    ("easy_walk.py", "find_best_point_in_cell", "global_map"),
    ("spotGrid.py", "compute_arc_slope_profile", "baseline_m"),
    ("spotGrid.py", "compute_arc_slope_profile", "lateral_half_width_m"),
    ("spotGrid.py", "correct_terrain", "cell_size"),                    # passo 1
    ("spotGrid.py", "compute_gradient_and_roughness", "is_valid"),      # passo 1
    ("arcVerification.py", "compute_safe_frontier", "body"),            # passo 4
    ("easy_walk.py", "navigate_to", "keep_heading"),                    # passo 4
    ("prm_graph.py", "add_stop_node", "is_robot"),                      # cadute
    ("easy_walk.py", "shortcut_index", "global_map"),
    ("spotGrid.py", "correct_terrain", "unwritten_value"),              # celle mai scritte
    ("spotGrid.py", "correct_terrain", "return_unwritten"),
    ("arcVerification.py", "make_grid_snapshot", "unwritten"),
    ("spotGrid.py", "compute_arc_slope_profile", "valid_2d"),           # pendenza su celle attendibili
    ("spotGrid.py", "_gather_heights", "valid_2d"),
    ("prm_graph.py", "update_local_grid_data", "valid_2d"),
    ("prm_graph.py", "compute_path_cost", "allow_direct_first_hop"),   # revisione 2026-10-06 sera
    ("spotGrid.py", "return_local_grid", "max_attempts"),
]


def _parse(path):
    with open(path, encoding="utf-8") as f:
        return ast.parse(f.read(), filename=path)


def _module_names(tree):
    """Nomi definiti a livello di modulo: costanti, funzioni, classi."""
    consts, funcs, classes = set(), set(), {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    consts.add(t.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.add(node.name)
        elif isinstance(node, ast.ClassDef):
            classes[node.name] = {n.name for n in node.body
                                  if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    return consts, funcs, classes


def _func_params(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            a = node.args
            names = [p.arg for p in list(a.posonlyargs) + list(a.args) + list(a.kwonlyargs)]
            if a.vararg:
                names.append(a.vararg.arg)
            if a.kwarg:
                names.append(a.kwarg.arg)
            return names
    return None


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    print(f"Controllo pre-volo su: {root}\n")

    problems = []
    trees = {}

    for fname in REQUIRED:
        path = os.path.join(root, fname)
        if not os.path.isfile(path):
            problems.append(f"{fname}: FILE MANCANTE")
            continue
        try:
            trees[fname] = _parse(path)
        except SyntaxError as e:
            problems.append(f"{fname}: ERRORE DI SINTASSI riga {e.lineno}: {e.msg}")

    for fname, want in REQUIRED.items():
        if fname not in trees:
            continue
        consts, funcs, classes = _module_names(trees[fname])
        missing = []
        for c in want["costanti"]:
            if c not in consts:
                missing.append(f"costante {c}")
        for f in want["funzioni"]:
            if f not in funcs:
                missing.append(f"funzione {f}()")
        for cls, meth in want["metodi"]:
            if cls not in classes:
                missing.append(f"classe {cls}")
            elif meth not in classes[cls]:
                missing.append(f"metodo {cls}.{meth}()")
        if missing:
            problems.append(f"{fname}: copia VECCHIA -- mancano: " + ", ".join(missing))
        else:
            print(f"  OK   {fname}")

    if "movements.py" in trees:
        _, _, classes = _module_names(trees["movements.py"])
        if "RobotFallenError" not in classes:
            problems.append("movements.py: copia VECCHIA -- manca la classe RobotFallenError")

    for fname, func, param in REQUIRED_PARAMS:
        if fname not in trees:
            continue
        params = _func_params(trees[fname], func)
        if params is None:
            problems.append(f"{fname}: funzione {func}() non trovata")
        elif param not in params:
            problems.append(f"{fname}: {func}() non accetta '{param}' -- copia vecchia")

    print()
    if problems:
        print("=" * 70)
        print("NON LANCIARE LA MISSIONE. Problemi trovati:")
        print("=" * 70)
        for p in problems:
            print(f"  - {p}")
        print("\nRicopia i file segnalati dalla versione piu' recente.")
        return 1

    print("=" * 70)
    print("Tutti i file sono allineati fra loro.")
    print("=" * 70)
    print("NB: questo controllo verifica solo che i moduli siano coerenti.")
    print("    Non dice nulla sul comportamento del robot sul campo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())