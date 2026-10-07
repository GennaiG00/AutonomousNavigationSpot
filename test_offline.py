import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial import cKDTree

# Importiamo i moduli dell'SDK locale
import spotGrid
import environmentMap

import numpy as np
import matplotlib.pyplot as plt
import os
import spotGrid  # Pipeline di calcolo geometrico
import PRM       # Il tuo modulo di Path Planning (Probabilistic RoadMap)


# =====================================================================
# 1. FUNZIONI INTEGRATE DA EASY_WALK (VERSIONE OTTIMIZZATA CON KDTREE)
# =====================================================================

def check_line_of_sight(x1, y1, x2, y2, kdtree, cells_fused, num_points=20):
    """
    Verifica se la linea retta tra (x1, y1) e (x2, y2) è libera da ostacoli.
    Usa il KDTree pre-costruito per lookups spaziali ultra-rapidi.
    """
    # Genera punti campionati lungo la linea tra il punto A e il punto B
    t = np.linspace(0, 1, num_points)
    line_x = x1 + t * (x2 - x1)
    line_y = y1 + t * (y2 - y1)
    line_pts = np.vstack([line_x, line_y]).T

    # Interroga il KDTree per TUTTI i punti della linea contemporaneamente
    _, indices = kdtree.query(line_pts)

    # Se anche solo un punto interseca un ostacolo (cells_fused == 0), la linea è bloccata
    if np.any(cells_fused[indices] == 0):
        return False  # Ostacolo rilevato

    return True  # Linea di vista libera


def sample_cell_points(env, cell_row, cell_col, num_samples=20):
    """
    Campiona punti casuali all'interno di una macro-cella della mappa globale.
    """
    world_pos = env.get_world_position_from_cell(cell_row, cell_col)
    if world_pos is None:
        return []

    cell_center_x, cell_center_y = world_pos
    half_size = env.cell_size / 2.0

    samples = []
    for _ in range(num_samples):
        # Offset casuale (all'80% della dimensione per evitare i bordi netti)
        offset_x = np.random.uniform(-half_size * 0.8, half_size * 0.8)
        offset_y = np.random.uniform(-half_size * 0.8, half_size * 0.8)

        # Rotazione dell'offset nel frame del mondo (se origin_yaw != 0)
        cos_yaw = np.cos(env.origin_yaw)
        sin_yaw = np.sin(env.origin_yaw)

        world_offset_x = offset_x * cos_yaw - offset_y * sin_yaw
        world_offset_y = offset_x * sin_yaw + offset_y * cos_yaw

        samples.append((cell_center_x + world_offset_x, cell_center_y + world_offset_y))

    return samples


def find_best_point_in_cell(robot_x, robot_y, env, cell_row, cell_col, kdtree, cells_fused):
    """
    Campiona 20 punti in una cella e trova quello con traiettoria libera
    più vicino al centro della cella stessa.
    """
    sampled_points = sample_cell_points(env, cell_row, cell_col, num_samples=20)
    if not sampled_points:
        return None, None, [], []

    cell_center = env.get_world_position_from_cell(cell_row, cell_col)
    if cell_center is None:
        return None, None, [], []

    cell_center_x, cell_center_y = cell_center
    valid_samples = []
    rejected_samples = []

    # Controllo di ogni candidato tramite la linea di vista accelerata
    for sample_x, sample_y in sampled_points:
        if check_line_of_sight(robot_x, robot_y, sample_x, sample_y, kdtree, cells_fused):
            valid_samples.append((sample_x, sample_y))
        else:
            rejected_samples.append((sample_x, sample_y))

    if not valid_samples:
        print(f"[WARNING] Nessun percorso libero trovato verso la cella ({cell_row},{cell_col})")
        return None, None, valid_samples, rejected_samples

    # Seleziona il punto valido più vicino al centro geometrico della cella
    best_point = None
    min_distance = float('inf')

    for sample_x, sample_y in valid_samples:
        dist = np.sqrt((sample_x - cell_center_x) ** 2 + (sample_y - cell_center_y) ** 2)
        if dist < min_distance:
            min_distance = dist
            best_point = (sample_x, sample_y)

    return best_point[0], best_point[1], valid_samples, rejected_samples


def draw_explored_sides(ax, cell_x, cell_y, half_size, sides_status, cos_yaw, sin_yaw):
    """ Disegna i bordi esplorati sulle celle """
    if sides_status == 0b0000:
        return
    edge_inset = 0.05
    # Nord (Bit 3)
    if sides_status & 0b1000:
        ax.plot([cell_x - half_size, cell_x + half_size], [cell_y + half_size, cell_y + half_size], 'r-', linewidth=4,
                alpha=0.8)
    # Est (Bit 2)
    if sides_status & 0b0100:
        ax.plot([cell_x + half_size, cell_x + half_size], [cell_y - half_size, cell_y + half_size], 'r-', linewidth=4,
                alpha=0.8)
    # Sud (Bit 1)
    if sides_status & 0b0010:
        ax.plot([cell_x - half_size, cell_x + half_size], [cell_y - half_size, cell_y - half_size], 'r-', linewidth=4,
                alpha=0.8)
    # Ovest (Bit 0)
    if sides_status & 0b0001:
        ax.plot([cell_x - half_size, cell_x - half_size], [cell_y - half_size, cell_y + half_size], 'r-', linewidth=4,
                alpha=0.8)


def visualize_grid_with_candidates(pts, cells_fused, color, robot_x, robot_y,
                                   candidates, chosen_point, iteration, env=None):
    """
    Visualizzazione della griglia locale, dei punti candidati (validi/scartati)
    e del target finale scelto allineato con la mappa globale.
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot dei punti della local grid di Spot
    x = pts[:, 0]
    y = pts[:, 1]
    colors_norm = color.astype(np.float32) / 255.0
    ax.scatter(x, y, c=colors_norm, s=4, alpha=0.5, label='Local Grid Terrain (Rosso=Ostacolo, Blu=Libero)')

    local_x_min, local_x_max = x.min(), x.max()
    local_y_min, local_y_max = y.min(), y.max()

    # Sovrapposizione della griglia globale macro dell'EnvironmentMap
    if env is not None:
        for row in range(env.rows):
            for col in range(env.cols):
                world_pos = env.get_world_position_from_cell(row, col)
                if world_pos is None:
                    continue
                cell_x, cell_y = world_pos

                # Mostra solo le macro-celle che si sovrappongono alla local grid visibile
                margin = env.cell_size
                if not (local_x_min - margin <= cell_x <= local_x_max + margin and
                        local_y_min - margin <= cell_y <= local_y_max + margin):
                    continue

                half_size = env.cell_size / 2.0
                grid_corners = [
                    (-half_size, -half_size), (half_size, -half_size),
                    (half_size, half_size), (-half_size, half_size)
                ]

                cos_yaw, sin_yaw = np.cos(env.origin_yaw), np.sin(env.origin_yaw)
                world_corners = []
                for gx, gy in grid_corners:
                    wx = cell_x + (gx * cos_yaw - gy * sin_yaw)
                    wy = cell_y + (gx * sin_yaw + gy * cos_yaw)
                    world_corners.append((wx, wy))

                cell_status, sides_status = env.get_cell_status(row, col)
                if cell_status == 1:
                    facecol, edgecol, alpha = 'lightgreen', 'darkgreen', 0.2
                elif cell_status == -1:
                    facecol, edgecol, alpha = 'lightcoral', 'darkred', 0.3
                else:
                    facecol, edgecol, alpha = 'none', 'gray', 0.5

                rect = patches.Polygon(world_corners, linewidth=1.5, edgecolor=edgecol,
                                       facecolor=facecol, alpha=alpha, linestyle='--' if cell_status == 0 else '-',
                                       zorder=2)
                ax.add_patch(rect)

                # Etichetta di riga, colonna sulla macro-cella
                ax.text(cell_x, cell_y, f'{row},{col}', ha='center', va='center',
                        fontsize=8, color='black', weight='bold', zorder=3,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

                if cell_status != 1 and sides_status != 0b0000:
                    draw_explored_sides(ax, cell_x, cell_y, half_size, sides_status, cos_yaw, sin_yaw)

    # Disegna i candidati scartati (X rossa)
    if 'rejected' in candidates and candidates['rejected']:
        rj = np.array(candidates['rejected'])
        ax.scatter(rj[:, 0], rj[:, 1], c='red', marker='x', s=100, linewidths=2.5, zorder=5, label='Scartati (No LOS)')

    # Disegna i candidati validi (Cerchio giallo)
    if 'valid' in candidates and candidates['valid']:
        vd = np.array(candidates['valid'])
        ax.scatter(vd[:, 0], vd[:, 1], c='yellow', edgecolors='orange', marker='o', s=80, linewidths=2, zorder=5,
                   label='Validi (LOS OK)')

    # Disegna il punto migliore scelto (Stella verde)
    if chosen_point is not None:
        ax.plot(chosen_point[0], chosen_point[1], 'g*', markersize=20, markeredgewidth=2, label='Target Scelto',
                zorder=6)

        # Linea dal robot al target scelto
        ax.plot([robot_x, chosen_point[0]], [robot_y, chosen_point[1]], 'g--', linewidth=2, alpha=0.8)
        target_dist = np.sqrt((chosen_point[0] - robot_x) ** 2 + (chosen_point[1] - robot_y) ** 2)
        ax.text((robot_x + chosen_point[0]) / 2, (robot_y + chosen_point[1]) / 2, f'{target_dist:.2f}m',
                fontsize=10, color='darkgreen', weight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # Posizione attuale del Robot (Cerchio blu grande)
    ax.plot(robot_x, robot_y, 'bo', markersize=15, label='Robot (Centro Local Grid)', zorder=7)

    ax.set_xlim(local_x_min - 0.5, local_x_max + 0.5)
    ax.set_ylim(local_y_min - 0.5, local_y_max + 0.5)
    ax.set_xlabel('X [m] (VISION Frame)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y [m] (VISION Frame)', fontsize=12, fontweight='bold')
    ax.set_title(f'Test Offline - Iterazione {iteration}: Verifica Line Of Sight con KDTree', fontsize=12,
                 fontweight='bold')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


# =====================================================================
# 2. FUNZIONE PRINCIPALE (MAIN) DI TEST OFFLINE
# =====================================================================

def main():
    # 1. PARAMETRI DELLA GRIGLIA LOGGATA DA SPOT
    num_x = 128
    num_y = 128
    cell_size = 0.03  # 3 centimetri per cella

    percorso_file = os.path.expanduser('~/Downloads/local_grid_log (1).txt')
    print(f"Tentativo di caricamento da: {percorso_file}")

    try:
        # Carichiamo le coordinate reali del mondo X, Y e l'altezza Z dal log
        data = np.loadtxt(percorso_file, skiprows=7, delimiter=',', max_rows=num_x * num_y)
        print(f"[OK] Dati caricati correttamente! Trovate {len(data)} righe.")

        pts = data[:, :2]  # Colonne 0 e 1: coordinate spaziali X, Y nel mondo VISION
        terrain_values = data[:, 2]  # Colonna 2: altitudini Z

    except Exception as e:
        print(f"[ERRORE FATALE] Impossibile leggere il file log: {e}")
        return

    # 2. MAPPA DI VALIDITÀ E ZONA CIECA SIMULATA
    valid_values = np.ones_like(terrain_values)
    valid_2d = valid_values.reshape((num_x, num_y))
    valid_2d[80:90, 80:90] = 0.0  # Applichiamo la zona cieca di test (10x10 celle)
    valid_values = valid_2d.ravel()

    # 3. PIPELINE GEOMETRICA (PENDENZA E RUGOSITÀ)
    print("Calcolo Pendenza e Rugosità...")
    grad_values, rough_values = spotGrid.compute_gradient_and_roughness(
        terrain_values, valid_values, num_x, num_y, cell_size
    )

    grad_2d = grad_values.reshape((num_x, num_y))
    rough_2d = rough_values.reshape((num_x, num_y))

    # Calcolo della mappa dei costi di transito (Costmap)
    terrain_costs_2d = (grad_2d * 10.0) + (rough_2d * 20.0)
    terrain_costs_1d = terrain_costs_2d.ravel()

    # 4. FUSIONE STRATI ED ESTRAZIONE OSTACOLI ASSOLUTI (cells_fused)
    cells_fused = np.ones_like(terrain_values)
    cells_fused[grad_values > 0.3] = 0  # Alzata da 0.35 a 0.50 (più tolleranza alla pendenza)
    cells_fused[rough_values > 0.3] = 0  # Alzata da 0.05 a 0.12 (più tolleranza alla rugosità)
    cells_fused[valid_values == 0] = 0  # Rimane il veto sui punti ciechi

    # 5. SIMULAZIONE DI PATH PLANNING PRM (OFFLINE)
    print("\n--- AVVIO PIANIFICAZIONE PRM OFFLINE ---")

    # Scegliamo un punto di partenza (Start) e uno di arrivo (Goal) sensati usando le coordinate reali.
    # Impostiamo lo start esattamente al centro della griglia locale estratta
    #center_idx = (num_x // 2) * num_y + (num_y // 2)
    robot_x, robot_y = -14 , 22

    # Spostiamo il goal a sinistra (-0.6) e in alto (+0.8) nel corridoio pulito
    goal_x, goal_y = -12 , 22.5

    print(f"[SIM] Posizione START Robot: X={robot_x:.3f}, Y={robot_y:.3f}")
    print(f"[SIM] Posizione GOAL Target: X={goal_x:.3f}, Y={goal_y:.3f}")

    # Costruiamo la rete stradale probabilistica (PRM)
    nodes, edges, start_idx, goal_idx, node_costs = PRM.build_prm(
        pts=pts,
        cells_fused=cells_fused,
        terrain_costs=terrain_costs_1d,
        robot_x=robot_x,
        robot_y=robot_y,
        goal_x=goal_x,
        goal_y=goal_y,
        radius=4.0,  # Copre l'intera estensione locale del log
        n_samples=200,  # Numero di nodi stradali da campionare nello spazio libero
        k_neighbors=8  # Connessioni massime per nodo
    )

    path_indices = None
    if nodes is not None:
        print(f"[PRM] Mappa stradale creata con {len(nodes)} nodi e {len(edges)} archi candidati.")
        # Lanciamo Dijkstra considerando la distanza geometrica moltiplicata per il costo del terreno
        print("[PRM] Ricerca del percorso ottimo con Dijkstra pesato...")
        path_indices = PRM.plan_prm_dijkstra(nodes, edges, start_idx, goal_idx, node_costs)

        if path_indices is not None:
            print(f"[OK] Percorso calcolato con successo! Trovati {len(path_indices)} waypoint ottimizzati.")
        else:
            print("[ATTENZIONE] Dijkstra non ha trovato percorsi liberi. Il Goal è isolato o circondato da ostacoli.")
    else:
        print("[ERRORE] Impossibile generare il PRM. Spazio attorno al robot completamente ostruito.")

    # 6. VISUALIZZAZIONE GRAFICA AD ALTA PRECISIONE (CON OVERLAY IN METRI)
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # Definiamo i confini in metri per mappare correttamente imshow con scatter/plot
    extent_m = [pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max()]

    # Riquadro A: Elevazione del Terreno
    im0 = axs[0].imshow(terrain_values.reshape(num_x, num_y), cmap='terrain', origin='lower', extent=extent_m)
    axs[0].set_title('1. Mappa delle Altezze (Z)')
    axs[0].set_xlabel('Mondo X (m)')
    axs[0].set_ylabel('Mondo Y (m)')
    fig.colorbar(im0, ax=axs[0], label='Metri')

    # Riquadro B: Costmap di navigazione
    im1 = axs[1].imshow(terrain_costs_2d, cmap='viridis', origin='lower', extent=extent_m)
    axs[1].set_title('2. Costmap Finale (Ostacoli Soft)')
    axs[1].set_xlabel('Mondo X (m)')
    fig.colorbar(im1, ax=axs[1], label='Intensità Costo')

    # Riquadro C: Overlay Algoritmo PRM e Percorso Calcolato
    cells_fused_2d = cells_fused.reshape((num_x, num_y))
    # Sfondo binario: Bianco = Libero, Grigio = Ostacolo Duro
    axs[2].imshow(cells_fused_2d, cmap='gray', origin='lower', alpha=0.5, extent=extent_m)
    axs[2].set_title('3. Overlay PRM Graph & Path Calcolato')
    axs[2].set_xlabel('Mondo X (m)')

    if nodes is not None:
        # Disegniamo i nodi della roadmap generati casualmente nello spazio sicuro (puntini azzurri)
        axs[2].scatter(nodes[2:, 0], nodes[2:, 1], color='cyan', s=15, alpha=0.6, label='Nodi Roadmap PRM')

        # Disegniamo i punti di START e GOAL reali della simulazione
        axs[2].scatter(robot_x, robot_y, color='lime', s=120, marker='o', edgecolors='black', label='START (Robot)',
                       zorder=5)
        axs[2].scatter(goal_x, goal_y, color='red', s=120, marker='X', edgecolors='black', label='GOAL (Target)',
                       zorder=5)

        # Se Dijkstra ha trovato una soluzione, disegna la linea del percorso in rosso spesso
        if path_indices is not None:
            path_pts = nodes[path_indices]
            axs[2].plot(path_pts[:, 0], path_pts[:, 1], color='red', linewidth=3, marker='o', markersize=5,
                        label='Percorso Ottimo')

    axs[2].grid(True, linestyle='--', alpha=0.5)
    axs[2].legend(loc='upper right')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()