import os
import numpy as np
import matplotlib.pyplot as plt

# 1. Specifica il percorso del tuo file .npy
file_path = "/home/andrea/spot_data/MissionMap/Mission_16-07-2026_15-27-35/iteration_0_cell_0_1_terrain.npy"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"Impossibile trovare il file al percorso: {file_path}")

# 2. Carica i dati
data = np.load(file_path, allow_pickle=True)

print("=== METADATI DELLA MATRICE ===")
print(f"File caricato:        {os.path.basename(file_path)}")
print(f"Tipo di dato (dtype): {data.dtype}")
print(f"Formato (shape):      {data.shape}")
print(f"Valore minimo:        {data.min():.4f}")
print(f"Valore massimo:       {data.max():.4f}")
print("=============================\n")

# Verifichiamo che sia effettivamente una matrice 2D
if len(data.shape) == 2:
    num_rows, num_cols = data.shape

    # =========================================================================
    # PARTE 1: STAMPA DELLA SOTTO-MATRICE NEL TERMINALE (Opzione 2)
    # =========================================================================
    # Configura NumPy per non troncare mai l'output e mostrare i decimali in modo leggibile
    np.set_printoptions(threshold=np.inf, precision=4, suppress=True, linewidth=200)

    # Definiamo una finestra di window_size x window_size celle al centro esatto della mappa
    window_size = 20
    center_row, center_col = num_rows // 2, num_cols // 2

    row_start = max(0, center_row - window_size // 2)
    row_end = min(num_rows, center_row + window_size // 2)
    col_start = max(0, center_col - window_size // 2)
    col_end = min(num_cols, center_col + window_size // 2)

    print(
        f"🔎 [TERMINALE] Zoom numerico al centro della mappa (Righe {row_start}:{row_end}, Colonne {col_start}:{col_end}):")
    print(data[row_start:row_end, col_start:col_end])
    print("\n=========================================================================\n")

    # =========================================================================
    # PARTE 2: VISUALIZZAZIONE GRAFICA CON RECOVERY DEI DATI AL PASSAGGIO MOUSE (Opzione 1)
    # =========================================================================
    print("🎨 [GRAFICO] Apertura della mappa interattiva...")
    print("👉 Muovi il mouse sopra l'immagine: vedrai i valori esatti in tempo reale in basso a destra!")

    fig, ax = plt.subplots(figsize=(10, 8))

    # origin='lower' mette la riga 0 in basso (coerente con i sistemi di coordinate robotici)
    im = ax.imshow(data, cmap='terrain', origin='lower')
    fig.colorbar(im, label='Valore Terrain / Altezza (m)')


    # Funzione lambda per aggiornare le coordinate mostrate sulla barra di stato di Matplotlib
    def format_coord(x, y):
        col = int(x + 0.5)
        row = int(y + 0.5)
        if 0 <= col < num_cols and 0 <= row < num_rows:
            val = data[row, col]
            return f"Col (X): {col:<3} | Row (Y): {row:<3} | Valore (Z): {val:.4f}"
        return ""


    ax.format_coord = format_coord

    plt.title(f"Mappa Interattiva: {os.path.basename(file_path)}\n(Valori live in basso a destra nella finestra)")
    plt.xlabel("Colonne (X)")
    plt.ylabel("Righe (Y)")
    plt.grid(True, which='both', color='white', linestyle='--', linewidth=0.5, alpha=0.5)

    plt.show()

else:
    print(f"[ATTENZIONE] Il file caricato ha una dimensione di shape pari a {len(data.shape)}.")
    print("Questo script combinato è progettato specificamente per matrici 2D (es. 128x128).")