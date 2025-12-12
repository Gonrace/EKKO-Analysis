# ==============================================================================
# EKKO ANALYSIS SCRIPT V3.4 (VISUAL UPGRADE)
# ==============================================================================
# Nouveautés V3.4 :
# - Amélioration de la lisibilité des graphiques (espacement, polices).
# - Ajout d'un marqueur "PIC MAX" (étoile rouge) sur la courbe d'énergie.
# - Ligne de référence pour le volume fort (-10dB).
# - Code nettoyé pour Python 3.10+ (compatible 3.14).
# ==============================================================================

import os
import sys
import json
import zipfile
import shutil
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# Charge les variables d'environnement
load_dotenv()

# Vérification des dépendances
try:
    from tqdm import tqdm
except ImportError:
    print("ERREUR: Module 'tqdm' manquant. Installez-le avec: pip install tqdm")
    sys.exit()

from acrcloud.recognizer import ACRCloudRecognizer

# --- 1. CONFIGURATION ---
config = {
    'host': os.getenv('ACR_HOST'),
    'access_key': os.getenv('ACR_ACCESS_KEY'),
    'access_secret': os.getenv('ACR_ACCESS_SECRET'),
    'timeout': 10
}

INPUT_FOLDER = "input"
OUTPUT_FOLDER = "output"

if not os.path.exists(INPUT_FOLDER): os.makedirs(INPUT_FOLDER)
if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)

# --- 2. PARAMÈTRES D'ANALYSE ---
slice_duration_seconds = 12
step_seconds = 10
SILENCE_MARKER = "--- (Silence ou Bruit) ---"
MAX_GAP_TOLERANCE_STEPS = 4 

# --- 3. PARAMÈTRES DE CLASSIFICATION & SCORE ---
STATIONARY_THRESHOLD = 0.001
STATE_WINDOW_SECONDS = 3
GYRO_WEIGHT = 15.0
YAW_CHANGE_WEIGHT = 50.0

# ==============================================================================
# --- FONCTIONS UTILITAIRES ---
# ==============================================================================

def choose_zip_file():
    print(f"--- [ÉTAPE 1/7] Sélection du fichier dans '{INPUT_FOLDER}/' ---")
    zip_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.zip')]
    
    if not zip_files:
        print(f"ERREUR : Aucun fichier .zip trouvé dans le dossier '{INPUT_FOLDER}'.")
        print("-> Veuillez déplacer vos enregistrements dans ce dossier.")
        sys.exit()
    
    print("Fichiers disponibles :")
    for i, filename in enumerate(zip_files):
        print(f"  [{i + 1}] {filename}")
    
    while True:
        try:
            choice = int(input(f"\nChoix (1-{len(zip_files)}): ")) - 1
            if 0 <= choice < len(zip_files):
                return os.path.join(INPUT_FOLDER, zip_files[choice])
        except ValueError: pass

def create_output_directory(zip_filepath):
    filename = os.path.basename(zip_filepath)
    base_name = os.path.splitext(filename)[0]
    i = 1
    while True:
        folder_name = f"ANALYSE_{i}_{base_name}"
        output_dir = os.path.join(OUTPUT_FOLDER, folder_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"\n--- [NOUVEAU] Dossier de sortie : {output_dir} ---")
            return output_dir
        i += 1

def extract_zip_data(zip_filename):
    print(f"\n--- [ÉTAPE 2/7] Extraction ---")
    temp_folder = "temp_extraction"
    if os.path.exists(temp_folder): shutil.rmtree(temp_folder)
    os.makedirs(temp_folder)
    try:
        with zipfile.ZipFile(zip_filename, 'r') as zf: zf.extractall(temp_folder)
    except Exception as e: print(f"ERREUR ZIP : {e}"); sys.exit()
    
    paths = {'audio': None, 'sensor': None}
    for root, _, files in os.walk(temp_folder):
        for file in files:
            if file.lower().endswith(('.wav', '.m4a', '.mp3')): paths['audio'] = os.path.join(root, file)
            elif file.endswith('.csv'): paths['sensor'] = os.path.join(root, file)
    if not all(paths.values()): print("ERREUR : Fichiers manquants dans le ZIP."); shutil.rmtree(temp_folder); sys.exit()
    return paths['audio'], paths['sensor'], temp_folder

# ==============================================================================
# --- ANALYSE SPECTRALE (BPM MOUVEMENT) ---
# ==============================================================================

def calculate_motion_bpm(df_chunk):
    if len(df_chunk) < 50: return 0
    signal = df_chunk['accel_mag'].values
    signal = signal - np.mean(signal)
    try:
        duration = df_chunk['timestamp'].iloc[-1] - df_chunk['timestamp'].iloc[0]
        fs = len(df_chunk) / duration
    except: return 0
    
    fft_spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), d=1./fs)
    mask = (freqs > 0.5) & (freqs < 3.0)
    target_freqs = freqs[mask]
    target_spectrum = np.abs(fft_spectrum[mask])
    
    if len(target_spectrum) == 0: return 0
    peak_freq = target_freqs[np.argmax(target_spectrum)]
    return peak_freq * 60

# ==============================================================================
# --- ANALYSE AUDIO & NETTOYAGE ---
# ==============================================================================

def analyze_audio_timeline(audio_path, total_duration):
    print("\n--- [ÉTAPE 3/7] Analyse Audio (ACRCloud)... ---")
    recognizer = ACRCloudRecognizer(config)
    
    if total_duration == 0: return []
    start_times = list(range(0, int(total_duration), step_seconds))
    
    def analyze_slice(start_time):
        try:
            res_str = recognizer.recognize_by_file(audio_path, start_time, slice_duration_seconds)
            res_dict = json.loads(res_str)
        except: res_dict = {}

        title = SILENCE_MARKER; artist = ""
        status = res_dict.get('status', {})
        if status.get('code') == 0 and 'music' in res_dict.get('metadata', {}):
            song = res_dict['metadata']['music'][0]
            raw_title = song.get('title', 'Titre Inconnu')
            title = raw_title.strip().title()
            artists_list = song.get('artists', [])
            if artists_list: artist = artists_list[0]['name'].strip().title()
            else: artist = "Artiste Inconnu"
        return { "timestamp": start_time, "title": title, "artist": artist }

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(executor.map(analyze_slice, start_times), total=len(start_times), desc="Analyse"))
    results.sort(key=lambda x: x['timestamp'])
    return results

def consolidate_timeline(raw_results):
    print("--- [ÉTAPE 3b] Consolidation intelligente de la timeline... ---")
    if not raw_results: return []

    filled_results = [r.copy() for r in raw_results]
    n = len(filled_results)
    for i in range(n):
        current = filled_results[i]
        if current['title'] != SILENCE_MARKER:
            for lookahead in range(1, MAX_GAP_TOLERANCE_STEPS + 2):
                if i + lookahead >= n: break
                target = filled_results[i + lookahead]
                if target['title'] == current['title']:
                    for k in range(1, lookahead):
                        filled_results[i+k]['title'] = current['title']
                        filled_results[i+k]['artist'] = current['artist']
                    break

    consolidated = []
    current_block = None
    for res in filled_results:
        title = res['title']; artist = res['artist']; ts = res['timestamp']
        if current_block is None:
            current_block = { "title": title, "artist": artist, "start": ts, "end": ts + step_seconds }
            continue
        if title == current_block['title']: current_block['end'] = ts + step_seconds
        else:
            consolidated.append(current_block)
            current_block = { "title": title, "artist": artist, "start": ts, "end": ts + step_seconds }
    if current_block: consolidated.append(current_block)
    return consolidated

# ==============================================================================
# --- TRAITEMENT CAPTEURS & SCORE ---
# ==============================================================================

def process_sensor_data(sensor_path):
    print("\n--- [ÉTAPE 4/7] Traitement des capteurs... ---")
    try: sensor_df = pd.read_csv(sensor_path)
    except: return pd.DataFrame(), 0
    if sensor_df.empty: return sensor_df, 0
    
    total_dur = sensor_df['timestamp'].iloc[-1] - sensor_df['timestamp'].iloc[0]
    sr = 1 / sensor_df['timestamp'].diff().mean(); win = int(STATE_WINDOW_SECONDS * sr)
    
    sensor_df['accel_mag'] = np.sqrt(sensor_df['accel_x']**2 + sensor_df['accel_y']**2 + sensor_df['accel_z']**2)
    sensor_df['accel_var'] = sensor_df['accel_mag'].rolling(window=win, center=True).var()
    sensor_df['state'] = np.where(sensor_df['accel_var'] < STATIONARY_THRESHOLD, 'immobile', 'en_mouvement')
    
    sensor_df['gyro_mag'] = np.sqrt(sensor_df['gyro_x']**2 + sensor_df['gyro_y']**2 + sensor_df['gyro_z']**2)
    sensor_df['yaw_change'] = sensor_df['attitude_yaw'].diff().fillna(0)
    sensor_df['party_power'] = sensor_df['accel_mag'] + (sensor_df['gyro_mag'] * GYRO_WEIGHT) + (np.abs(sensor_df['yaw_change']) * YAW_CHANGE_WEIGHT)
    
    return sensor_df, total_dur

def generate_final_report(raw_results, sensor_df):
    print("\n--- [ÉTAPE 5/7] Calcul des scores EKKO (Formule Corrigée v2)... ---")
    clean_blocks = consolidate_timeline(raw_results)
    summary = []
    
    if not sensor_df.empty:
        base_ts = sensor_df['timestamp'].iloc[0]
        for block in clean_blocks:
            if block['title'] == SILENCE_MARKER: continue
            
            # On extrait les données pour la durée de cette chanson
            sub = sensor_df[(sensor_df['timestamp'] >= base_ts + block['start']) & (sensor_df['timestamp'] < base_ts + block['end'])]
            if sub.empty: continue
            
            # 1. Calcul du Party Power (Intensité pure QUAND on bouge)
            motion = sub[sub['state'] == 'en_mouvement']
            if not motion.empty:
                pp = motion['party_power'].mean()
                bpm_mouv = calculate_motion_bpm(motion)
            else:
                pp = 0
                bpm_mouv = 0
            
            # 2. Volume moyen
            vol = sub['audio_power_db'].mean()
            
            # 3. Pourcentage de mouvement (Constance)
            # On évite la division par zéro
            pct_move = (len(motion) / len(sub)) * 100 if len(sub) > 0 else 0

            summary.append({
                "Titre": block['title'], "Artiste": block['artist'],
                "Début_s": block['start'], "Fin_s": block['end'],
                "Durée (s)": int(block['end'] - block['start']),
                "BPM Mouvement": bpm_mouv, 
                "Party Power": pp, # L'intensité brute
                "Volume (dB)": vol, 
                "% Mouv": pct_move # La constance
            })

    final_df = pd.DataFrame(summary)
    if final_df.empty: return pd.DataFrame(), pd.DataFrame(), clean_blocks
    
    # 1. Gestion des Outliers (Pics anormaux)
    # On définit le "Max" non pas comme le vrai max, mais comme le seuil des 95% meilleurs.
    # Tout ce qui est au-dessus sera considéré comme 100%.
    robust_max = final_df['Party Power'].quantile(0.95)
    if robust_max == 0: robust_max = 1 # Sécurité
    
    # 2. Normalisation de l'Intensité (0 à 100)
    # On utilise .clip(upper=100) pour que les pics exceptionnels ne dépassent pas 100
    final_df['Intensity Score'] = (final_df['Party Power'] / robust_max * 100).clip(upper=100)
    
    # 3. Calcul du Score EKKO Pondéré
    # Poids : 70% pour l'intensité de la danse, 30% pour la durée de la danse
    final_df['Score EKKO'] = (final_df['Intensity Score'] * 0.7) + (final_df['% Mouv'] * 0.3)
    
        
    rank_df = final_df.sort_values(by='Score EKKO', ascending=False)
    return final_df, rank_df, clean_blocks

# ==============================================================================
# --- VISUALISATION ---
# ==============================================================================

def generate_visualization(sensor_df, raw_results, clean_blocks, summary_df, zip_filename, output_dir):
    base_name = os.path.basename(zip_filename)
    raw_name = os.path.splitext(base_name)[0]
    out_png = os.path.join(output_dir, raw_name + "_viz.png")
    
    print(f"\n--- [ÉTAPE 6/7] Génération du graphique : {out_png} ---")
    if sensor_df.empty: return

    times = (sensor_df['timestamp'] - sensor_df['timestamp'].iloc[0]) / 60
    
    fig, axes = plt.subplots(4, 1, figsize=(22, 18), sharex=True, gridspec_kw={'height_ratios': [3, 2, 1, 3]})
    fig.patch.set_facecolor('#1C1C1E')
    fig.subplots_adjust(hspace=0.3)

    # 1. ÉNERGIE
    ax1 = axes[0]
    ax1.plot(times, sensor_df['party_power'], c='cyan', alpha=0.7, linewidth=1, label='Instantané')
    max_idx = sensor_df['party_power'].idxmax()
    ax1.scatter(times[max_idx], sensor_df['party_power'].max(), c='red', s=100, marker='*', zorder=10, label='Pic Max')

    ax2 = ax1.twinx()
    if not summary_df.empty:
        lbl = 'Score EKKO'
        for _, r in summary_df.iterrows():
            ax2.hlines(r['Score EKKO'], r['Début_s']/60, r['Fin_s']/60, colors='yellow', lw=4, alpha=0.9, label=lbl)
            lbl = None
    
    ax1.set_title('⚡ ÉNERGIE DE LA SOIRÉE (Party Power & Score EKKO)', color='white', fontsize=14, fontweight='bold', pad=10)
    ax1.set_ylabel('Force G', color='cyan'); ax2.set_ylabel('Score / 100', color='yellow')
    ax1.tick_params(axis='y', labelcolor='cyan', colors='white'); ax2.tick_params(axis='y', labelcolor='yellow', colors='white')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.2); ax1.set_facecolor('#2a2a2e')
    lines1, labels1 = ax1.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', facecolor='#333333', labelcolor='white')

    # 2. VOLUME
    ax_vol = axes[1]
    ax_vol.plot(times, sensor_df['audio_power_db'], c='magenta', alpha=0.8)
    ax_vol.axhline(y=-10, color='white', linestyle=':', alpha=0.5, label='Seuil Très Fort')
    ax_vol.set_title('🔊 VOLUME SONORE AMBIANT', color='white', fontsize=12, pad=10)
    ax_vol.set_ylabel('dB', color='magenta'); ax_vol.tick_params(colors='white')
    ax_vol.set_facecolor('#2a2a2e'); ax_vol.grid(True, axis='y', linestyle='--', alpha=0.2)

    # 3. ÉTATS
    ax_state = axes[2]
    mapping = {'en_mouvement': 1, 'immobile': 0}
    ax_state.fill_between(times, 0, 1, where=sensor_df['state'].map(mapping)==1, color='#007AFF', alpha=0.6)
    ax_state.fill_between(times, 0, 1, where=sensor_df['state'].map(mapping)==0, color='#3A3A3C', alpha=0.6)
    ax_state.set_title('📱 ÉTAT DU TÉLÉPHONE', color='white', fontsize=12, pad=10)
    ax_state.set_yticks([]); ax_state.set_facecolor('#2a2a2e')
    legend_patches = [mpatches.Patch(color='#007AFF', label='En Mouvement'), mpatches.Patch(color='#3A3A3C', label='Immobile')]
    ax_state.legend(handles=legend_patches, loc='center left', facecolor='#333333', labelcolor='white')

    # 4. TIMELINE
    ax_time = axes[3]
    all_titles = set([b['title'] for b in clean_blocks if b['title'] != SILENCE_MARKER])
    unique_titles = list(all_titles)
    
    if unique_titles:
        cmap = plt.colormaps['tab20']
        color_dict = {t: cmap(i % 20) for i, t in enumerate(unique_titles)}
    else: color_dict = {}

    for r in raw_results:
        t_min = r['timestamp'] / 60
        if r['title'] == SILENCE_MARKER:
            ax_time.scatter(t_min, 1.0, c='#555555', marker='x', s=20, alpha=0.6)
        else:
            c = color_dict.get(r['title'], 'white')
            ax_time.scatter(t_min, 1.0, color=c, marker='o', s=30, alpha=0.8)

    y_clean = 0.0
    for b in clean_blocks:
        start, end = b['start'] / 60, b['end'] / 60
        width = end - start
        if b['title'] == SILENCE_MARKER:
            rect = mpatches.Rectangle((start, y_clean - 0.3), width, 0.6, color='#555555', alpha=0.3)
        else:
            c = color_dict.get(b['title'], 'blue')
            rect = mpatches.Rectangle((start, y_clean - 0.3), width, 0.6, color=c, alpha=0.8)
            if width > 0.5: 
                ax_time.text(start + width/2, y_clean, f"{b['title']}", ha='center', va='center', color='white', fontsize=9, fontweight='bold', rotation=0)
        ax_time.add_patch(rect)

    ax_time.set_ylim(-0.8, 1.5); ax_time.set_yticks([0, 1]); ax_time.set_yticklabels(['Corrigé', 'Brut'], color='white')
    ax_time.set_title('🎵 TIMELINE MUSICALE & DÉTECTION', color='white', fontsize=12, pad=10)
    ax_time.set_facecolor('#2a2a2e'); ax_time.tick_params(colors='white')

    for ax in axes:
        for b in clean_blocks:
            if b['title'] not in [SILENCE_MARKER, 'END']:
                ax.axvline(b['start']/60, c='white', ls=':', alpha=0.15)

    axes[-1].set_xlabel('Temps (minutes)', color='white', fontsize=12)
    plt.savefig(out_png, dpi=150, facecolor=fig.get_facecolor())
    
    # --- CORRECTION ICI : ON UTILISE LA BONNE VARIABLE ---
    print(f"Graphique sauvegardé : {out_png}")

# ==============================================================================
# --- MAIN ---
# ==============================================================================
def main():
    zip_file = choose_zip_file()
    out_dir = create_output_directory(zip_file)
    audio, sensor, tmp = extract_zip_data(zip_file)
    
    sens_df, dur = process_sensor_data(sensor)
    raw_res = analyze_audio_timeline(audio, dur)
    sum_df, rank_df, clean_blocks = generate_final_report(raw_res, sens_df)
    
    if not rank_df.empty:
        csv_path = os.path.join(out_dir, "classement_ekko.csv")
        rank_df.to_csv(csv_path, index=False, float_format='%.2f')
        cols = ['Score EKKO', 'Titre', 'Artiste', 'BPM Mouvement', 'Party Power']
        print(f"\n--- CLASSEMENT (Top 10) ---\n{rank_df[cols].head(10).to_string(index=False)}")

    generate_visualization(sens_df, raw_res, clean_blocks, sum_df, zip_file, out_dir)
    
    try: shutil.rmtree(tmp); print("\nNettoyage OK.")
    except: pass

if __name__ == '__main__':
    if 'METTEZ_VOTRE_HOST_ICI' in config.values(): print("ERREUR : Clés manquantes."); sys.exit()
    main()