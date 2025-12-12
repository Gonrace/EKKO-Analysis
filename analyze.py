# ==============================================================================
# EKKO ANALYSIS SCRIPT V3.3
# ==============================================================================
# Correctifs v3.3 :
# - Fix crash "FileNotFoundError" en nettoyant le chemin du fichier ZIP.
# - Fix warning "MatplotlibDeprecationWarning" pour les couleurs.
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

# Charge les variables d'environnement (.env)
load_dotenv()

# Vérification de tqdm
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

# --- CONFIGURATION DES DOSSIERS ---
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
    # On récupère juste le nom du fichier sans le chemin
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

        title = SILENCE_MARKER
        artist = ""
        status = res_dict.get('status', {})
        
        if status.get('code') == 0 and 'music' in res_dict.get('metadata', {}):
            song = res_dict['metadata']['music'][0]
            raw_title = song.get('title', 'Titre Inconnu')
            title = raw_title.strip().title()
            artists_list = song.get('artists', [])
            if artists_list:
                artist = artists_list[0]['name'].strip().title()
            else:
                artist = "Artiste Inconnu"
            
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
        title = res['title']
        artist = res['artist']
        ts = res['timestamp']
        
        if current_block is None:
            current_block = { "title": title, "artist": artist, "start": ts, "end": ts + step_seconds }
            continue
        
        if title == current_block['title']:
            current_block['end'] = ts + step_seconds
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
    
    sr = 1 / sensor_df['timestamp'].diff().mean()
    win = int(STATE_WINDOW_SECONDS * sr)
    sensor_df['accel_mag'] = np.sqrt(sensor_df['accel_x']**2 + sensor_df['accel_y']**2 + sensor_df['accel_z']**2)
    sensor_df['accel_var'] = sensor_df['accel_mag'].rolling(window=win, center=True).var()
    sensor_df['state'] = np.where(sensor_df['accel_var'] < STATIONARY_THRESHOLD, 'immobile', 'en_mouvement')
    
    sensor_df['gyro_mag'] = np.sqrt(sensor_df['gyro_x']**2 + sensor_df['gyro_y']**2 + sensor_df['gyro_z']**2)
    sensor_df['yaw_change'] = sensor_df['attitude_yaw'].diff().fillna(0)
    sensor_df['party_power'] = sensor_df['accel_mag'] + (sensor_df['gyro_mag'] * GYRO_WEIGHT) + (np.abs(sensor_df['yaw_change']) * YAW_CHANGE_WEIGHT)
    
    return sensor_df, total_dur

def generate_final_report(raw_results, sensor_df):
    print("\n--- [ÉTAPE 5/7] Calcul des scores EKKO... ---")
    clean_blocks = consolidate_timeline(raw_results)
    summary = []
    
    if not sensor_df.empty:
        base_ts = sensor_df['timestamp'].iloc[0]
        for block in clean_blocks:
            if block['title'] == SILENCE_MARKER: continue
            
            sub = sensor_df[(sensor_df['timestamp'] >= base_ts + block['start']) & (sensor_df['timestamp'] < base_ts + block['end'])]
            if sub.empty: continue
            
            motion = sub[sub['state'] == 'en_mouvement']
            pp = sub['party_power'].mean()
            bpm_mouv = calculate_motion_bpm(motion) if not motion.empty else 0
            vol = sub['audio_power_db'].mean()
            pct_move = (len(motion) / len(sub)) * 100

            summary.append({
                "Titre": block['title'], "Artiste": block['artist'],
                "Début_s": block['start'], "Fin_s": block['end'],
                "Durée (s)": int(block['end'] - block['start']),
                "BPM Mouvement": bpm_mouv, "Party Power": pp,
                "Volume (dB)": vol, "% Mouv": pct_move
            })

    final_df = pd.DataFrame(summary)
    if final_df.empty: return pd.DataFrame(), pd.DataFrame(), clean_blocks

    max_pp = final_df['Party Power'].max()
    if max_pp > 0:
        norm_pp = (final_df['Party Power'] / max_pp) * 100
        final_df['Score EKKO'] = norm_pp * (final_df['% Mouv'] / 100)
    else: final_df['Score EKKO'] = 0
        
    rank_df = final_df.sort_values(by='Score EKKO', ascending=False)
    return final_df, rank_df, clean_blocks

# ==============================================================================
# --- VISUALISATION (CORRIGÉE) ---
# ==============================================================================

def generate_visualization(sensor_df, raw_results, clean_blocks, summary_df, zip_filename, output_dir):
    # --- FIX 1 : Utilisation de os.path.basename pour nettoyer le nom du fichier ---
    base_name = os.path.basename(zip_filename) 
    raw_name = os.path.splitext(base_name)[0]
    out_png = os.path.join(output_dir, raw_name + "_viz.png")
    
    print(f"\n--- [ÉTAPE 6/7] Graphique : {out_png} ---")
    if sensor_df.empty: return

    times = (sensor_df['timestamp'] - sensor_df['timestamp'].iloc[0]) / 60
    fig, ax = plt.subplots(4, 1, figsize=(20, 18), sharex=True, gridspec_kw={'height_ratios': [3, 2, 1, 4]})
    fig.patch.set_facecolor('#1C1C1E')

    # 1. Énergie
    ax[0].plot(times, sensor_df['party_power'], c='cyan', alpha=0.6, label='Instantané')
    ax2 = ax[0].twinx()
    if not summary_df.empty:
        for _, r in summary_df.iterrows():
            ax2.hlines(r['Score EKKO'], r['Début_s']/60, r['Fin_s']/60, colors='yellow', lw=3)
    ax[0].set_title('Énergie & Score EKKO', color='white'); ax[0].set_facecolor('#2a2a2e')
    ax2.set_ylim(0, 105); ax2.tick_params(axis='y', labelcolor='yellow')

    # 2. Volume
    ax[1].plot(times, sensor_df['audio_power_db'], c='magenta'); ax[1].set_title('Volume (dB)', color='white'); ax[1].set_facecolor('#2a2a2e')

    # 3. États
    mapping = {'en_mouvement': 1, 'immobile': 0}
    ax[2].fill_between(times, 0, 1, where=sensor_df['state'].map(mapping)==1, color='blue', alpha=0.5, label='Mouvement')
    ax[2].fill_between(times, 0, 1, where=sensor_df['state'].map(mapping)==0, color='gray', alpha=0.5, label='Immobile')
    ax[2].set_title('État', color='white'); ax[2].set_yticks([]); ax[2].set_facecolor('#2a2a2e')

    # 4. TIMELINE
    ax4 = ax[3]
    all_titles = set([b['title'] for b in clean_blocks if b['title'] != SILENCE_MARKER])
    all_titles.update([r['title'] for r in raw_results if r['title'] != SILENCE_MARKER])
    unique_titles = list(all_titles)
    
    # --- FIX 2 : Remplacement de plt.cm.get_cmap (obsolète) ---
    if unique_titles:
        # On utilise le nouveau système de colormaps
        cmap = plt.colormaps['tab20']
        color_dict = {t: cmap(i % 20) for i, t in enumerate(unique_titles)}
    else: color_dict = {}

    for r in raw_results:
        t_min = r['timestamp'] / 60
        if r['title'] == SILENCE_MARKER:
            ax4.scatter(t_min, 1.0, c='gray', marker='x', s=30, alpha=0.5)
        else:
            c = color_dict.get(r['title'], 'white')
            ax4.scatter(t_min, 1.0, color=c, marker='o', s=50, edgecolors='white', linewidth=0.5)

    y_clean = 0.0
    for b in clean_blocks:
        start = b['start'] / 60
        end = b['end'] / 60
        width = end - start
        
        if b['title'] == SILENCE_MARKER:
            rect = mpatches.Rectangle((start, y_clean - 0.2), width, 0.4, color='gray', alpha=0.2)
        else:
            c = color_dict.get(b['title'], 'blue')
            rect = mpatches.Rectangle((start, y_clean - 0.2), width, 0.4, color=c, alpha=0.9)
            ax4.text(start + width/2, y_clean, f"{b['artist']}\n{b['title']}", 
                     ha='center', va='center', color='white', fontsize=8, fontweight='bold', rotation=0)
        ax4.add_patch(rect)

    ax4.set_ylim(-0.5, 1.5); ax4.set_yticks([0, 1])
    ax4.set_yticklabels(['Corrigé', 'Brut'], color='white')
    ax4.set_title('Timeline Musicale', color='white')
    ax4.set_facecolor('#2a2a2e'); ax4.tick_params(colors='white')
    
    for a in ax:
        for b in clean_blocks:
            if b['title'] not in [SILENCE_MARKER, 'END']:
                a.axvline(b['start']/60, c='white', ls=':', alpha=0.2)

    plt.savefig(out_png, dpi=100, facecolor='#1C1C1E')

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