"""
Generate synthetic Likert survey data for
Customer Segmentation Model Adoption (TAM-based)

Variabel laten TAM (Technology Acceptance Model):
  PEOU → PU → ATU → BI → ASU
  PEOU → ATU

Korelasi antar konstruk dibuat realistis via latent variable model.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 150  # jumlah responden

# ─────────────────────────────────────────────
# 1. Bangkitkan latent scores dengan korelasi TAM
# ─────────────────────────────────────────────
# Latent mean sekitar 3.8 (sedikit positif dari skala 1–5)

# PEOU: exogenous
PEOU_lat = np.random.normal(0, 1, N)

# PU: dipengaruhi PEOU (H1: β=0.45)
PU_lat = 0.45 * PEOU_lat + np.random.normal(0, np.sqrt(1 - 0.45**2), N)

# ATU: dipengaruhi PEOU (H2: β=0.30) dan PU (H3: β=0.40)
ATU_lat = (0.30 * PEOU_lat + 0.40 * PU_lat
           + np.random.normal(0, np.sqrt(1 - 0.30**2 - 0.40**2), N))

# BI: dipengaruhi PU (H4: β=0.35) dan ATU (H5: β=0.40)
BI_lat = (0.35 * PU_lat + 0.40 * ATU_lat
          + np.random.normal(0, np.sqrt(1 - 0.35**2 - 0.40**2), N))

# ASU: dipengaruhi BI (H6: β=0.55)
ASU_lat = 0.55 * BI_lat + np.random.normal(0, np.sqrt(1 - 0.55**2), N)

latents = {
    "PEOU": PEOU_lat,
    "PU":   PU_lat,
    "ATU":  ATU_lat,
    "BI":   BI_lat,
    "ASU":  ASU_lat,
}

# Skala latent → skor Likert (mean berbeda per konstruk)
target_means = {"PEOU": 3.80, "PU": 3.90, "ATU": 3.75, "BI": 3.85, "ASU": 3.65}
target_std   = {"PEOU": 0.75, "PU": 0.70, "ATU": 0.78, "BI": 0.72, "ASU": 0.80}


def latent_to_likert(latent, n_items, t_mean, t_std, noise=0.55):
    """Convert latent scores → correlated Likert items."""
    # Rescale latent to target distribution
    z = (latent - latent.mean()) / latent.std()
    scaled = z * t_std + t_mean
    items = []
    for _ in range(n_items):
        raw = scaled + np.random.normal(0, noise, len(latent))
        clipped = np.clip(np.round(raw), 1, 5).astype(int)
        items.append(clipped)
    return items


# ─────────────────────────────────────────────
# 2. Bangkitkan item per konstruk
# ─────────────────────────────────────────────
constructs_items = {
    "PU": {
        "latent": latents["PU"],
        "n_items": 5,
        "labels": [
            "PU1_Model_segmentasi_meningkatkan_efisiensi_kerja",
            "PU2_Model_membantu_menyelesaikan_pekerjaan_lebih_cepat",
            "PU3_Model_meningkatkan_produktivitas",
            "PU4_Model_meningkatkan_kualitas_keputusan_pemasaran",
            "PU5_Model_segmentasi_berguna_bagi_bisnis",
        ],
    },
    "PEOU": {
        "latent": latents["PEOU"],
        "n_items": 5,
        "labels": [
            "PEOU1_Mudah_mempelajari_model_segmentasi",
            "PEOU2_Mudah_mengoperasikan_model",
            "PEOU3_Antarmuka_model_jelas_dan_mudah_dipahami",
            "PEOU4_Mudah_menjadi_terampil_menggunakan_model",
            "PEOU5_Model_segmentasi_tidak_memerlukan_usaha_besar",
        ],
    },
    "ATU": {
        "latent": latents["ATU"],
        "n_items": 4,
        "labels": [
            "ATU1_Menggunakan_model_segmentasi_adalah_ide_baik",
            "ATU2_Penggunaan_model_merupakan_keputusan_bijak",
            "ATU3_Saya_menyukai_penggunaan_model_segmentasi",
            "ATU4_Model_segmentasi_memberikan_pengalaman_positif",
        ],
    },
    "BI": {
        "latent": latents["BI"],
        "n_items": 4,
        "labels": [
            "BI1_Saya_berniat_menggunakan_model_di_masa_depan",
            "BI2_Saya_akan_selalu_mencoba_menggunakan_model",
            "BI3_Saya_berencana_menggunakan_model_secara_rutin",
            "BI4_Saya_akan_merekomendasikan_model_kepada_rekan",
        ],
    },
    "ASU": {
        "latent": latents["ASU"],
        "n_items": 3,
        "labels": [
            "ASU1_Saya_menggunakan_model_secara_rutin",
            "ASU2_Frekuensi_penggunaan_model_saya_tinggi",
            "ASU3_Model_sudah_terintegrasi_dalam_pekerjaan_saya",
        ],
    },
}

# ─────────────────────────────────────────────
# 3. Demografi
# ─────────────────────────────────────────────
gender = np.random.choice(["Laki-laki", "Perempuan"], N, p=[0.55, 0.45])
age_group = np.random.choice(
    ["<25", "25-34", "35-44", "45-54", "≥55"], N,
    p=[0.10, 0.35, 0.30, 0.18, 0.07]
)
education = np.random.choice(
    ["SMA/Sederajat", "Diploma", "S1", "S2/S3"], N,
    p=[0.10, 0.15, 0.55, 0.20]
)
experience = np.random.choice(
    ["<1 tahun", "1-3 tahun", "4-6 tahun", ">6 tahun"], N,
    p=[0.12, 0.30, 0.33, 0.25]
)
division = np.random.choice(
    ["Pemasaran", "CRM", "Analis Data", "Penjualan", "IT/Teknologi"], N,
    p=[0.28, 0.20, 0.22, 0.18, 0.12]
)

# ─────────────────────────────────────────────
# 4. Assemble DataFrame
# ─────────────────────────────────────────────
data = {
    "Responden_ID": [f"R{str(i+1).zfill(3)}" for i in range(N)],
    "Jenis_Kelamin": gender,
    "Kelompok_Usia": age_group,
    "Pendidikan": education,
    "Pengalaman_Kerja": experience,
    "Divisi": division,
}

for code, info in constructs_items.items():
    items = latent_to_likert(
        info["latent"], info["n_items"],
        target_means[code], target_std[code]
    )
    for label, col in zip(info["labels"], items):
        data[label] = col

df = pd.DataFrame(data)
df.to_csv("data/survey_likert.csv", index=False)

print(f"Data berhasil dibangkitkan: data/survey_likert.csv")
print(f"  Responden : {N}")
print(f"  Kolom     : {df.shape[1]}")
print(f"\nPreview 3 baris pertama:")
print(df.iloc[:, :8].to_string())
