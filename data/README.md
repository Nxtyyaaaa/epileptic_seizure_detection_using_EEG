# EEG Datasets Directory Structure & Upload Guide

This project is configured to evaluate EEG seizure detection across three experimental tracks:

```
data/
├── raw/
│   ├── bonn/       <- Track A: University of Bonn EEG Benchmark
│   ├── chbmit/     <- Track B: CHB-MIT Scalp EEG Clinical Database
│   └── siena/      <- Track C: Siena Scalp EEG External Database
└── processed/      <- Cached preprocessed windows and extracted feature matrices
```

---

### 1. Bonn EEG Dataset (Track A)
- **Official Source:** University of Bonn Epileptology (Andrzejak et al., 2001)
- **Folder Location:** `data/raw/bonn/`
- **Structure:**
  You can upload either:
  - Subdirectories `A/`, `B/`, `C/`, `D/`, `E/` (or `Z/`, `O/`, `N/`, `F/`, `S/`) with their `.txt` files:
    ```
    data/raw/bonn/
    ├── A/ (or Z/) -> 100 text files (Z001.txt ... Z100.txt)
    ├── B/ (or O/) -> 100 text files (O001.txt ... O100.txt)
    ├── C/ (or N/) -> 100 text files (N001.txt ... N100.txt)
    ├── D/ (or F/) -> 100 text files (F001.txt ... F100.txt)
    └── E/ (or S/) -> 100 text files (S001.txt ... S100.txt)
    ```
  - Or place all files directly inside `data/raw/bonn/` prefixed with `Z`, `O`, `N`, `F`, `S` (or `A`, `B`, `C`, `D`, `E`).
- **Characteristics:** 4,096 samples per segment at 173.61 Hz (~23.6s per file).

---

### 2. CHB-MIT Scalp EEG Dataset (Track B)
- **Official Source:** PhysioNet CHB-MIT Scalp EEG Database (Goldberger et al.)
- **Folder Location:** `data/raw/chbmit/`
- **Structure:**
  Upload patient folders (e.g., `chb01`, `chb02`, `chb03`, `chb04`, `chb05`, `chb06` for the 4-6 patient pilot study):
  ```
  data/raw/chbmit/
  ├── chb01/
  │   ├── chb01-summary.txt
  │   ├── chb01_01.edf
  │   ├── chb01_02.edf
  │   └── ...
  ├── chb02/
  │   ├── chb02-summary.txt
  │   └── ...
  └── ...
  ```
- **Characteristics:** Multichannel EDF files recorded at 256 Hz with annotated seizure start/end timestamps in the summary files.

---

### 3. Siena Scalp EEG Dataset (Track C)
- **Official Source:** PhysioNet Siena Scalp EEG Database (Detti et al.)
- **Folder Location:** `data/raw/siena/`
- **Structure:**
  Upload subject folders (e.g., `pn01`, `pn02`, ...):
  ```
  data/raw/siena/
  ├── pn01/
  │   ├── pn01_01.edf
  │   └── pn01_01.txt (annotation timestamps)
  └── ...
  ```
- **Characteristics:** Multichannel EDF files recorded at 512 Hz, harmonized to 256 Hz with anti-aliasing filters.

---

*Note: If the raw data folders are empty, the pipeline will automatically generate synthetic mock waveforms to verify functionality.*

