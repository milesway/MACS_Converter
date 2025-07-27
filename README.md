# MACS to HuggingFace Datasets Converter



<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python ≥ 3.8](https://img.shields.io/badge/Python-3.8%2B-informational)](#)
[![HF Datasets](https://img.shields.io/badge/🤗-datasets-blue)](https://huggingface.co/datasets/milesway6/MACS)

</div>

**Turn the raw Multilingual Audio Captioning in real-life Scenes (MACS) release into a single, streaming-ready Hugging Face `datasets.Dataset`—in one command.**

**Link to the processed data (processed using the script in this repo): [https://huggingface.co/datasets/milesway6/MACS](https://huggingface.co/datasets/milesway6/MACS)**

---

## Why this repo?

The official MACS bundle on Zenodo (≈ 10 GB) ships as WAV files + mixed-format metadata (`CSV`, `YAML`).
That layout is perfect for archival—but clunky for modelling. This converter:

* **Merges** captions, tags, annotator IDs & scene labels into a **single tidy table**.
* Includes origin audio in the dataset
* Outputs a **`DatasetDict` with one logical split** (`all`) that streams audio on-the-fly.
* Optionally **pushes straight to the Hugging Face Hub**—ready for reproducible sharing.

---

## 1. Download the original data

The converter depends on the *unaltered* MACS files—**audio is not redistributed here**.

1. Head to the Zenodo record for audio caption
   [https://zenodo.org/records/5114771](https://zenodo.org/records/5114771)

2. Audio files are in a seprate link
    [https://zenodo.org/record/2589280](https://zenodo.org/record/2589280)

3. Download and unpack. Organize your files so that it at least includes key files like following:

   ```
   MACS_raw/
   ├── audio/                    # 5 322 × 10-s WAVs
   ├── tau_meta/
   │    └── meta.csv             # scene labels, IDs, etc.
   └── MACS.yaml                 # captions, tags, annotators
   ```

4. *(Optional)* Verify checksums listed on Zenodo.

---

## 2. Quick install

```bash
git clone https://github.com/milesway/MACS_Converter.git
cd MACS_Converter

conda create -n MACS_Converter python==3.11
conda activate MACS_Converter
pip install -r requirements.txt
```

---

## 3. One-line conversion

```bash
python macs_to_hf.py \
  --audio-root MACS_raw/audio \
  --meta-csv   MACS_raw/tau_meta/meta.csv \
  --yaml-file  MACS_raw/MACS.yaml \
  --out-dir    macs_hf
```

💡 **Tip:** Use absolute paths if you run the script from elsewhere.


---

## 4. CLI options

| Flag                    | Required              | Description                                                         |
| ----------------------- | --------------------- | ------------------------------------------------------------------- |
| `--audio-root PATH`     | ✔                     | Directory that **contains the WAV files**                           |
| `--meta-csv PATH`       | ✔                     | `meta.csv` from `tau_meta/` (tab- or comma-separated)               |
| `--yaml-file PATH`      | ✔                     | `MACS.yaml` (captions & tags)                                       |
| `--out-dir DIR`         | ✖ (default `macs_hf`) | Where to write the processed dataset                                |
| `--push-to-hub REPO_ID` | ✖                     | Push to the HF Hub after conversion (e.g. `username/MACS_captions`) |
| `--private`             | ✖                     | Mark the Hub repo as private (only with `--push-to-hub`)            |
| `--hf_token`            | ✖                     | Token from your HF account                                          |
| `-h`, `--help`          | –                     | Show full help                                                      |

---



## 5. Push to the Hugging Face Hub

```bash
python scripts/macs_to_hf.py \
  --audio-root MACS_raw/audio \
  --meta-csv   MACS_raw/tau_meta/meta.csv \
  --yaml-file  MACS_raw/MACS.yaml \
  --push-to-hub your-hf-username/MACS_captions \
  --hf_token YOUR_HF_TOKEN \
  --private # optional

```

Authenticate once with `huggingface-cli login` or provide your token with `--hf_token`; the script takes care of the rest.

---

## 6. License

* **Code** in this repository—MIT.
* **Dataset** Refer the original MACS distribution.
*  This project is only a restruction of publicly available dataset.

---



### Disclaimer

> This repository is **not affiliated** with the MACS authors, their institutions, or Zenodo.
> The converter is provided **“as is”** without warranties of any kind. Use at your own risk and ensure compliance with the original dataset license and any applicable regulations in your jurisdiction.
