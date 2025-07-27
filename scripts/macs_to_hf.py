#!/usr/bin/env python3
"""
MACS to HuggingFace Datasets Converter

Convert the raw Multilingual Audio Captioning in real-life Scenes (MACS) 
release into a single, streaming-ready Hugging Face datasets.Dataset.
"""

import argparse
import csv
from pathlib import Path
import pandas as pd
import yaml
import datasets
from huggingface_hub import login


def smart_read_csv(path):
    """Read CSV with automatic delimiter detection."""
    with open(path, "r") as f:
        dialect = csv.Sniffer().sniff(f.readline(), [",", "\t", ";"])
    return pd.read_csv(path, sep=dialect.delimiter)


def convert_macs_to_hf(audio_root, meta_csv, yaml_file, out_dir=None, 
                       push_to_hub=None, private=False, hf_token=None):
    """
    Convert MACS dataset to HuggingFace format.
    
    Args:
        audio_root: Path to directory containing WAV files
        meta_csv: Path to meta.csv file from tau_meta/
        yaml_file: Path to MACS.yaml file (captions & tags)
        out_dir: Directory to save the processed dataset
        push_to_hub: Repository ID to push to HF Hub (e.g. 'username/MACS_captions')
        private: Whether to mark the Hub repo as private
        hf_token: HuggingFace token for authentication
    """
    
    # Convert to Path objects
    audio_root = Path(audio_root)
    meta_csv = Path(meta_csv)
    yaml_file = Path(yaml_file)
    
    print(f"🎵 Reading audio metadata from: {meta_csv}")
    print(f"📝 Reading captions from: {yaml_file}")
    print(f"🔊 Audio files location: {audio_root}")
    
    # 1. Read meta.csv
    meta_df = smart_read_csv(meta_csv)
    meta_df["basename"] = meta_df["filename"].str.replace(r"^audio/", "", regex=True)
    meta_lookup = meta_df.set_index("basename").to_dict(orient="index")
    
    # 2. Read MACS.yaml
    with open(yaml_file) as f:
        yaml_items = yaml.safe_load(f)["files"]
    
    print(f"📊 Processing {len(yaml_items)} audio files...")
    
    # 3. Process and merge data
    rows = []
    for item in yaml_items:
        basename = Path(item["filename"]).name
        annotations = item["annotations"]
        sentences = [a["sentence"] for a in annotations]
        tags = [a["tags"] for a in annotations]
        annotator = [a["annotator_id"] for a in annotations]
        
        # Get meta.csv extras (scene_label, identifier, source_label)
        m = meta_lookup.get(basename, {})
        scene_label = m.get("scene_label")  # audio class
        identifier = m.get("identifier")
        source_label = m.get("source_label")
        
        rows.append({
            "filename": basename,
            "scene": scene_label,
            "audio": str(audio_root / basename),
            "captions": sentences,
            "tags": tags,
            "annotators": annotator,
            "audio_identifier": identifier,
            "audio_source_label": source_label,
        })
    
    # 4. Create HuggingFace dataset
    print("🤗 Creating HuggingFace dataset...")
    
    features = datasets.Features({
        "filename": datasets.Value("string"),
        "scene": datasets.Value("string"),
        "audio": datasets.Audio(),
        "captions": datasets.Sequence(datasets.Value("string")),
        "tags": datasets.Sequence(
            datasets.Sequence(datasets.Value("string"))
        ),
        "annotators": datasets.Sequence(datasets.Value("int32")),
        "audio_identifier": datasets.Value("string"),
        "audio_source_label": datasets.Value("string"),
    })
    
    ds_all = datasets.Dataset.from_pandas(
        pd.DataFrame(rows), features=features, preserve_index=False
    )
    
    
    print(f"✅ Dataset created with {len(ds_all)} samples")
    print(f"📋 Sample: {ds_all[0]}")
    
    # 5. Save to disk if output directory specified
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"💾 Saving dataset to: {out_dir}")
        ds_all.save_to_disk(out_dir)
        print("✅ Dataset saved successfully!")
    
    # 6. Push to Hub if requested
    if push_to_hub:
        print(f"🚀 Pushing to HuggingFace Hub: {push_to_hub}")
        
        # Login with token if provided
        if hf_token:
            login(token=hf_token)
        
        ds_all.push_to_hub(
            push_to_hub, 
            private=private
        )
        print("✅ Dataset pushed to Hub successfully!")
    
    return ds_all


def main():
    parser = argparse.ArgumentParser(
        description="Convert MACS dataset to HuggingFace format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python macs_to_hf.py --audio-root MACS_raw/audio --meta-csv MACS_raw/tau_meta/meta.csv --yaml-file MACS_raw/MACS.yaml --out-dir macs_hf
  
  python macs_to_hf.py --audio-root MACS_raw/audio --meta-csv MACS_raw/tau_meta/meta.csv --yaml-file MACS_raw/MACS.yaml --push-to-hub username/MACS_captions --private
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--audio-root", 
        type=str, 
        required=True,
        help="Directory that contains the WAV files"
    )
    parser.add_argument(
        "--meta-csv", 
        type=str, 
        required=True,
        help="meta.csv from tau_meta/ (tab- or comma-separated)"
    )
    parser.add_argument(
        "--yaml-file", 
        type=str, 
        required=True,
        help="MACS.yaml (captions & tags)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--out-dir", 
        type=str, 
        default="macs_hf",
        help="Where to write the processed dataset (default: macs_hf)"
    )
    parser.add_argument(
        "--push-to-hub", 
        type=str,
        help="Push to the HF Hub after conversion (e.g. 'username/MACS_captions')"
    )
    parser.add_argument(
        "--private", 
        action="store_true",
        help="Mark the Hub repo as private (only with --push-to-hub)"
    )
    parser.add_argument(
        "--hf_token", 
        type=str,
        help="Token from your HF account"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.private and not args.push_to_hub:
        parser.error("--private can only be used with --push-to-hub")
    
    if not Path(args.audio_root).exists():
        parser.error(f"Audio root directory does not exist: {args.audio_root}")
    
    if not Path(args.meta_csv).exists():
        parser.error(f"Meta CSV file does not exist: {args.meta_csv}")
    
    if not Path(args.yaml_file).exists():
        parser.error(f"YAML file does not exist: {args.yaml_file}")
    
    # Convert the dataset
    convert_macs_to_hf(
        audio_root=args.audio_root,
        meta_csv=args.meta_csv,
        yaml_file=args.yaml_file,
        out_dir=args.out_dir,
        push_to_hub=args.push_to_hub,
        private=args.private,
        hf_token=args.hf_token
    )
    
    print("🎉 Conversion completed successfully!")


if __name__ == "__main__":
    main()
