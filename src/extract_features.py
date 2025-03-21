import os
import time
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import numpy as np

# Paths
BARK_CSV_PATH = "data/processed/bark_segments.csv"
TFRECORD_DIRS = [
    "data/embeddings/audioset_v1_embeddings/bal_train",
    "data/embeddings/audioset_v1_embeddings/unbal_train",
]
OUTPUT_EMBEDDINGS_PATH = "data/features/bark_embeddings.csv"
MISSING_SEGMENTS_PATH = "data/features/missing_segments.csv"
STATS_SUMMARY_PATH = "data/features/stats_summary.csv"

# Load bark labels
BARK_LABELS = {"/m/0btp2", "/m/07qf0zm", "/m/05r5c", "/m/02yds9", "/m/0brhx", "/m/03s_tn", "/m/0cq_cl"}

def load_bark_segments(csv_path):
    """Loads the bark_segments.csv file and returns a DataFrame."""
    df = pd.read_csv(csv_path, dtype=str)
    print(f"📂 Loaded bark segments from: {csv_path}")
    print(f"🔍 Found {len(df)} bark segments.\n")
    return df

def get_tfrecord_files():
    """Finds all TFRecord files in the specified directories."""
    tfrecord_files = []
    for directory in TFRECORD_DIRS:
        if os.path.exists(directory):
            files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".tfrecord")]
            tfrecord_files.extend(files)
    print(f"📦 Found {len(tfrecord_files)} TFRecord files.\n")
    return tfrecord_files

def parse_tfrecord_fn(proto):
    """Parses a single TFRecord example, correctly decoding the embeddings."""
    feature_description = {
        "video_id": tf.io.FixedLenFeature([], tf.string),
        "start_time_seconds": tf.io.FixedLenFeature([], tf.float32),
        "end_time_seconds": tf.io.FixedLenFeature([], tf.float32),
        "labels": tf.io.VarLenFeature(tf.int64),
        "audio_embedding": tf.io.VarLenFeature(tf.string),  # Stored as byte strings
    }
    example = tf.io.parse_single_example(proto, feature_description)

    # Convert sparse tensors to dense
    audio_embedding = tf.sparse.to_dense(example["audio_embedding"], default_value=b"")

    # Decode raw bytes to uint8
    decoded_frames = tf.map_fn(lambda x: tf.io.decode_raw(x, tf.uint8), audio_embedding, dtype=tf.uint8)

    # Print extracted data for debugging
    print(f"📽 Video ID: {example['video_id'].numpy().decode('utf-8')}")
    print(f"🕒 Segment: {example['start_time_seconds'].numpy()} – {example['end_time_seconds'].numpy()}")
    print(f"🏷 Labels: {example['labels'].numpy().tolist()}")
    print(f"🎵 Embedding frames: {len(decoded_frames.numpy())}")

    return {
        "video_id": example["video_id"],
        "start_time_seconds": example["start_time_seconds"],
        "end_time_seconds": example["end_time_seconds"],
        "labels": tf.sparse.to_dense(example["labels"]),
        "audio_embedding": decoded_frames,  # Return decoded frames instead of raw bytes
    }



def process_tfrecords(tfrecord_files, bark_df):
    """Processes TFRecords and matches embeddings to bark segments."""
    matched_count = 0
    unmatched_count = 0
    matched_embeddings = []
    missing_segments = []

    for i, tfrecord_file in enumerate(tfrecord_files):
        print(f"📖 Processing file {i+1}/{len(tfrecord_files)}: {tfrecord_file}")

        raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
        parsed_dataset = raw_dataset.map(parse_tfrecord_fn)

        for sample in parsed_dataset:
            video_id = sample["video_id"].numpy().decode("utf-8")
            start_time_sec = sample["start_time_seconds"].numpy()
            end_time_sec = sample["end_time_seconds"].numpy()
            labels = sample["labels"].numpy().tolist()
            embeddings = sample["audio_embedding"].numpy()

            # Check if embeddings are empty
            if embeddings.size == 0:
                print(f"⚠️ Empty embedding for Video ID: {video_id}, segment {start_time_sec}-{end_time_sec}")
                continue  # Skip empty embeddings

            # Match with bark segments
            match = bark_df[
                (bark_df["YTID"] == video_id) & 
                (bark_df["start_seconds"].astype(float) == start_time_sec) & 
                (bark_df["end_seconds"].astype(float) == end_time_sec)
            ]

            if not match.empty:
                matched_count += 1
                matched_embeddings.append({
                    "video_id": video_id,
                    "start_time": start_time_sec,
                    "end_time": end_time_sec,
                    "embedding": embeddings.tolist(),
                    "source_tfrecord": tfrecord_file  # Store TFRecord file path
                })
            else:
                unmatched_count += 1
                missing_segments.append([video_id, start_time_sec, end_time_sec])

    # Save matched embeddings
    pd.DataFrame(matched_embeddings).to_csv(OUTPUT_EMBEDDINGS_PATH, index=False)
    print(f"\n✅ Saved embeddings for {matched_count} matched segments.")

    return matched_count, unmatched_count


if __name__ == "__main__":
    bark_df = load_bark_segments(BARK_CSV_PATH)
    tfrecord_files = get_tfrecord_files()

    matched, unmatched = process_tfrecords(tfrecord_files, bark_df)

    print(f"\n🎯 Matching complete: {matched} matched, {unmatched} unmatched.")
