import os
import numpy as np
import pandas as pd
import tensorflow as tf

# ==== File paths ====
CSV_PATH = "data/processed/bark_segments.csv"
TFRECORD_DIR = "data/embeddings/audioset_v1_embeddings/bal_train"
X_SAVE_PATH = "data/processed/X.npy"
Y_SAVE_PATH = "data/processed/y.npy"

# ==== Load filtered bark segments ====
df = pd.read_csv(CSV_PATH)
df["positive_labels"] = df["positive_labels"].astype(str).str.strip()
df["bark_labels"] = df["bark_labels"].astype(str).str.strip()
df["start_seconds"] = df["start_seconds"].astype(float)

# ==== Build lookup sets ====
bark_ids = set(df["YTID"])
ytid_start_map = set(zip(df["YTID"], df["start_seconds"]))
ytid_to_label = {ytid: 1 for ytid in bark_ids}

# ==== TFRecord parsing ====
def parse_tfrecord(example_proto):
    feature_description = {
        "video_id": tf.io.FixedLenFeature([], tf.string),
        "start_time_seconds": tf.io.FixedLenFeature([], tf.float32),
        "end_time_seconds": tf.io.FixedLenFeature([], tf.float32),
        "labels": tf.io.VarLenFeature(tf.int64),
        "audio_embedding": tf.io.FixedLenSequenceFeature([128], tf.string, allow_missing=True)
    }
    return tf.io.parse_single_example(example_proto, feature_description)

# ==== Main feature extraction ====
def extract_features():
    X, y = [], []

    for filename in os.listdir(TFRECORD_DIR):
        if not filename.endswith(".tfrecord"):
            continue

        file_path = os.path.join(TFRECORD_DIR, filename)
        dataset = tf.data.TFRecordDataset(file_path)

        for raw_example in dataset:
            try:
                example = parse_tfrecord(raw_example)
                video_id = example["video_id"].numpy().decode()
                start_time = float(example["start_time_seconds"].numpy())

                # Fuzzy match with tolerance ±0.5 seconds
                matched = False
                for delta in [-0.5, 0.0, 0.5]:
                    rounded_time = round(start_time + delta, 1)
                    if (video_id, rounded_time) in ytid_start_map:
                        matched = True
                        break

                if not matched:
                    print(f"⚠️ Skipping {video_id} at {start_time}s (no match)")
                    continue

                # Convert embeddings from bytes → 128D uint8 array
                embedding_bytes = example["audio_embedding"].numpy()
                embeddings = np.array([np.frombuffer(e, dtype=np.uint8) for e in embedding_bytes])

                if embeddings.size == 0:
                    print(f"⚠️ Empty embeddings for {video_id}")
                    continue

                mean_embedding = np.mean(embeddings, axis=0)
                X.append(mean_embedding)
                y.append(1)  # All are bark

            except Exception as e:
                print(f"❌ Failed to parse a record in {filename}: {e}")
                continue

    # Convert to arrays and save
    X = np.array(X)
    y = np.array(y)

    np.save(X_SAVE_PATH, X)
    np.save(Y_SAVE_PATH, y)

    print(f"✅ Saved {len(X)} bark samples as X.npy and y.npy")

# ==== Run it ====
if __name__ == "__main__":
    extract_features()
