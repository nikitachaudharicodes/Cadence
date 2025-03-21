import tensorflow as tf
import numpy as np
import os

def parse_tfrecord_fn(proto):
    context_features = {
        "video_id": tf.io.FixedLenFeature([], tf.string),
        "start_time_seconds": tf.io.FixedLenFeature([], tf.float32),
        "end_time_seconds": tf.io.FixedLenFeature([], tf.float32),
        "labels": tf.io.VarLenFeature(tf.int64),
    }

    sequence_features = {
        "audio_embedding": tf.io.VarLenFeature(tf.string),
    }

    context, sequence = tf.io.parse_single_sequence_example(
        proto,
        context_features=context_features,
        sequence_features=sequence_features
    )

    return {
        "video_id": context["video_id"],
        "start_time_seconds": context["start_time_seconds"],
        "end_time_seconds": context["end_time_seconds"],
        "labels": tf.sparse.to_dense(context["labels"]),
        "audio_embedding": tf.sparse.to_dense(sequence["audio_embedding"]),
    }

def inspect_tfrecord(tfrecord_path, max_examples=3):
    print(f"📁 Inspecting TFRecord: {tfrecord_path}\n")
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    parsed = dataset.map(parse_tfrecord_fn)

    for i, sample in enumerate(parsed):
        video_id = sample["video_id"].numpy().decode("utf-8")
        start = sample["start_time_seconds"].numpy()
        end = sample["end_time_seconds"].numpy()
        labels = sample["labels"].numpy().tolist()
        raw_embeddings = sample["audio_embedding"].numpy()

        print(f"🎥 Video ID: {video_id}")
        print(f"🕒 Segment: {start:.1f} – {end:.1f}")
        print(f"🏷 Labels: {labels}")
        print(f"🎵 Embedding frames: {len(raw_embeddings)}")

        for j, e in enumerate(raw_embeddings[:3]):  # show first 3 frames
            arr = np.frombuffer(e, dtype=np.uint8)
            print(f"  🔹 Frame {j+1}: shape={arr.shape}, values={arr[:10]} ...")  # show first 10 values

        print("-" * 60)

        if i + 1 >= max_examples:
            break

# Replace this with any TFRecord you want to inspect
if __name__ == "__main__":
    tfrecord_path = "data/embeddings/audioset_v1_embeddings/bal_train/0A.tfrecord"
    if not os.path.exists(tfrecord_path):
        print("❌ TFRecord not found. Update the path.")
    else:
        inspect_tfrecord(tfrecord_path)
