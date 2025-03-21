import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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
        sequence_features=sequence_features,
    )
    return {
        "video_id": context["video_id"],
        "start_time_seconds": context["start_time_seconds"],
        "end_time_seconds": context["end_time_seconds"],
        "labels": tf.sparse.to_dense(context["labels"]),
        "audio_embedding": tf.sparse.to_dense(sequence["audio_embedding"]),
    }

def load_embeddings(tfrecord_path, max_samples=5):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    parsed_dataset = dataset.map(parse_tfrecord_fn)

    all_embeddings = []
    metadata = []

    for i, sample in enumerate(parsed_dataset):
        video_id = sample["video_id"].numpy().decode("utf-8")
        start = sample["start_time_seconds"].numpy()
        end = sample["end_time_seconds"].numpy()
        labels = sample["labels"].numpy().tolist()
        raw_frames = sample["audio_embedding"].numpy()

        # Convert bytes -> uint8 vectors -> stack
        frames = [np.frombuffer(f, dtype=np.uint8) for f in raw_frames]
        embeddings = np.stack(frames)
        all_embeddings.append(embeddings)
        metadata.append(f"{video_id}_{int(start)}-{int(end)} | {labels}")

        if i + 1 >= max_samples:
            break

    return all_embeddings, metadata

def visualize_embeddings(embeddings_list, metadata):
    flat_embeddings = np.vstack(embeddings_list)
    labels = []
    for i, meta in enumerate(metadata):
        labels += [meta] * 10  # 10 frames per segment

    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(flat_embeddings)

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=10, random_state=42)
    tsne_result = tsne.fit_transform(flat_embeddings)

    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    axs[0].scatter(pca_result[:, 0], pca_result[:, 1], c='blue', s=8)
    axs[0].set_title("PCA of Audio Embeddings")
    axs[1].scatter(tsne_result[:, 0], tsne_result[:, 1], c='green', s=8)
    axs[1].set_title("t-SNE of Audio Embeddings")
    plt.suptitle("AudioSet Embedding Visualization (Sampled Segments)")
    plt.tight_layout()
    plt.show()

# === Run ===
if __name__ == "__main__":
    tfrecord_path = "data/embeddings/audioset_v1_embeddings/bal_train/0A.tfrecord"
    if os.path.exists(tfrecord_path):
        embeddings, meta = load_embeddings(tfrecord_path)
        visualize_embeddings(embeddings, meta)
    else:
        print("❌ TFRecord not found at expected path.")
