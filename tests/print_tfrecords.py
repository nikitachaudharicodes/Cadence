import tensorflow as tf
import glob


import tensorflow as tf
import tensorflow as tf

def parse_sequence_example(example_proto):
    context_features = {
        'video_id': tf.io.FixedLenFeature([], tf.string),
        'start_time_seconds': tf.io.FixedLenFeature([], tf.float32),
        'end_time_seconds': tf.io.FixedLenFeature([], tf.float32),
        'labels': tf.io.VarLenFeature(tf.int64)
    }

    sequence_features = {
        'audio_embedding': tf.io.FixedLenSequenceFeature([], tf.string)
    }

    context, sequence = tf.io.parse_single_sequence_example(
        example_proto,
        context_features=context_features,
        sequence_features=sequence_features
    )
    return context, sequence


tfrecord_path = "data/embeddings/audioset_v1_embeddings/bal_train/*.tfrecord"
files = glob.glob(tfrecord_path)

for tfrecord_file in files[:1]:  # just 1 file to test
    dataset = tf.data.TFRecordDataset(tfrecord_file)
    
    for serialized_example in dataset.take(1):
        context, sequence = parse_sequence_example(serialized_example)

        print("🎥 Video ID:", context["video_id"].numpy().decode())
        print("🕒 Start:", context["start_time_seconds"].numpy())
        print("🕒 End:", context["end_time_seconds"].numpy())
        print("🏷 Labels:", tf.sparse.to_dense(context["labels"]).numpy())
        print("🎵 Embeddings shape:", len(sequence["audio_embedding"]))
        break
