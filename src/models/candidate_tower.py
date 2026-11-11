
import tensorflow as tf


class CandidateTower(tf.keras.Model):

    def __init__(self, article_vocab, metadata_vocabs, embedding_dim=32):
        super().__init__()

        self.article_lookup = tf.keras.layers.StringLookup(
            vocabulary=article_vocab,
            mask_token=None
        )

        self.article_embedding = tf.keras.layers.Embedding(
            self.article_lookup.vocabulary_size(), embedding_dim
        )

        self.metadata_lookups = {}
        self.metadata_embeddings = {}

        for feature, vocab in metadata_vocabs.items():
            self.metadata_lookups[feature] = tf.keras.layers.StringLookup(
                vocabulary=vocab,
                mask_token=None
            )

            self.metadata_embeddings[feature] = tf.keras.layers.Embedding(
                self.metadata_lookups[feature].vocabulary_size(), 8
            )

        self.dense1 = tf.keras.layers.Dense(64, activation="relu")
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(embedding_dim)

    def call(self, inputs, training=False):

        article_id = self.article_lookup(inputs["article_id"])
        vectors = [self.article_embedding(article_id)]

        for feature in self.metadata_lookups:
            indices = self.metadata_lookups[feature](inputs[feature])
            vectors.append(self.metadata_embeddings[feature](indices))

        combined = tf.concat(vectors, axis=1)

        x = self.dense1(combined)
        x = self.dropout(x, training=training)

        return self.dense2(x)
