
import tensorflow as tf


class QueryTower(tf.keras.Model):

    def __init__(
        self,
        customer_vocab,
        club_vocab,
        news_vocab,
        embedding_dim=32
    ):
        super().__init__()

        self.customer_lookup = tf.keras.layers.StringLookup(
            vocabulary=customer_vocab,
            mask_token=None
        )

        self.club_lookup = tf.keras.layers.StringLookup(
            vocabulary=club_vocab,
            mask_token=None
        )

        self.news_lookup = tf.keras.layers.StringLookup(
            vocabulary=news_vocab,
            mask_token=None
        )

        self.customer_embedding = tf.keras.layers.Embedding(
            self.customer_lookup.vocabulary_size(),
            embedding_dim
        )

        self.club_embedding = tf.keras.layers.Embedding(
            self.club_lookup.vocabulary_size(),
            8
        )

        self.news_embedding = tf.keras.layers.Embedding(
            self.news_lookup.vocabulary_size(),
            8
        )

        self.dense1 = tf.keras.layers.Dense(64, activation="relu")
        self.dense2 = tf.keras.layers.Dense(embedding_dim)

    def call(self, inputs):

        customer_id = self.customer_lookup(inputs["customer_id"])
        club_status = self.club_lookup(inputs["club_status"])
        news_frequency = self.news_lookup(inputs["news_frequency"])

        customer_vector = self.customer_embedding(customer_id)
        club_vector = self.club_embedding(club_status)
        news_vector = self.news_embedding(news_frequency)

        age = tf.cast(inputs["age"], tf.float32)
        age = tf.expand_dims(age, axis=-1) / 100.0

        combined = tf.concat(
            [customer_vector, club_vector, news_vector, age],
            axis=1
        )

        x = self.dense1(combined)

        return self.dense2(x)
