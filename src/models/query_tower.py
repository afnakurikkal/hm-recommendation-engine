import tensorflow as tf


class QueryTower(tf.keras.Model):
    """User side of the two-tower model.

    Inputs (dict of tensors, one row per customer):
      user_idx     int    Week 1 user index (0 = unknown / not seen in training)
      club         int    club_member_status code (0 = unknown)
      news         int    fashion_news_frequency code (0 = unknown)
      month        int    1-12, month of the purchase (context)
      day_of_week  int    1-7, Sunday = 1 (context, same as Spark dayofweek)
      user_num     float  [n] standardized numeric user features
    Output: L2-normalized user embedding of size embedding_dim.
    """

    def __init__(self, n_users, n_club, n_news, embedding_dim=64,
                 id_dim=32, id_dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.id_dropout = id_dropout
        self.user_embedding = tf.keras.layers.Embedding(n_users + 1, id_dim)
        self.club_embedding = tf.keras.layers.Embedding(n_club + 1, 4)
        self.news_embedding = tf.keras.layers.Embedding(n_news + 1, 4)
        self.month_embedding = tf.keras.layers.Embedding(13, 4)
        self.dow_embedding = tf.keras.layers.Embedding(8, 4)
        self.hidden = tf.keras.layers.Dense(128, activation="relu")
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.projection = tf.keras.layers.Dense(embedding_dim)

    def call(self, inputs, training=False):
        user_idx = inputs["user_idx"]
        if training and self.id_dropout > 0:
            # Randomly hide some user IDs (set them to 0) so the shared
            # "unknown user" embedding is trained too. New customers then
            # still get recommendations based on their features.
            hide = tf.random.uniform(tf.shape(user_idx)) < self.id_dropout
            user_idx = tf.where(hide, tf.zeros_like(user_idx), user_idx)

        x = tf.concat([
            self.user_embedding(user_idx),
            self.club_embedding(inputs["club"]),
            self.news_embedding(inputs["news"]),
            self.month_embedding(inputs["month"]),
            self.dow_embedding(inputs["day_of_week"]),
            tf.cast(inputs["user_num"], tf.float32),
        ], axis=1)
        x = self.dropout(self.hidden(x), training=training)
        return tf.math.l2_normalize(self.projection(x), axis=1)
