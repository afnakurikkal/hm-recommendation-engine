import tensorflow as tf


class CandidateTower(tf.keras.Model):
    """Item side of the two-tower model.

    Inputs (dict of tensors, one row per article):
      item_idx  int    Week 1 item index (0 = unknown / not seen in training)
      cats      int    [4] category codes: product_type, colour_group,
                       department, garment_group (0 = unknown)
      item_num  float  [n] standardized numeric item features (popularity)
    Output: L2-normalized item embedding of size embedding_dim.
    """

    def __init__(self, n_items, cat_sizes, embedding_dim=64,
                 id_dim=32, id_dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.id_dropout = id_dropout
        self.item_embedding = tf.keras.layers.Embedding(n_items + 1, id_dim)
        self.cat_embeddings = [
            tf.keras.layers.Embedding(size + 1, 8) for size in cat_sizes
        ]
        self.hidden = tf.keras.layers.Dense(128, activation="relu")
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.projection = tf.keras.layers.Dense(embedding_dim)

    def call(self, inputs, training=False):
        item_idx = inputs["item_idx"]
        if training and self.id_dropout > 0:
            # Same idea as the query tower: train the "unknown item" embedding
            # so brand-new articles can still be recommended from metadata.
            hide = tf.random.uniform(tf.shape(item_idx)) < self.id_dropout
            item_idx = tf.where(hide, tf.zeros_like(item_idx), item_idx)

        parts = [self.item_embedding(item_idx)]
        for column, embedding in enumerate(self.cat_embeddings):
            parts.append(embedding(inputs["cats"][:, column]))
        parts.append(tf.cast(inputs["item_num"], tf.float32))

        x = tf.concat(parts, axis=1)
        x = self.dropout(self.hidden(x), training=training)
        return tf.math.l2_normalize(self.projection(x), axis=1)
