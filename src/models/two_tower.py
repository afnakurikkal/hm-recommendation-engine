import tensorflow as tf


class TwoTowerModel(tf.keras.Model):
    """Combines the two towers and holds the training loss."""

    def __init__(self, query_tower, candidate_tower, temperature=0.05, **kwargs):
        super().__init__(**kwargs)
        self.query_tower = query_tower
        self.candidate_tower = candidate_tower
        self.temperature = temperature

    def call(self, inputs, training=False):
        user_vectors = self.query_tower(inputs["query"], training=training)
        item_vectors = self.candidate_tower(inputs["candidate"], training=training)
        # cosine similarity of every user with every item in the batch
        return tf.matmul(user_vectors, item_vectors, transpose_b=True) / self.temperature

    def in_batch_loss(self, batch, training=False):
        """Softmax loss with in-batch negative sampling.

        Row i of the batch is (customer i, article i they bought). For
        customer i, article i is the positive and the other articles in the
        batch are the negatives.
        - Duplicates of the positive article, and other articles bought by the
          same customer, are ignored (they are not real negatives).
        - logQ correction: popular articles appear as negatives much more
          often, so their log-frequency is subtracted to avoid punishing them.
        """
        logits = self(
            {"query": batch["query"], "candidate": batch["candidate"]},
            training=training,
        )
        logits = logits - batch["log_q"][None, :]

        user_keys = batch["user_key"]
        item_keys = batch["item_key"]
        n = tf.shape(logits)[0]
        is_positive = tf.eye(n, dtype=tf.bool)
        same_item = tf.equal(item_keys[:, None], item_keys[None, :])
        same_user = tf.equal(user_keys[:, None], user_keys[None, :])
        ignore = tf.logical_and(
            tf.logical_or(same_item, same_user), tf.logical_not(is_positive)
        )
        logits = tf.where(ignore, tf.constant(-1e9, logits.dtype), logits)

        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.cast(is_positive, logits.dtype), logits=logits
        )
        return tf.reduce_mean(loss)
