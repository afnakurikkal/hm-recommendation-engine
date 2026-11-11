
import tensorflow as tf

class TwoTowerModel(tf.keras.Model):

    def __init__(self, query_tower, candidate_tower):
        super().__init__()

        self.query_tower = query_tower
        self.candidate_tower = candidate_tower

    def call(self, inputs):

        user_embeddings = self.query_tower(inputs["customer"])

        item_embeddings = self.candidate_tower(inputs["candidate"])

        scores = tf.matmul(
            user_embeddings,
            item_embeddings,
            transpose_b=True
        )

        return scores
