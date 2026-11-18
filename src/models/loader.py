"""Rebuild the trained towers from model_config.json + saved weights."""
import json

import numpy as np
import tensorflow as tf

from src import config
from src.models.candidate_tower import CandidateTower
from src.models.query_tower import QueryTower


def load_model_config(path=config.MODEL_CONFIG_FILE):
    with open(path) as f:
        return json.load(f)


def dummy_query_inputs(cfg, n=1):
    return {
        "user_idx": tf.zeros([n], tf.int32),
        "club": tf.zeros([n], tf.int32),
        "news": tf.zeros([n], tf.int32),
        "month": tf.ones([n], tf.int32),
        "day_of_week": tf.ones([n], tf.int32),
        "user_num": tf.zeros([n, len(cfg["user_num_cols"])], tf.float32),
    }


def dummy_candidate_inputs(cfg, n=1):
    return {
        "item_idx": tf.zeros([n], tf.int32),
        "cats": tf.zeros([n, len(cfg["item_cat_cols"])], tf.int32),
        "item_num": tf.zeros([n, len(cfg["item_num_cols"])], tf.float32),
    }


def load_query_tower(cfg=None, weights=config.QUERY_WEIGHTS_FILE):
    cfg = cfg or load_model_config()
    tower = QueryTower(cfg["n_users"], cfg["n_club"], cfg["n_news"],
                       cfg["embedding_dim"], cfg["id_dim"], cfg["id_dropout"])
    tower(dummy_query_inputs(cfg), training=False)      # build the weights
    tower.load_weights(weights)
    return tower


def load_candidate_tower(cfg=None, weights=config.CANDIDATE_WEIGHTS_FILE):
    cfg = cfg or load_model_config()
    tower = CandidateTower(cfg["n_items"], cfg["item_cat_sizes"],
                           cfg["embedding_dim"], cfg["id_dim"], cfg["id_dropout"])
    tower(dummy_candidate_inputs(cfg), training=False)
    tower.load_weights(weights)
    return tower


def load_trained_users(path=config.TRAINED_USERS_FILE):
    return set(np.load(path).tolist())
