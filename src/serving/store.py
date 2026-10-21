"""Connection to the feature store.

REDIS_URL=redis://host:6379/0   real Redis (default, used in production / Docker)
REDIS_URL=memory://             in-process store (fakeredis) - for laptops without
                                Redis and for the automated tests. The API loads
                                the features into it at start-up.
"""
import redis

from src import config

_memory_server = None


def is_memory(url=None):
    return (url or config.REDIS_URL).startswith("memory://")


def connect(url=None, check=True):
    url = url or config.REDIS_URL
    if is_memory(url):
        import fakeredis

        global _memory_server
        if _memory_server is None:
            _memory_server = fakeredis.FakeServer()
        return fakeredis.FakeRedis(server=_memory_server)
    client = redis.Redis.from_url(url, socket_connect_timeout=5, health_check_interval=30)
    if check:
        client.ping()
    return client
