import threading
class SingletonMeta(type):
    _instances = {}
    _lock = threading.Lock() # Lock shared across all singleton classes
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock: # Only one thread allowed in this block
                if cls not in cls._instances:  # Double-check
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]
