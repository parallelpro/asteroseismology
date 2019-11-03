import os

__all__ = ["sep"]

sep = "\\" if os.name=="nt" else "/"
