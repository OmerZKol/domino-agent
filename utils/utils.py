import gzip
import json

import itertools
import DominoGame
import numpy as np
from PIL import Image

from typing import List

def encode_board(obj):
        """Defines how to encode the board object to save it
        """
        if isinstance(obj, DominoGame.Board):
            return {'d1': obj.d1, 'd2': obj.d2, 'history': obj.history}
        return obj

def save_jsongz(board_data, filename):
    """Save data as a gzipped json file.

    Args:
        board_data: the board data to save
        filename (str): Name of file to save as.
    """
    with open(filename, "w") as outfile:
        json.dump(board_data, outfile, default=encode_board)

def load_jsongz(filename: str) -> dict:
    """Load a gzipped json file.

    Args:
        filename (str): Name of file to load.

    Returns:
        dict: Dictionary loaded from file.
    """
    print("loading file")
    with open(filename, 'r') as file:
        loaded_board = json.load(file)
    print("done")
    return loaded_board