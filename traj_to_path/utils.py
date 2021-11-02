from pathlib import Path
import numpy as np

def parse_mat(mat_path: str) -> np.ndarray:

    """
    Parse AMBER correlation or distance matrix.
    
    Parameters
    ----------
    mat_path : str
        Path to matrix file
    
    Returns
    -------
    mat : np.array
    """

    content = Path(mat_path).read_text().splitlines()

    res = []
    for line in content:
        res.append(line.strip().split())

    mat = np.array(res, dtype=np.float)

    return mat
