# Built-in modules
import os
from typing import Any, Generator

# Third-party modules
import pickle
from tqdm import tqdm 

# Custom modules
from interactor import Interactor
from structure import Structure

def pickling(data: Any, path: str) -> None:
    '''
    Pickle an object and store it.

    Parameters
    ----------
    data : Any
        Pickable object that will be stored.
    path : str
        Storing path.
    '''
    with open(path, 'wb') as handle:
        pickle.dump(
            obj = data,
            file = handle, 
            protocol = pickle.HIGHEST_PROTOCOL
            )

def unpickling(path: str) -> Any:
    '''
    Retrieves and unpickles a pickled object.

    Parameters
    ----------
    path : str
        Storing path of the object to unpickle.

    Returns
    -------
    Any
        Unpickled object.
    '''
    with open(path, 'rb') as handle:
        return pickle.load(file = handle)

def read_fasta_str(fasta: str) -> tuple[str, str]:
    '''
    Parses a FASTA string into its header and sequence.

    Parameters
    ----------
    fasta : str
        FASTA string.

    Returns
    -------
    tuple[str, str]
        Tuple containing the header and the sequence.
    '''
    header, sequence = fasta.split('\n', 1)
    header = header.replace('>', '')
    sequence = sequence.replace('\n', '')

    return header, sequence

def iterate_folder(folder: str, start:int = 0, limit: int = -1) -> Generator[Any, None, None]:
    for i, file in enumerate(tqdm(sorted(os.listdir(folder)))):
        if i == limit:
            break
        if i < start:
            continue
        extension = file.split('.')[-1]
        if extension == 'pkl':
            yield unpickling(f'{folder}/{file}')
        elif extension == 'int':
            yield Interactor.unpickle(file)
        elif extension == 'pdb' or extension == 'pdbpkl':
            yield Structure(file)