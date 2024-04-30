# Built-in libraries
from typing import Literal

# Third-party libraries
import h5py
import requests
import numpy as np

# Custom libraries
from logger import logger

class ProteinEmbeddings:

    # protembedding conda environment
    # mamba create -n protembedding python=3.10 numpy h5py requests

    def download_uniprot_embedding(
        self, 
        resource: str = 'arabidopsis', 
        type: Literal['per-residue', 'per-protein'] = 'per-protein'
        ) -> None:
        
        # Tranlsate species resource to resource code
        resources = {
            'reviewed': 'uniprot_sprot',
            'arabidopsis': 'UP000006548_3702',
            'caenoharbditis': 'UP000001940_6239',
            'escherichia': 'UP000000625_83333',
            'human': 'UP000005640_9606',
            'mouse': 'UP000000589_10090',
            'rat': 'UP000002494_10116',
            'covid': 'UP000464024_2697049'
        }
        resource_code = resources[resource]

        # Download embedding
        url = f'https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/embeddings/{resource_code}/{type}.h5'
        response = requests.get(url)

        # Save embedding
        with open(f'embedding_{resource_code}_{type}.h5', mode = "wb") as file:
            file.write(response.content)

        # Logging
        logger.info(f"Embedding for {resource} {type} downloaded successfully.")

    def explore_uniprot_embeddings(self, file_path: str) -> None:

        with h5py.File(file_path, "r") as file:
            logger.debug(f"number of entries: {len(file.items())}")
            for sequence_id, embedding in file.items():
                logger.debug(
                    f"  id: {sequence_id}, "
                    f"  embeddings shape: {embedding.shape}, "
                    f"  embeddings mean: {np.array(embedding).mean()}"
                )

if __name__ == '__main__':
    pe = ProteinEmbeddings()
    pe.download_uniprot_embedding(resource = 'arabidopsis', type = 'per-residue')
    pe.explore_uniprot_embeddings('embedding_UP000006548_3702_per-protein.h5')

