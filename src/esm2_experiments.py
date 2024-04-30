# Built-in libraries
import os
from typing import Literal

# Third-party libraries
import torch
import pandas as pd

# Custom libraries
import utils
from esm2 import ESM2
import variables as var
from logger import logger
from interactor import Interactor

class ESM2Experiment:
    def generate_representations_for_mikc(self) -> None: 
        '''
        Generate per-residue and per-protein embeddings for MIKC proteins.
        Proteins 5515th (A0A4S4ERF5) and 10989th (A0A9E7GU77) generate 
        out-of-memory error cause they're too big (>2000 aa)
        '''
        # Initialize ESM2 model
        esm2 = ESM2()

        # Loop and load MIKC data
        for interactor in utils.iterate_folder(var.INTERACTORS):

            # Skip if already embedding already generated 
            if interactor.uniprot_id + '.pkl' in os.listdir(var.ESM2_PERPROTEIN):
                continue
            
            # Prepare data as (label, seq) tuple
            mikc_data = [(interactor.uniprot_id, interactor.seq)]

            # Run ESM2
            try:
                esm2.prepare_data(mikc_data)
                esm2.run_model()
                per_residue, per_protein = self.esm2.extract_representations()
                esm2.contact_maps(save_folder = var.ESM2_CONTACTMAPS)
            except torch.cuda.OutOfMemoryError:
                logger.warning(f'{interactor.uniprot_id} gives out-of-memory error, possibly because seq length is too large ({len(interactor.seq)} residues)')
                continue

            # Save embeddings
            utils.pickling(per_residue, f"{var.ESM2_PERRESIDUE}/{interactor.uniprot_id}.pkl")
            utils.pickling(per_protein, f"{var.ESM2_PERPROTEIN}/{interactor.uniprot_id}.pkl")

            # Logging
            logger.info(f'Embedding for {interactor.uniprot_id} has been generated and saved')

    def build_embedding(
            self, 
            type: Literal['per_resiude', 'per_protein'], 
            species: str = 'all'
            ) -> pd.DataFrame:

        # Load embeddings
        embeddings = {}
        folder = var.ESM2_PERRESIDUE if type == 'per_residue' else var.ESM2_PERPROTEIN
        for uniprot_id_embedding in utils.iterate_folder(folder):
            for uniprot_id, embedding in uniprot_id_embedding.items():
            
                # Filter by species
                interactor = Interactor.unpickle(uniprot_id)
                if species != 'all' and interactor.taxon_id != species:
                    continue
                
                # Build embedding dictionary
                embeddings[uniprot_id] = embedding

        # Prepare data in data frame
        df = pd.DataFrame.from_dict(embeddings, orient = 'index')

        # Logging
        logger.info(f'{type} embeddings for {species} species have been built')
    
        return df
    
    def species(self) -> pd.Series:

        species = {interactor.uniprot_id : interactor.taxon_id for interactor in utils.iterate_folder(var.INTERACTORS)}

        return pd.Series(species)


if __name__ == '__main__':
    esm2_experiment = ESM2Experiment()
    df = esm2_experiment.build_embedding(type = 'per_protein', species = 'all')
    species = esm2_experiment.species()
    print(species.loc[df.index])

    