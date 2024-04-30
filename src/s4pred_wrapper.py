# Built-in libraries
from typing import Literal

# Third-party libraries
import torch
import numpy as np

# Custom libraries
from logger import logger
from s4pred.network import S4PRED

class S4PREDWrapper:

    def __init__(self, threads: int, device: Literal['cpu', 'gpu']):
        self.threads = torch.set_num_threads(threads)
        self.device = 'cuda:0' if torch.cuda.is_available() and device == 'gpu' else 'cpu'
        self.model = self.load()
        
    def load(self) -> S4PRED:
        '''
        Load S4PRED model.

        Returns
        -------
        S4PRED
            S4PRED model.
        '''
        # Prepare model
        model = S4PRED().to(self.device)
        model.eval()
        model.requires_grad = False

        # Load model parameters
        weight_files = [
            '/home/asanchez/fold/s4pred/weights/weights_1.pt',
            '/home/asanchez/fold/s4pred/weights/weights_2.pt',
            '/home/asanchez/fold/s4pred/weights/weights_3.pt',
            '/home/asanchez/fold/s4pred/weights/weights_4.pt',
            '/home/asanchez/fold/s4pred/weights/weights_5.pt'
        ]
        model.model_1.load_state_dict(torch.load(weight_files[0], map_location=lambda storage, loc: storage))
        model.model_2.load_state_dict(torch.load(weight_files[1], map_location=lambda storage, loc: storage))
        model.model_3.load_state_dict(torch.load(weight_files[2], map_location=lambda storage, loc: storage))
        model.model_4.load_state_dict(torch.load(weight_files[3], map_location=lambda storage, loc: storage))
        model.model_5.load_state_dict(torch.load(weight_files[4], map_location=lambda storage, loc: storage))

        # Logger
        logger.info('Model is loaded')

        return model

    def aa2int(self, seq: str) -> list[int]:
        '''
        Convert amino acid sequence to numerical sequence based on specific
        dictionary {A:0, R:1, N:2, D:3, C:4, Q:5, E:6, G:7, H:8, I:9, L:10, 
        K:11, M:12, F:13, P:14, S:15, T:16, W:17, Y:18, V:19}

        Parameters
        ----------
        seq : str
            Protein sequence

        Returns
        -------
        list[int]
            Numerical sequence
        '''
        aanumdict = {
            'A':0, 'R':1, 'N':2, 'D':3, 'C':4, 'Q':5, 'E':6, 'G':7, 'H':8, 
            'I':9, 'L':10, 'K':11, 'M':12, 'F':13, 'P':14, 'S':15, 'T':16,
            'W':17, 'Y':18, 'V':19
            }
        return [aanumdict.get(res, 20) for res in seq]

    def predict(self, seq: str) -> tuple[np.ndarray, np.ndarray]:   
        '''
        Predict secondary structure of a protein sequence with S4PRED model.

        Parameters
        ----------
        seq : str
            Protein sequence

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Secondary structure {0: 'coil', 1: 'helix', 2: 'sheet'} and 
            confidence for each type per position.
        '''
        
        # Numerize the sequence
        numerical_seq = self.aa2int(seq)

        # Predict secondary structure
        with torch.no_grad():
            ss_conf = self.model(torch.tensor(numerical_seq).to(self.device))
            ss = ss_conf.argmax(-1)
            ss_conf = ss_conf.exp()
            tsum = ss_conf.sum(-1)
            tsum = torch.cat((tsum.unsqueeze(1),tsum.unsqueeze(1),tsum.unsqueeze(1)),1)
            ss_conf /= tsum
            ss = ss.cpu().numpy()
            ss_conf = ss_conf.cpu().numpy()

        # Logger
        logger.info('Secondary structure is predicted')

        return ss, ss_conf
    
    def remove_cterminal(self, seq: str,  ss: np.ndarray, slack: int) -> str:
        '''
        Remove MADS-box C-terminal residues from a protein sequence by 
        identifying the longest helix and assuming it's the last one of the 
        K-domain.

        Parameters
        ----------
        seq : str
            Protein sequence
        ss : np.ndarray
            Secondary structure {0: 'coil', 1: 'helix', 2: 'sheet'} per position.
        slack : int
            S4PRED's output's deviates from AlphaFold's output by a few 
            residues, so we can allow some slack, include a few more residues 
            and not being that stringent.

        Returns
        -------
        str
            Protein sequence without the C-terminal domain.
        '''
        # Replace non-ones with zeros
        ss[ss != 1] = 0
        # Difference between positions i and i - 1
        diff = np.diff(ss)
        # Start and end of the helixes
        start = np.where(diff == 1)[0]
        end = np.where(diff == -1)[0]
        # End of the longest helix
        end_khelix = end[np.argmax(end-start)]

        # Logger
        logger.info('C-terminal residues are removed')

        return seq[:end_khelix + slack]

if __name__ == '__main__':
    sep3 = 'MGRGRVELKRIENKINRQVTFAKRRNGLLKKAYELSVLCDAEVALIIFSNRGKLYEFCSSSSMLRTLERYQKCNYGAPEPNVPSREALAVELSSQQEYLKLKERYDALQRTQRNLLGEDLGPLSTKELESLERQLDSSLKQIRALRTQFMLDQLNDLQSKERMLTETNKTLRLRLADGYQMPLQLNPNQEEVDHYGRHHHQQQQHSQAFFQPLECEPILQIGYQGQQDGMGAGPSVNNYMLGWLPYDTNSI'

    s4pred = S4PREDWrapper(threads=4, device='gpu') 
    ss, _ = s4pred.predict(sep3)
    seq = s4pred.remove_cterminal(sep3, ss, 2)
    print(seq)
    