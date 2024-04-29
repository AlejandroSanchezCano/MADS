# Built-in libraries


# Third-party libraries
import torch
from transformers import T5Tokenizer, T5EncoderModel

# Custom libraries
from logger import logger

class ProtTrans:

    # huggingface conda environment
    # pip install --upgrade transformers py3Dmol accelerate
    # pip install --upgrade nvitop
    # mamba install scipy
    # mamba install -c conda-forge matplotlib
    # mamba install seaborn -c conda-forge
    # pip install sentencepiece  
    # pip install protobuf

    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.tokenizer, self.model = self.load()

    def load(self) -> tuple[T5Tokenizer, T5EncoderModel]:

        # Load model and tokenizer
        tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
        model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(self.device)

        # Logger
        logger.info('Tokenizer and model are loaded')

        return tokenizer, model  
    
    def generate_embedding(self, *seqs: list[str]) -> torch.Tensor:
        
        seqs = ["PRTwesgddgfdEINO", "SEQWENCE"]
        # Tokenize sequences
        ids = self.tokenizer(seqs, add_special_tokens=False, padding="longest")
        input_ids = torch.tensor(ids['input_ids']).to(self.device)
        attention_mask = torch.tensor(ids['attention_mask']).to(self.device)

        # Generate embeddings
        with torch.no_grad():
            embedding_repr = self.model(input_ids = input_ids, attention_mask = attention_mask)

        emb_0 = embedding_repr.last_hidden_state[0,:7] 
        emb_0_per_protein = emb_0.mean(dim=0)
        sep3 = embedding_repr.last_hidden_state[0:7].mean(dim=0)
        pi = embedding_repr.last_hidden_state[1,:7].mean(dim=0)
        print(sep3 == pi, sum(sep3 == pi).sum())
        print(emb_0, emb_0.shape)
        print(emb_0_per_protein, emb_0_per_protein.shape)
        print(embedding_repr.last_hidden_state, embedding_repr.last_hidden_state.shape)
        print(embedding_repr.last_hidden_state[1,:8], embedding_repr.last_hidden_state[1,:8].shape)

        # Logging
        logger.info('Embeddings are generated')

    
if __name__ == '__main__':
    sep3 = 'MGRGRVELKRIENKINRQVTFAKRRNGLLKKAYELSVLCDAEVALIIFSNRGKLYEFCSSSSMLRTLERYQKCNYGAPEPNVPSREALAVELSSQQEYLKLKERYDALQRTQRNLLGEDLGPLSTKELESLERQLDSSLKQIRALRTQFMLDQLNDLQSKERMLTETNKTLRLRLADGYQMPLQLNPNQEEVDHYGRHHHQQQQHSQAFFQPLECEPILQIGYQGQQDGMGAGPSVNNYMLGWLPYDTNSI'
    pi = 'MGRGKIEIKRIENANNRVVTFSKRRNGLVKKAKEITVLCDAKVALIIFASNGKMIDYCCPSMDLGAMLDQYQKLSGKKLWDAKHENLSNEIDRIKKENDSLQLELRHLKGEDIQSLNLKNLMAVEHAIEHGLDKVRDHQMEILISKRRNEKMMAEEQRQLTFQLQQQEMAIASNARGMMMRDHDGQFGYRVQPIQPNLQEKIMSLVID'

    pt = ProtTrans()
    pt = pt.generate_embedding(sep3, pi)

