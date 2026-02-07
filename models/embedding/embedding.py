"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""

"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
from torch import nn
from models.embedding.positional_encoding import PositionalEncoding


class Embedding(nn.Module):
    """
    token embedding + positional encoding (sinusoid)
    positional encoding can give positional information to network
    """

    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        """
        class for word embedding that included positional information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        super(Embedding, self).__init__()
        #TokenEmbedding
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=1)
        self.pos_emb = PositionalEncoding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        #  we apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks.
        return self.drop_out(tok_emb + pos_emb)

