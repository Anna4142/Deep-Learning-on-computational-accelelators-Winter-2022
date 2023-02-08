import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.functional import dropout
import copy
import numpy as np
from torch.autograd import Variable

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    # TODO: Implement a mask for the second attention in the decoder layer 
    # ====== YOUR CODE: ======
    #The function takes a sequence seq and creates a triangular mask of ones and zeros of size len_s x len_s where the diagonal and above is     #1 and below is 0. This mask is then expanded to have the same batch size as the input sequence seq.
    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(torch.ones((len_s, len_s), device=seq.device), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    # ========================

 

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.  
class PositionalEncoder(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoder, self).__init__()
        # TODO:
        # ====== YOUR CODE: ======
        #A common way to do this is by using sine and cosine functions of different frequencies to create a unique encoding for each position in the sequence.
        position_enc = torch.tensor([[pos / np.power(10000, 2 * i / d_hid) for i in range(d_hid)] for pos in range(n_position)])

        # Make position encoding of dimension (n_position, d_hid)
        position_enc[:, 0::2] = torch.sin(position_enc[:, 0::2])
        position_enc[:, 1::2] = torch.cos(position_enc[:, 1::2])
        position_enc = position_enc.unsqueeze(1)
        self.register_buffer('position_enc', position_enc)
        
  
        
        return x

        # ========================

    def forward(self, x):
        # TODO: implment a positional encoding
        # ====== YOUR CODE: ======
        x = x + self.position_enc[:x.size(1), :].to(x.device)
        # ========================

    
def attention(q, k, v, d_k, mask=None):
    attn = torch.matmul(q /  math.sqrt(d_k) , k.transpose(2, 3))
    if mask is not None:
        attn = attn.masked_fill(mask == 0, -1e9)

    attn = F.softmax(attn, dim=-1)
    output = torch.matmul(attn, v)

    return output, attn


    
class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model , dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_model // n_head

        self.w_qs = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.fc = nn.Linear(n_head * self.d_k, d_model, bias=False)
        
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):
        d_k,  n_head = self.d_k,  self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q
        # TODO:
        # Pass through the pre-attention projection: b x lq x (n*dk)
        # Separate different heads: b x lq x n x dv
        # Transpose for attention dot product: b x n x lq x dv
        # Apply mask if needed.
        # Calculate attention using the given function
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        # Apply dropout and normalizaiton
        # ====== YOUR CODE: ======
        
        d_k, n_head = self.d_k, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q

        # Pass through the pre-attention projection
        q = self.w_qs(q)
        k = self.w_ks(k)
        v = self.w_vs(v)

        # Separate different heads
        q = q.view(sz_b, len_q, n_head, d_k)
        k = k.view(sz_b, len_k, n_head, d_k)
        v = v.view(sz_b, len_v, n_head, d_k)

        # Transpose for attention dot product
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply mask if needed
        if mask is not None:
            mask = mask.unsqueeze(1)
            q = q * mask
            k = k * mask
            v = v * mask

        # Calculate attention using the given function
        scores = self.attention(q, k, v)

        # Transpose to move the head dimension back
        scores = scores.transpose(1, 2)

        # Combine the last two dimensions to concatenate all the heads together
        concat = scores.view(sz_b, len_q, -1)

        # Apply dropout and normalization
        output = self.fc(concat)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)

        return output

        # ========================
        
    
        return q, attn


    
class FeedForward(nn.Module):
    def __init__(self, dim, d_ff=2048, dropout = 0.1):
        super().__init__() 
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(dim, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, dim)
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

    
class Norm(nn.Module):
    def __init__(self, dim, eps = 1e-6):
        super().__init__()
    
        self.size = dim
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid =2048, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x

    
class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model,  n_head,  dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model,  n_head, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, dropout=dropout)

    def forward(
            self, dec_input, enc_output,
            slf_attn_mask=None, dec_enc_attn_mask=None):
        # TODO:
        # 1. Apply self attention
        # 2. Apply cross attention
        # 3. Apply PositionwiseFeedForward 
        # ====== YOUR CODE: ======
        raise NotImplementedError()
        # ========================
        return dec_output, dec_slf_attn, dec_enc_attn
  
class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src_vocab, n_layers, n_head, 
            d_model, pad_idx, d_word_vec= 512,  dropout=0.1, n_position=200):

        super().__init__()
        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoder(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, n_head,dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.d_model = d_model

    def forward(self, src_seq, src_mask, return_attns=False):

        enc_slf_attn_list = []
        # -- Forward
        enc_output = self.src_word_emb(src_seq)

        enc_output = self.dropout(self.position_enc(enc_output))
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, n_trg_vocab, n_layers, n_head, 
            d_model, pad_idx,  d_word_vec= 512, n_position=200, dropout=0.1):

        super().__init__()

        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoder(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, n_head, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.d_model = d_model

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):
        # TODO:
        # 1. Apply word embedding on target words
        # 2. Dropout
        # 3. Normalization
        # 4. Decoder layer * n_layers
        # ====== YOUR CODE: ======
        raise NotImplementedError()
        # ========================
        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,

    


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''
    def __init__(
            self, n_src_vocab, n_trg_vocab, d_model , n_layers, n_head, pad_idx, 
            dropout=0.1, n_position=200):

        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.encoder = Encoder(
            n_src_vocab=n_src_vocab,n_layers=n_layers, n_head=n_head, d_model=d_model,pad_idx=pad_idx)
        self.decoder = Decoder(
            n_trg_vocab=n_trg_vocab, n_layers=n_layers, n_head=n_head, d_model=d_model,pad_idx=pad_idx)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 
                
        self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)

    def forward(self, src_seq, trg_seq):
        # TODO:
        # 1. Calculate Masks
        # 2. Run encoder
        # 3. Run decoder
        # ====== YOUR CODE: ======
        raise NotImplementedError()
        # ========================
        
        seq_logit = self.trg_word_prj(dec_output)

        return seq_logit.view(-1, seq_logit.size(2))