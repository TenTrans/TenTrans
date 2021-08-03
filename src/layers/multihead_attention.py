import math
import torch
import torch.nn as nn
from torch import Tensor
import itertools


class MultiHeadedAttention(nn.Module):
    ID = itertools.count()

    def __init__(self, num_heads: int, size: int, dropout: float = 0.1, ):
        """
        Create a multi-headed attention layer.
        :param num_heads: the number of heads
        :param size: model size (must be divisible by num_heads)
        :param dropout: probability of dropping a unit
        """
        super().__init__()

        self.id = next(MultiHeadedAttention.ID)
        assert size % num_heads == 0

        self.head_size = head_size = size // num_heads
        self.model_size = size
        self.num_heads = num_heads

        self.k_layer = nn.Linear(size, num_heads * head_size)
        self.v_layer = nn.Linear(size, num_heads * head_size)
        self.q_layer = nn.Linear(size, num_heads * head_size)

        self.output_layer = nn.Linear(size, size)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()
        
    def forward(self,
                k: Tensor,
                v: Tensor,
                q: Tensor,
                mask: Tensor = None,
                cache=None):
        """
        Computes multi-headed attention.
        :param k: keys   [B, M, D] with M being the sentence length.
        :param v: values [B, M, D]
        :param q: query  [B, M, D]
        :param mask: optional mask [B, 1, M]
        :return:
        """
        batch_size = k.size(0)
        num_heads = self.num_heads

        # project the queries (q), keys (k), and values (v)
        q = self.q_layer(q)
        if cache is not None:
            if self.id in cache:
                if k.size() == q.size() and q.size(1) == 1: # cross attention
                    k, v = self.k_layer(k), self.v_layer(v)
                    k_, v_ = cache[self.id]
                    k = torch.cat([k_, k],
                                  dim=2)  
                    v = torch.cat([v_, v],
                                  dim=2)  
                else: # encoder attention
                    k, v = cache[self.id]  
            else: # cache the first computation
                k, v = self.k_layer(k), self.v_layer(v)
            cache[self.id] = k, v
        else:
            k, v = self.k_layer(k), self.v_layer(v)


        # reshape q, k, v for our computation to [batch_size, num_heads, seq_len, head_dim]
        k = k.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        q = q.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)

        # compute scores
        q = q / math.sqrt(self.head_size)

        # batch x num_heads x query_len x key_len
        scores = torch.matmul(q, k.transpose(2, 3))
        # mask
        # we add a dimension for the heads to it below: [B, 1, 1, M]
        # ~ equals to (1-mask)
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float('-inf'))

        # apply attention dropout and compute context vectors.
        attention = self.softmax(scores)
        attention = self.dropout(attention)

        # get context vector (select values with attention) and reshape
        # back to [B, M, D]
        context = torch.matmul(attention, v)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, num_heads * self.head_size)
        output = self.output_layer(context)

        return output


    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_layer.weight)
        nn.init.xavier_uniform_(self.q_layer.weight)
        nn.init.xavier_uniform_(self.v_layer.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)