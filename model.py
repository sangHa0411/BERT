import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PaddingMask(nn.Module) :
    def __init__(self) :
        super(PaddingMask , self).__init__() 
    
    def forward(self, in_tensor) :
        batch_size, seq_size = in_tensor.shape
        flag_tensor = torch.where(in_tensor == 0.0 , 1.0 , 0.0)
        flag_tensor = torch.reshape(flag_tensor , (batch_size, 1, 1, seq_size)) 
        return flag_tensor

# Multihead Attention Layer
class MultiHeadAttention(nn.Module) :
    def __init__(self, d_model, num_heads) :
        super(MultiHeadAttention , self).__init__()
        self.d_model = d_model # vector size 
        self.num_heads = num_heads # head_size
        self.depth = int(d_model / num_heads)

        self.q_layer = nn.Linear(d_model, d_model)
        self.k_layer = nn.Linear(d_model, d_model)
        self.v_layer = nn.Linear(d_model, d_model)
        self.o_layer = nn.Linear(d_model, d_model)

        self.scale = torch.sqrt(torch.tensor(self.depth , dtype=torch.float32 , requires_grad=False))

    # tensor shape : (batch_size, sen_size, d_model)
    def split(self, tensor) :
        sen_size = tensor.shape[1]
        tensor = torch.reshape(tensor, (-1, sen_size, self.num_heads, self.depth))     
        tensor = tensor.permute(0,2,1,3) # (batch_size, num_heads, sen_size, depth)
        return tensor

    # tensor shape : (batch_size, num_heads, sen_size, dk)
    def merge(self, tensor) :
        sen_size = tensor.shape[2]
        tensor = tensor.permute(0,2,1,3) # (batch_size, sen_size, num_heads, depth)
        tensor = torch.reshape(tensor, (-1, sen_size, self.d_model)) #(batch_size, sen_size, embedding_dim)
        return tensor
    
    # scaled dot production
    def scaled_dot_production(self, q_tensor, k_tensor, v_tensor, m_tensor) :
        q_tensor = self.split(q_tensor)
        k_tensor = self.split(k_tensor)
        v_tensor = self.split(v_tensor)
        
        k_tensor_T = k_tensor.permute(0,1,3,2) # (batch_size, num_heads, depth, sen_size)

        qk_tensor = torch.matmul(q_tensor , k_tensor_T) # (batch_size, num_heads, sen_size, sen_size)
        qk_tensor /= self.scale

        if m_tensor != None :
            qk_tensor -= (m_tensor*1e+6)
            
        qk_tensor = F.softmax(qk_tensor, dim=-1)
        att = torch.matmul(qk_tensor, v_tensor) # (batch_size, num_heads, sen_size, depth)

        return att

    def forward(self, q_in, k_in, v_in, m_in) :
        q_tensor = self.q_layer(q_in)
        k_tensor = self.k_layer(k_in)
        v_tensor = self.v_layer(v_in)
        
        att_tensor = self.scaled_dot_production(q_tensor , k_tensor , v_tensor , m_in)
        multi_att_tensor = self.merge(att_tensor)
        
        o_tensor = self.o_layer(multi_att_tensor)
        return o_tensor

# Feedforward layer
class FeedForward(nn.Module) :
    def __init__(self, hidden_size, d_model) :
        super(FeedForward , self).__init__()
        self.hidden_size = hidden_size
        self.d_model = d_model
        # gelu activation and input, output dim are same
        self.ff = nn.Sequential(nn.Linear(d_model , hidden_size), 
                                nn.GELU(),
                                nn.Linear(hidden_size , d_model))

    def forward(self , in_tensor) :
        o_tensor = self.ff(in_tensor)
        return o_tensor

# Transformer Encoder Block
class EncoderBlock(nn.Module) :
    def __init__(self, d_model, num_heads, hidden_size, drop_rate, norm_rate) :
        super(EncoderBlock, self).__init__()
        # multihead attention layer & feedforward layer
        self.mha_layer = MultiHeadAttention(d_model , num_heads)
        self.ff_layer = FeedForward(hidden_size , d_model)
        # dropout layer & layer normalization layer
        self.drop1_layer = nn.Dropout(drop_rate)
        self.norm1_layer = nn.LayerNorm(d_model, eps=norm_rate)
        self.drop2_layer = nn.Dropout(drop_rate)
        self.norm2_layer = nn.LayerNorm(d_model, eps=norm_rate)
                
    def forward(self, in_tensor, m_tensor) :
        # mutlihead attention sub layer
        mha_tensor = self.mha_layer(in_tensor , in_tensor , in_tensor , m_tensor)
        mha_tensor = self.drop1_layer(mha_tensor)
        h_tensor = self.norm1_layer(in_tensor + mha_tensor)
        # feed forward sub layer
        ff_tensor = self.ff_layer(h_tensor)
        ff_tensor = self.drop2_layer(ff_tensor)
        o_tensor = self.norm2_layer(h_tensor + ff_tensor)

        return o_tensor

# Transformer Encoder
class TransformerEncoder(nn.Module) :
    def __init__(self, layer_size, max_size, v_size, d_model, num_heads, hidden_size, drop_rate, norm_rate) :
        super(TransformerEncoder , self).__init__()
        self.layer_size = layer_size
        self.max_size = max_size
        self.v_size = v_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.drop_rate = drop_rate
        self.norm_rate = norm_rate

        self.em = nn.Embedding(num_embeddings=v_size, 
            embedding_dim=d_model, 
            padding_idx=0
        ) # id embedding
        self.type_em = nn.Embedding(num_embeddings=3,
            embedding_dim=d_model,
            padding_idx=0
        ) # token type embedding
        self.pos_em = nn.Embedding(num_embeddings=max_size+1,
            embedding_dim=d_model,
            padding_idx=0
        ) # positional embedding

        self.en_blocks = nn.ModuleList()
        self.drop_layer = nn.Dropout(drop_rate)
        self.norm_layer = nn.LayerNorm(d_model , eps=norm_rate)

        for i in range(layer_size) :
            self.en_blocks.append(EncoderBlock(d_model, num_heads, hidden_size, drop_rate, norm_rate))
        self.o_layer = nn.Linear(d_model, v_size)
        self.sop_layer = nn.Linear(d_model, 3)
        
        self.init_param()
        
    def init_param(self) :
        for p in self.parameters() :
            if p.dim() > 1 :
                nn.init.xavier_uniform_(p)

    def get_feature(self, input_id_tensor, pos_id_tensor, type_id_tensor, mask_id_tensor) :
        # encoder input tensor
        em_tensor = self.em(input_id_tensor) # embedding
        pos_tensor = self.pos_em(pos_id_tensor) # positional embedding
        type_tensor = self.type_em(type_id_tensor) # type embedding

        en_tensor = em_tensor + pos_tensor + type_tensor # encoded tensor
        en_tensor = self.drop_layer(en_tensor) # dropout layer
        
        tensor_ptr = en_tensor
        for i in range(self.layer_size) :
            tensor_ptr = self.en_blocks[i](tensor_ptr, mask_id_tensor)
        return tensor_ptr


    # input_ids tensor
    # attention_mask tensor    
    def forward(self, input_id_tensor, pos_id_tensor, type_id_tensor, mask_id_tensor) :
        feature_tensor = self.get_feature(input_id_tensor, pos_id_tensor, type_id_tensor, mask_id_tensor)
        cls_tensor = feature_tensor[:,0,:]

        output_tensor = self.o_layer(feature_tensor)
        sop_tensor = self.sop_layer(cls_tensor)
        return sop_tensor, output_tensor
