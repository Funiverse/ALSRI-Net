import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.attention import EncoderLayer




SMICharLen = 53
SEQCharLen = 21

seq_embedding_size = 64
smi_embedding_size = 64
ligand_embedding_size = pocket_embedding_size = 64


conv_filters = [[1,32],[3,32],[3,64],[5,128]] #1.改这个
conv_filters2 = [[1,8],[3,8],[3,16],[5,32]]
embedding_size = output_dim = 256
d_ff = 256
n_heads = 8
n_layer = 1


class Squeeze(nn.Module):  # Dimention Module
    def forward(self, input: torch.Tensor):
        return input.squeeze()

class ShortBasicConvBlock(nn.Module):  # Dilated Convolution
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False, dilation=d)
    def forward(self, input):
        output = self.conv(input)
        return output

class ShortConvBlock(nn.Module):
    def __init__(self, nIn, nOut, add=True):
        super().__init__()
        n = int(nOut / 4)
        n1 = nOut - 3 * n
        self.c1 = nn.Conv1d(nIn, n, 1, padding=0)
        self.br1 = nn.Sequential(nn.BatchNorm1d(n), nn.PReLU())
        self.d1 = ShortBasicConvBlock(n, n1, 3, 1, 1)  # Dilated scale:1(2^0)
        self.d2 = ShortBasicConvBlock(n, n, 3, 1, 2)  # Dilated scale:2(2^1)
        self.d4 = ShortBasicConvBlock(n, n, 3, 1, 4)  # Dilated scale:4(2^2)
        self.d8 = ShortBasicConvBlock(n, n, 3, 1, 8)  # Dilated scale:8(2^3)
        self.br2 = nn.Sequential(nn.BatchNorm1d(nOut), nn.PReLU())

        if nIn != nOut:
            add = False
        self.add = add

    def forward(self, input):

        output1 = self.c1(input)
        output1 = self.br1(output1)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)

        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8

        combine = torch.cat([d1, add1, add2, add3], 1)

        if self.add:
            combine = input + combine
        output = self.br2(combine)
        return output




# embedding
class ShortConvEmbedding(nn.Module):
    def __init__(self, embed_size,conv_filters,type):
        super().__init__()
        if type ==1:  #pocket
            self.embedding = nn.Embedding(SEQCharLen, embed_size)
            self.convolutions = nn.ModuleList()
            for kernel_size, out_channels in conv_filters:
                conv = nn.Conv1d(embed_size, out_channels, kernel_size,padding = (kernel_size - 1) // 2)
                self.convolutions.append(conv)

        if type == 2:
            self.embedding = nn.Embedding(SMICharLen, embed_size)
            self.convolutions = nn.ModuleList()
            for kernel_size, out_channels in conv_filters:
                conv = nn.Conv1d(embed_size, out_channels, kernel_size, padding=(kernel_size - 1) // 2)
                self.convolutions.append(conv)

    def forward(self, inputs):
        conv_embedding = self.embedding(inputs) # 32*66*64

        conv_embedding = torch.transpose(conv_embedding,-1,-2) #32*64*66
        conv_hidden = []
        for layer in self.convolutions:
            conv = F.relu(layer(conv_embedding))
            conv_hidden.append(conv)
        res_embed = torch.cat(conv_hidden, dim=1)
        res_embed = res_embed.transpose(-1, -2)  # (batch_size, seq_len, num_filters)


        return res_embed
class LongConvEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_size, conv_filters, output_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.convolutions = nn.ModuleList()
        for kernel_size, out_channels in conv_filters:
            conv = nn.Conv1d(embedding_size, out_channels, kernel_size, padding = (kernel_size - 1) // 2)
            self.convolutions.append(conv)
        # The dimension of concatenated vectors obtained from multiple one-dimensional convolutions
        self.num_filters = sum([f[1] for f in conv_filters])
        self.projection = nn.Linear(self.num_filters, output_dim)


    def forward(self, inputs):
        embeds = self.embed(inputs)
        embeds = embeds.transpose(-1,-2) # (batch_size, embedding_size, seq_len)
        conv_hidden = []
        for layer in self.convolutions:
            conv = F.relu(layer(embeds))
            conv_hidden.append(conv)
        res_embed = torch.cat(conv_hidden, dim = 1).transpose(-1,-2) # (batch_size, seq_len, num_filters)
        res_embed = self.projection(res_embed)
        return res_embed

class Seq_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_emb = LongConvEmbedding(SEQCharLen, embedding_size, conv_filters, output_dim)
        self.layers  = nn.ModuleList([EncoderLayer(256,128,0.1,0.0,n_heads) for _ in range(n_layer)])
    def forward(self, seq_input):
        output_emb = self.seq_emb(seq_input)
        for layer in self.layers:
            output_emb = layer(output_emb,output_emb)
        return output_emb
class Smi_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = self.emb = LongConvEmbedding(SMICharLen, embedding_size, conv_filters, output_dim)
        self.layers  = nn.ModuleList([EncoderLayer(256,128,0.1,0.1,n_heads) for _ in range(n_layer)])
    def forward(self,smi_input):
        output_emb = self.emb(smi_input)
        for layer in self.layers:
            output_emb = layer(output_emb,output_emb)
        return output_emb
class SA_Module(nn.Module):
    def __init__(self):
        super().__init__()

        self.pkt_conv_embedding = ShortConvEmbedding(seq_embedding_size,conv_filters2,1)
        self.smi_conv_embedding = ShortConvEmbedding(smi_embedding_size,conv_filters2,2)

        pkt_oc = 64
        smi_oc = 128
        conv_pkt = []
        ic = seq_embedding_size
        for oc in [32,64, pkt_oc]:
            conv_pkt.append(nn.Conv1d(ic, oc, 3))
            conv_pkt.append(nn.BatchNorm1d(oc))
            conv_pkt.append(nn.PReLU())
            ic = oc
        conv_pkt.append(nn.AdaptiveMaxPool1d(1))
        conv_pkt.append(Squeeze())
        self.conv_pkt = nn.Sequential(*conv_pkt)

        # Ligand ShortBasicConvBlock Module
        conv_smi = []
        ic = smi_embedding_size
        for oc in [32, 64, smi_oc]:
            conv_smi.append(ShortConvBlock(ic, oc))
            ic = oc
        conv_smi.append(nn.AdaptiveMaxPool1d(1))
        conv_smi.append(Squeeze())
        self.conv_smi = nn.Sequential(*conv_smi)




        # Cross-Attention Module
        self.smi_attention_poc = EncoderLayer(pocket_embedding_size, smi_embedding_size, 0.1, 0.1, 2)  # 注意力机制
        self.squeeze = Squeeze()
        self.cat_dropout = nn.Dropout(0.2)


    def forward(self, pkt, smi):
        pkt_embed = self.pkt_conv_embedding(pkt)  # 口袋（的seq）进行embedding
        smi_embed = self.smi_conv_embedding(smi)  # 配体smi进行embedding
        smi_attention = smi_embed

        smi_embed = self.smi_attention_poc(smi_embed, pkt_embed)
        pkt_embed = self.smi_attention_poc(pkt_embed, smi_attention)

        pkt_embed = torch.transpose(pkt_embed, 1, 2)
        pkt_conv = self.conv_pkt(pkt_embed)
        smi_embed = torch.transpose(smi_embed, 1, 2)
        smi_conv = self.conv_smi(smi_embed)

        concat = torch.cat([pkt_conv, smi_conv], dim=1)

        concat = self.cat_dropout(concat)
        return concat
class LA_Module(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_encoder = Seq_Encoder()
        self.smi_encoder = Smi_Encoder()

    def forward(self,seq_encode, smi_encode):
        seq_outputs = self.seq_encoder(seq_encode)
        smi_outputs = self.smi_encoder(smi_encode)
        score = torch.matmul(seq_outputs, smi_outputs.transpose(-1, -2))/np.sqrt(embedding_size)
        return score

class LSA_Module(nn.Module):
    def __init__(self):
        super().__init__()
        self.la_module = LA_Module()
        self.sa_module = SA_Module()
        self.pooling= nn.AdaptiveMaxPool1d(1)
        self.cat_dropout = nn.Dropout(0.3)
        self.linear = nn.Sequential(nn.Linear(868,192))
        self.fc = nn.Sequential(
            Squeeze(),
            nn.Linear(192,64),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.Linear(64, 1),
            Squeeze())
        self.squeeze = Squeeze()
    def forward(self,pkt, smi,seq_encode, smi_encode,poc_encode):
        sa_features = self.sa_module(pkt=pkt,smi=smi)
        la_features = self.la_module(seq_encode=seq_encode, smi_encode=smi_encode)
        la_features = self.pooling(la_features)
        la_features = self.squeeze(la_features)
        la_features = self.linear(la_features)
        concat = sa_features+la_features
        concat = self.cat_dropout(concat)
        affinity = self.fc(concat)
        return affinity


