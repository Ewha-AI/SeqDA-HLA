import torch
import torch.nn as nn

    
class SelfAlignedCrossAttention(nn.Module):
    def __init__(self, opt):
        super(SelfAlignedCrossAttention, self).__init__()
        self.device = opt.device
        self.n_heads = opt.n_heads
        self.d_model = opt.d_model
        self.d_qkv = opt.d_qkv
        self.bs = opt.batch
        
        self.W_Q = nn.Linear(self.d_model, self.d_qkv * self.n_heads, bias=False)
        self.W_K = nn.Linear(self.d_model, self.d_qkv * self.n_heads, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.fc = nn.Linear(self.n_heads * self.d_qkv, self.d_model, bias=False)
    
    def cal_Aqq_Aqs(self, qt, qs):
        Aqq = torch.matmul(qt, qt.transpose(2,3))
        Aqs = torch.matmul(qt, qs.transpose(2,3))
        return Aqq, Aqs
        
    def forward(self, hla, pep):
        residual, bs = hla, hla.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        qt = self.W_Q(hla).view(bs, -1, self.n_heads, self.d_qkv).transpose(1,2)  # [b,34,d_qkv*3]>[b,34,3,d_qkv]>[b,3,34,d_qkv]
        qs = self.W_K(pep).view(bs, -1, self.n_heads, self.d_qkv).transpose(1,2)              
        v = torch.cat([qt, qs], dim=2)
        
        aqq, aqs = self.cal_Aqq_Aqs(qt, qs)
        a = self.softmax(torch.concat([aqq, aqs], dim=3))
        t_hat_pre = torch.matmul(a, v)
        t_hat_pre = t_hat_pre + qt 

        out = t_hat_pre.transpose(1, 2).reshape(bs, -1, self.n_heads * self.d_qkv)
        out = self.fc(out)
        output = nn.LayerNorm(self.d_qkv).to(self.device)(out + residual)
        return aqq, aqs, a, output


class SelfAttention(nn.Module):
    def __init__(self, opt):
        super(SelfAttention, self).__init__()
        self.device = opt.device
        self.n_heads = opt.n_heads
        self.d_model = opt.d_model
        self.d_qkv = opt.d_qkv
        self.bs = opt.batch
        
        self.W_Q = nn.Linear(self.d_model, self.d_qkv * self.n_heads, bias=False)
        self.W_K = nn.Linear(self.d_model, self.d_qkv * self.n_heads, bias=False)
        self.W_V = nn.Linear(self.d_model, self.d_qkv * self.n_heads, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.fc = nn.Linear(self.n_heads * self.d_qkv, self.d_model, bias=False)
    
    def forward(self, xq, xk, xv):
        residual, bs = xq, xq.size(0)
        
        q = self.W_Q(xq).view(bs, -1, self.n_heads, self.d_qkv).transpose(1,2)
        k = self.W_K(xk).view(bs, -1, self.n_heads, self.d_qkv).transpose(1,2)
        v = self.W_V(xv).view(bs, -1, self.n_heads, self.d_qkv).transpose(1,2)
                                     
        a = self.softmax(torch.matmul(q, k.transpose(2,3)))
        out = torch.matmul(a, v)
        out = out.transpose(1, 2).reshape(bs, -1, self.n_heads * self.d_qkv)
        out = self.fc(out)
        output = nn.LayerNorm(self.d_model).to(self.device)(out + residual)
        return a, output


class FeedForward(nn.Module):
    def __init__(self, opt):
        super(FeedForward, self).__init__()
        self.device = opt.device
        self.d_model = opt.d_model
        self.d_ff = opt.d_ff
        self.fc = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff, bias=False),
            nn.ReLU(),
            nn.Dropout(opt.dropout_rate),
            nn.Linear(self.d_ff, self.d_model, bias=False)
        )
    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        output = self.fc(inputs)
        return output


class SeqDAHLA(nn.Module):
    def __init__(self, opt):
        super(SeqDAHLA, self).__init__()
        
        self.relu = nn.ReLU()
        self.input_fc1 = nn.Linear(opt.embedding_dim, opt.d_model)
        self.input_fc2 = nn.Linear(opt.embedding_dim, opt.d_model)
        
        self.crossAttention = SelfAlignedCrossAttention(opt)
        self.selfAttention = SelfAttention(opt)
        self.feedforward = FeedForward(opt)
        
        self.classifier = nn.Sequential(
            nn.Linear(opt.d_model*34, 512),
            nn.ReLU(True),
            nn.Dropout(opt.dropout_rate),           
            nn.BatchNorm1d(num_features=512),
            nn.Linear(512, 32),
            nn.ReLU(True),  
            nn.Dropout(opt.dropout_rate),
            nn.Linear(32, 1)
        )
        
    def forward(self, hla, pep):
        hla = self.input_fc1(hla)
        pep = self.input_fc2(pep)
        
        aqq, aqs, cross_a, hp_attention = self.crossAttention(hla, pep)
        hp_attention = self.feedforward(hp_attention)
        self_a, out = self.selfAttention(hp_attention, hp_attention, hp_attention)
        out = out.view(out.shape[0], -1)

        final_out = self.classifier(out).squeeze()
        attn = [aqq, aqs, cross_a, self_a]  # [bs, heads, 34, 34] each
        
        return final_out, attn