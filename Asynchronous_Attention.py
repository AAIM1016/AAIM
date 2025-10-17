import torch
import torch.nn as nn

class Cross_AttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_subset, lag_len, q_points,k_points, attentiondrop=0.2):
        super(Cross_AttentionBlock, self).__init__()
        self.q_points=q_points
        self.k_points=k_points
        self.lag_len = lag_len
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.num_subset = num_subset
        if self.q_points == 90 and self.k_points == 50:
            self.in_nets_q_GM = nn.Conv2d(in_channels, num_subset * out_channels, 1, bias=True)
            self.in_nets_k_WM = nn.Conv2d(in_channels, num_subset * out_channels, 1, bias=True)
            self.V_mapping_WM = nn.Conv2d(in_channels, out_channels, 1, bias=True)
            self.out_nets_GM = nn.Sequential(
                    nn.Conv2d(out_channels * num_subset, out_channels, 1, bias=True), 
                    nn.BatchNorm2d(out_channels),
                )
            self.downs_GM = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
        elif self.q_points == 50 and self.k_points == 90:
            self.in_nets_q_WM = nn.Conv2d(in_channels, num_subset * out_channels, 1, bias=True)
            self.in_nets_k_GM = nn.Conv2d(in_channels, num_subset * out_channels, 1, bias=True)
            self.V_mapping_GM = nn.Conv2d(in_channels, out_channels, 1, bias=True)
            self.out_nets_WM = nn.Sequential(
                    nn.Conv2d(out_channels * num_subset, out_channels, 1, bias=True),  
                    nn.BatchNorm2d(out_channels),
                )
            self.downs_WM = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
            
        self.softmax = nn.Softmax(-1)
        self.relu = nn.LeakyReLU(0.1)
        self.attdrop = nn.Dropout(attentiondrop)
        
    def forward(self, x_q, x_k, x_v):
        N_q, C_q, T_q, V_q = x_q.size()  
        N_k, C_k, T_k, V_k = x_k.size()
        N_v, C_v, T_v, V_v = x_v.size()
        y_k = x_k.clone()
        y_q = x_q.clone()
        Q_in=y_q[:, :, 0:(T_q - self.lag_len + 0), :].contiguous()
        if self.q_points == 90 and self.k_points == 50:
            q = self.in_nets_q_GM(Q_in).view(N_q, self.num_subset, self.out_channels, T_q - self.lag_len, V_q)
        elif self.q_points == 50 and self.k_points == 90:
            q = self.in_nets_q_WM(Q_in).view(N_q, self.num_subset, self.out_channels, T_q - self.lag_len, V_q)  
        lags = torch.arange(0, self.lag_len + 1)
        K_in_list = [y_k[:, :, l:(T_k - self.lag_len + l), :].contiguous() for l in lags]
        attention_list = []
        scale = (self.out_channels * (T_q - self.lag_len)) ** 0.5
        for K_in in K_in_list:
            if self.q_points == 90 and self.k_points == 50:
                k = self.in_nets_k_WM(K_in).view(N_k, self.num_subset, self.out_channels, T_k - self.lag_len, V_k)
            elif self.q_points == 50 and self.k_points == 90:
                k = self.in_nets_k_GM(K_in).view(N_k, self.num_subset, self.out_channels, T_k - self.lag_len, V_k)
            attention = torch.einsum('nsctu,nsctv->nsuv', [q, k]) / scale
            attention_list.append(attention)
        attention_stack = torch.stack(attention_list)
        attention_max, _ = torch.max(attention_stack, dim=0)
        attention_avg = torch.mean(attention_stack, dim=0)
        p_attention = (attention_avg + attention_max) / 2
        attention = self.softmax((p_attention))  
        attention = self.attdrop(attention)
        num_channels = self.num_subset * self.in_channels
        if self.q_points == 90 and self.k_points == 50:
            y = torch.einsum('nsvu,nctu->nsctv', [attention,self.V_mapping_WM(x_v)])  
            y = y.contiguous().view(N_q, num_channels, T_q, V_q)
            y = self.out_nets_GM(y) 
            y = self.relu(self.downs_GM(x_q) + y)
        elif self.q_points == 50 and self.k_points == 90:
            y = torch.einsum('nsvu,nctu->nsctv', [attention,self.V_mapping_GM(x_v)])
            y = y.contiguous().view(N_q, num_channels, T_q, V_q)
            y = self.out_nets_WM(y)  
            y = self.relu(self.downs_WM(x_q) + y)  
        return y
        
class Self_AttentionBlock(nn.Module):  
    def __init__(self, in_channels, out_channels, num_subset, lag_len, attentiondrop=0.2):
        super(Self_AttentionBlock, self).__init__()
        self.lag_len = lag_len
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.num_subset = num_subset
        self.in_nets_q = nn.Conv2d(in_channels, num_subset * out_channels, 1, bias=True)
        self.in_nets_k = nn.Conv2d(in_channels, num_subset * out_channels, 1, bias=True)
        self.V_mapping = nn.Conv2d(in_channels, out_channels, 1, bias=True)
        self.out_nets = nn.Sequential(
            nn.Conv2d(out_channels * num_subset, out_channels, 1, bias=True), 
            nn.BatchNorm2d(out_channels),
        )
        self.ff_nets = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, 1, padding=0, bias=True),  
            nn.BatchNorm2d(out_channels),
        )
        self.downs1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=True),
            nn.BatchNorm2d(out_channels),
        )
        self.downs2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=True),
            nn.BatchNorm2d(out_channels),
        )
        self.softmax = nn.Softmax(-1)
        self.relu = nn.LeakyReLU(0.1)
        self.attdrop = nn.Dropout(attentiondrop)

    def forward(self, x):
        N, C, T, V = x.size()  
        y = x.clone()
        lags = torch.arange(0, self.lag_len + 1)
        K_in_list = [y[:, :, l:(T - self.lag_len + l), :].contiguous() for l in lags]
        Q_in = K_in_list[0]
        q = self.in_nets_q(Q_in).view(N, self.num_subset, self.out_channels, T - self.lag_len, V)
        attention_list = []
        scale = (self.out_channels * (T - self.lag_len)) ** 0.5
        for K_in in K_in_list:
            k = self.in_nets_k(K_in).view(N, self.num_subset, self.out_channels, T - self.lag_len, V)
            attention = torch.einsum('nsctu,nsctv->nsuv', [q, k]) / scale
            attention_list.append(attention)
        attention_stack = torch.stack(attention_list)
        attention_max, _ = torch.max(attention_stack, dim=0)
        attention_avg = torch.mean(attention_stack, dim=0)
        p_attention = (attention_avg + attention_max) / 2
        attention = self.softmax((p_attention))  
        attention = self.attdrop(attention)
        num_channels = self.num_subset * self.in_channels
        y = torch.einsum('nctu,nsuv->nsctv', [self.V_mapping(x), attention])
        y = y.contiguous().view(N, num_channels, T, V)
        y = self.out_nets(y) 
        y = self.relu(self.downs1(x) + y)  
        y = self.ff_nets(y)
        y = self.relu(self.downs2(x) + y)

        return y