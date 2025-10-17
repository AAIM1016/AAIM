import torch
import torch.nn as nn
from Asynchronous_Attention import Cross_AttentionBlock, Self_AttentionBlock

def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    # nn.init.constant_(conv.bias, 0)
def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)
def fc_init(fc):
    nn.init.xavier_normal_(fc.weight)
    nn.init.constant_(fc.bias, 0)

class AAIM(nn.Module):
    def __init__(self, num_point, GM_n, WM_n, config=None, attentiondrop=0.2):
        super(AAIM, self).__init__()
        self.GM_n=GM_n
        self.WM_n=WM_n
        self.num_point=num_point
        self.num_layer=len(config)
        in_channels = config[0][0]
        num_channel = 1
        self.input_map_GM = nn.Sequential(
            nn.Conv2d(num_channel, in_channels, kernel_size=(3, 1), padding=(1, 0), stride=(1, 1)),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )
        self.input_map_WM = nn.Sequential(
            nn.Conv2d(num_channel, in_channels, kernel_size=(3, 1), padding=(1, 0), stride=(1, 1)),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )
        self.config = config
        self.crossGW_layers = nn.ModuleList()
        self.crossWG_layers = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        for index, (in_channels, out_channels, num_subset, lag_len) in enumerate(self.config):
            self.crossGW_layers.append(Cross_AttentionBlock(in_channels, out_channels, num_subset, lag_len, q_points=GM_n, k_points=WM_n, attentiondrop=attentiondrop))
            self.crossWG_layers.append(Cross_AttentionBlock(in_channels, out_channels, num_subset, lag_len, q_points=WM_n, k_points=GM_n, attentiondrop=attentiondrop))
            self.attention_layers.append(Self_AttentionBlock(out_channels, out_channels, num_subset, lag_len, attentiondrop=attentiondrop))
        self.out_channels = config[-1][1] 
        self.num_point=num_point
        self.CrossFusion_net = nn.Sequential(
                  nn.Conv1d(self.out_channels*self.num_point, self.out_channels, 1),
                  nn.ReLU(),
                  nn.Conv1d(self.out_channels, 1, 1), 
                  nn.Softmax(-1)
                  )
        self.fc = nn.Sequential(
                nn.Linear(self.out_channels*self.num_point, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                #nn.Dropout(p=0.5),
                nn.Linear(64, 2),
                #nn.Sigmoid()
                )
 
    def forward(self, x):
        x = x.unsqueeze(dim=1) 
        GM_x = self.input_map_GM(x[:, :, :, :self.GM_n])
        WM_x = self.input_map_WM(x[:, :, :, self.GM_n:])
        layers_out=[]
        for i in range(self.num_layer):
            cross_module_GM = self.crossGW_layers[i]
            cross_module_WM = self.crossWG_layers[i]
            self_module=self.attention_layers[i]
            if i == 0:
                GM_CA_out = cross_module_GM(GM_x, WM_x, WM_x)
                WM_CA_out = cross_module_WM(WM_x, GM_x, GM_x)
                CA_out = torch.concat((GM_CA_out, WM_CA_out), dim=-1).contiguous()
                layers_out.append(self_module(CA_out))
            elif i > 0:
                GM_in = GM_x + layers_out[i - 1][:, :, :, :self.GM_n]
                WM_in = WM_x + layers_out[i - 1][:, :, :, self.GM_n:]
                GM_CA_out = cross_module_GM(GM_in, WM_in, WM_in)
                WM_CA_out = cross_module_WM(WM_in, GM_in, GM_in)
                CA_out = torch.concat((GM_CA_out, WM_CA_out), dim=-1).contiguous()
                layers_out.append(self_module(CA_out))
        layers_out = [torch.transpose(h.mean(2),2,1).contiguous().view(h.shape[0],self.out_channels*self.num_point,1) for h in layers_out]
        layers_out=torch.concat(layers_out,dim=2)
        cross_att = self.CrossFusion_net(layers_out)
        fusion_out = (layers_out * cross_att).sum(dim=-1)
        gm_out = fusion_out[:,:self.out_channels*self.GM_n]
        wm_out = fusion_out[:,self.out_channels*self.GM_n:]
        return gm_out, wm_out, self.fc(fusion_out)

if __name__ == '__main__':
    import argparse 
    import ast
    parser = argparse.ArgumentParser()  
    parser.add_argument('--GM_n', type=int, default=90, help=' ')
    parser.add_argument('--WM_n', type=int, default=50, help=' ')
    parser.add_argument('--config', type=ast.literal_eval, default='[[8, 8, 2, 6],[8, 8, 2, 6],[8, 8, 2, 6],]' )
    args = parser.parse_args()

    model = AAIM(num_point=args.GM_n+args.WM_n, GM_n=args.GM_n,WM_n=args.WM_n, config=args.config)
    data=torch.zeros(16, 130, 140)
    y=model(data)
    
    