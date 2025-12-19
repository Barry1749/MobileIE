import torch.nn as nn
import torch
from .utils import (
    MBRConv5,
    MBRConv3,
    MBRConv1,
    DropBlock,
    FST,
    FSTS,
)
from .final_amplify import (
    LocV3Module,
    LocV3ParamHead,
)

class MobileIELLENet(nn.Module):
    def __init__(self, channels, rep_scale=4):
        super(MobileIELLENet, self).__init__()
        self.channels = channels
        self.head = FST(
            nn.Sequential(
                MBRConv5(3, channels, rep_scale=rep_scale),
                nn.PReLU(channels),
                MBRConv3(channels, channels, rep_scale=rep_scale)
            ),
            channels
        )
        # self.att = nn.Sequential( 
        #     nn.AdaptiveAvgPool2d(1),
        #     MBRConv1(channels, channels, rep_scale=rep_scale),
        #     nn.Sigmoid()
        # )
        # self.att1= nn.Sequential( 
        #     MBRConv1(1, channels, rep_scale=rep_scale),
        #     nn.Sigmoid()
        # )
        self.body = FST(
            MBRConv3(channels, channels, rep_scale=rep_scale),
            channels
        )
        self.att = nn.Sequential( 
            nn.AdaptiveAvgPool2d(1),
            MBRConv1(channels, channels, rep_scale=rep_scale),
            nn.Sigmoid()
        )
        self.att1= nn.Sequential( 
            MBRConv1(1, channels, rep_scale=rep_scale),
            nn.Sigmoid()
        )
        self.tail = MBRConv3(channels, 3, rep_scale=rep_scale)
        self.tail_warm = MBRConv3(channels, 3, rep_scale=rep_scale)
        self.drop = DropBlock(3)

        self.locv3_param_head = LocV3ParamHead(in_channels=6, hidden=128)
        self.locv3_module = LocV3Module()
        
    def forward(self, x):
        x0 = self.head(x)
        x1 = self.body(x0)      
        x2 = self.att(x1)
        max_out, _ = torch.max(x2 * x1 , dim=1, keepdim=True)   
        x3 = self.att1(max_out)
        x4 = torch.mul(x2, x3) * x1

        x5 = self.tail(x4)
        locv3_params = self.locv3_param_head(x, x5)
        
        return self.locv3_module(
            original_rgb = x,
            base_rgb = x5,
            boost_strength              = locv3_params["boost_strength"],
            dark_thr                    = locv3_params["dark_thr"],
            low                         = locv3_params["low"],
            high                        = locv3_params["high"],
            restore_local_contrast_amount = locv3_params["restore_local_contrast_amount"],
            restore_local_contrast_sigma  = locv3_params["restore_local_contrast_sigma"],
            protect_highlights_strength   = locv3_params["protect_highlights_strength"],
            gray_world_balance_mix        = locv3_params["gray_world_balance_mix"],
        )

    def forward_warm(self, x):
        x = self.drop(x)
        x = self.head(x)
        x = self.body(x)
        return self.tail(x), self.tail_warm(x)

    def slim(self):
        net_slim = MobileIELLENetS(self.channels)
        weight_slim = net_slim.state_dict()
        for name, mod in self.named_modules():
            if isinstance(mod, MBRConv3) or isinstance(mod, MBRConv5) or isinstance(mod, MBRConv1):
               if '%s.weight' % name in weight_slim:
                    w, b = mod.slim()
                    weight_slim['%s.weight' % name] = w 
                    weight_slim['%s.bias' % name] = b 
            elif isinstance(mod, FST):
                weight_slim['%s.bias' % name] = mod.bias
                weight_slim['%s.weight1' % name] = mod.weight1
                weight_slim['%s.weight2' % name] = mod.weight2
            elif isinstance(mod, nn.PReLU):
                weight_slim['%s.weight' % name] = mod.weight
        net_slim.load_state_dict(weight_slim)
        return net_slim

class MobileIELLENetS(nn.Module):
    def __init__(self, channels):
        super(MobileIELLENetS, self).__init__()
        self.head = FSTS(
            nn.Sequential(
                nn.Conv2d(3, channels, 5, 1, 2),
                nn.PReLU(channels),
                nn.Conv2d(channels, channels, 3, 1, 1)
            ),
            channels
        )
        self.body = FSTS(
            nn.Conv2d(channels, channels, 3, 1, 1),
            channels
        )
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )
        self.att1 = nn.Sequential( 
            nn.Conv2d(1, channels, 1, 1),
            nn.Sigmoid()
        )
        self.tail = nn.Conv2d(channels, 3, 3, 1, 1)
        
    def forward(self, x):
        x0 = self.head(x)
        x1 = self.body(x0)
        x2 = self.att(x1)
        max_out, _ = torch.max(x2 * x1, dim=1, keepdim=True)   
        x3 = self.att1(max_out)
        x4 = torch.mul(x3, x2) * x1
        return self.tail(x4)
    
    
