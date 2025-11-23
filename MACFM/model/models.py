import torch as th
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from .torch_gcn import GCN

class Text_1D_ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(Text_1D_ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class Text_1D_SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(Text_1D_SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = th.mean(x, dim=1, keepdim=True)
        max_out, _ = th.max(x, dim=1, keepdim=True)
        x = th.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class DCAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=3):
        super(DCAM, self).__init__()
        self.ca = Text_1D_ChannelAttention(in_planes, ratio)
        self.sa = Text_1D_SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class AMFM(nn.Module):
    def __init__(self):
        super(AMFM, self).__init__()

    def forward(self, T_bar, G_bar):
        ht = th.tanh(T_bar)
        hg = th.sigmoid(G_bar)
        h = ht * ht + hg * hg
        z = th.sigmoid(h)
        zt = z * ht
        zg = (1.0 - z) * hg
        f = zt + zg
        return f

class MACFM(th.nn.Module): # for eng_datasets,set pretrained_model='./pretrained_models/Roberta-base-eng'
    def __init__(self, pretrained_model='./pretrained_models/Roberta-mid', nb_class=20, m=0.7, gcn_layers=2,
                 n_hidden=200, dropout=0.1):
        super(MACFM, self).__init__()
        self.tan_f = nn.Tanh()
        self.sigmoid_f = nn.Sigmoid()

        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = th.nn.Linear(self.feat_dim, nb_class)
        self.gcn = GCN(
            in_feats=self.feat_dim,
            n_hidden=n_hidden,
            n_classes=nb_class,
            n_layers=gcn_layers - 1,
            activation=F.elu,
            dropout=dropout
        )

        self.dcam = DCAM(self.feat_dim)

        self.amfm = AMFM()

    def forward(self, g, idx):
        input_ids, attention_mask = g.ndata['input_ids'][idx], g.ndata['attention_mask'][idx]

        cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]

        cls_feats = cls_feats.unsqueeze(-1)
        cls_feats = self.dcam(cls_feats)
        cls_feats = cls_feats.squeeze(-1)

        if self.training:
            cls_feats = cls_feats
            g.ndata['cls_feats'][idx] = cls_feats
        else:
            cls_feats = g.ndata['cls_feats'][idx]

        cls_logit = self.classifier(cls_feats)
        gcn_logit = self.gcn(g.ndata['cls_feats'], g, g.edata['edge_weight'])[idx]

        fused_logit = self.amfm(cls_logit, gcn_logit)
        pred = F.log_softmax(fused_logit, dim=1)

        return pred

