
from __future__ import absolute_import, division, print_function
import os, math, logging, numpy as np
from os.path import join
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from lib.models.DCNv2.dcn_v2 import DCN
from .graph.minigraph import GraphBuilder, TemporalGraphBuilder
from .graph.gnn_stub import IdentityGNN
from .graph.gnn_gsl import LearnableTopoGNN
from .graph.fusion import Graph2MapFusion
from .graph.prior import GridPriorBuilder
from .graph.association import EdgeScorer
from .props import build_proposals_lastframe

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

def get_model_url(data='imagenet', name='dla34', hash='ba72cf86'):
    return join('http://dl.yf.io/dla/models', data, '{}-{}.pth'.format(name, hash))

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=dilation, bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=dilation, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.stride = stride
    def forward(self, x, residual=None):
        if residual is None: residual = x
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out)
        out += residual; out = self.relu(out); return out

class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual
    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1)); x = self.bn(x)
        if self.residual: x += children[0]
        x = self.relu(x); return x

class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1, level_root=False, root_dim=0, root_kernel_size=1, dilation=1, root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0: root_dim = 2 * out_channels
        if level_root: root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride, dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1, dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels, stride, root_dim=0, root_kernel_size=root_kernel_size, dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels, root_dim=root_dim + out_channels, root_kernel_size=root_kernel_size, dilation=dilation, root_residual=root_residual)
        if levels == 1: self.root = Root(root_dim, out_channels, root_kernel_size, root_residual)
        self.level_root = level_root; self.root_dim = root_dim
        self.downsample = None; self.project = None; self.levels = levels
        if stride > 1: self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                                         nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM))
    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root: children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1); x = self.root(x2, x1, *children)
        else:
            children.append(x1); x = self.tree2(x1, children=children)
        return x

class DLA(nn.Module):
    def __init__(self, levels, channels, num_classes=1000, block=BasicBlock, residual_root=False, linear_root=False):
        super(DLA, self).__init__()
        self.channels = channels
        self.base_layer = nn.Sequential(nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3, bias=False),
                                        nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM), nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2, level_root=False, root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2, level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2, level_root=True, root_residual=residual_root)
    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride if i == 0 else 1, padding=dilation, bias=False, dilation=dilation),
                            nn.BatchNorm2d(planes, momentum=BN_MOMENTUM), nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)
    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(5):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        return y
    def load_pretrained_model(self, data='imagenet', name='dla34', hash='ba72cf86'):
        if name.endswith('.pth'):
            model_weights = torch.load(data + name)
        else:
            model_url = get_model_url(data, name, hash)
            model_weights = model_zoo.load_url(model_url)
        num_classes = len(model_weights[list(model_weights.keys())[-1]])
        self.fc = nn.Conv2d(self.channels[-1], num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.load_state_dict(model_weights, strict=False)

def dla34(pretrained=True, **kwargs):
    model = DLA([1, 1, 1, 2, 2, 1], [16, 32, 64, 128, 256, 512], block=BasicBlock, **kwargs)
    if pretrained: model.load_pretrained_model(data='imagenet', name='dla34', hash='ba72cf86')
    return model

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None: nn.init.constant_(m.bias, 0)

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2); c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]

class DeformConv(nn.Module):
    def __init__(self, chi, cho):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(nn.BatchNorm2d(cho, momentum=BN_MOMENTUM), nn.ReLU(inplace=True))
        self.conv = DCN(chi, cho, kernel_size=(3,3), stride=1, padding=1, dilation=1, deformable_groups=1)
    def forward(self, x): x = self.conv(x); x = self.actf(x); return x

class IDAUp(nn.Module):
    def __init__(self, o, channels, up_f):
        super(IDAUp, self).__init__()
        for i in range(1, len(channels)):
            c = channels[i]; f = int(up_f[i])
            proj = DeformConv(c, o); node = DeformConv(o, o)
            up = nn.ConvTranspose2d(o, o, f * 2, stride=f, padding=f // 2, output_padding=0, groups=o, bias=False)
            fill_up_weights(up)
            setattr(self, 'proj_' + str(i), proj); setattr(self, 'up_' + str(i), up); setattr(self, 'node_' + str(i), node)
    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])

class DLAUp(nn.Module):
    def __init__(self, startp, channels, scales, in_channels=None):
        super(DLAUp, self).__init__()
        self.startp = startp
        if in_channels is None: in_channels = channels
        self.channels = channels
        channels = list(channels); scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i), IDAUp(channels[j], in_channels[j:], scales[j:] // scales[j]))
            scales[j + 1:] = scales[j]; in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]
    def forward(self, layers):
        out = [layers[-1]]
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) - i - 2, len(layers))
            out.insert(0, layers[-1])
        return out

class THGSL(nn.Module):
    def __init__(self, heads, final_kernel, head_conv, out_channel=0):
        super(THGSL, self).__init__()

        self._use_gnn_fusion = (os.environ.get('USE_GNN_FUSION', '1') == '1')
        self._g2m_fuser = Graph2MapFusion(in_channels=16, out_channels=16)
        self.g2m_gate = nn.Parameter(torch.full((16,), -1.0))
        self._prior_builder = GridPriorBuilder()
        self._use_real_gnn = (os.environ.get('USE_REAL_GNN', '1') == '1')
        self._real_gnn = LearnableTopoGNN(in_dim=16, hidden_dim=64, out_dim=16, dropout=0.0)
        self._edge_scorer = EdgeScorer(in_dim=16, hidden=128, geom_dim=7, dropout=0.1)
        self._mini_graph_builder = GraphBuilder(knn_k=8, use_knn=True)
        self._temporal_graph_builder = TemporalGraphBuilder(
            knn_intra=int(os.environ.get('TEMP_KNN_INTRA', '8')),
            knn_inter=int(os.environ.get('TEMP_KNN_INTER', '4')),
            use_knn=True,
            align=(os.environ.get('TEMP_ALIGN','1')=='1'),
            align_ema=float(os.environ.get('TEMP_ALIGN_EMA','0.0')),
            align_max_shift=int(os.environ.get('TEMP_ALIGN_MAX','32')),
            align_topk=int(os.environ.get('TEMP_ALIGN_TOPK','30'))
        )
        self._temp_T = int(os.environ.get('TEMP_GNN_T', '5'))
        self._temp_topk = int(os.environ.get('TOPK_TEMP', '50'))

        self.first_level = 0
        self.last_level = 3
        self.backbone = dla34(pretrained=True)
        channels = [16, 32, 64, 128, 256]
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales)
        if out_channel == 0: out_channel = channels[self.first_level]
        self.ida_up = IDAUp(out_channel, channels[self.first_level:self.last_level], [2 ** i for i in range(self.last_level - self.first_level)])

        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(channels[self.first_level], head_conv, kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, classes, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True))
                if 'hm' in head: fc[-1].bias.data.fill_(-4.6)
                else: fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def _extract_p0(self, x):
        c0, c1, c2, c3, c4 = self.backbone(x)
        p0, p1, p2, p3, _ = self.dla_up([c0, c1, c2, c3, c4])
        y = [p0, p1, p2]
        self.ida_up(y, 0, len(y))
        p0 = y[-1]
        return p0, (p1, p2, p3)

    def _build_temporal_graph(self, p0_list, hm_head, wh_head, reg_head, topk, T):
        props_list = []
        for p0 in p0_list[-T:]:
            with torch.no_grad():
                hm_i = hm_head(p0)
                wh_i = wh_head(p0)
                reg_i = reg_head(p0)
            props_i = build_proposals_lastframe(p0, hm_i, wh_i, reg_i, topk=topk)
            try:
                prior = self._prior_builder.build_from_p0(p0, hm=hm_i)
                if prior and prior['feats'].shape[1] > 0:
                    props_i['feats']   = torch.cat([props_i['feats'],   prior['feats']],   dim=1)
                    props_i['centers'] = torch.cat([props_i['centers'], prior['centers']], dim=1)
                    props_i['types']   = torch.cat([torch.zeros((p0.shape[0], props_i['feats'].shape[1]-prior['feats'].shape[1]), device=p0.device, dtype=torch.long),
                                                    torch.ones((p0.shape[0], prior['feats'].shape[1]), device=p0.device, dtype=torch.long)], dim=1)
                else:
                    props_i['types'] = torch.zeros((p0.shape[0], props_i['feats'].shape[1]), device=p0.device, dtype=torch.long)
            except Exception:
                props_i['types'] = torch.zeros((p0.shape[0], props_i['feats'].shape[1]), device=p0.device, dtype=torch.long)
            props_list.append(props_i)
        graph_pack = self._temporal_graph_builder.build_seq(props_list)
        return graph_pack, props_list[-1]['hw']

    def forward(self, img_input, training=True, vid=None):
        B, N, C, H, W = img_input.shape
        p0_list = []
        temp_feat = []

        for i in range(N):
            x = img_input[:, i, :]
            p0, others = self._extract_p0(x)
            p0_list.append(p0)
            temp_feat.append([p0, *others])

        ret_temp = {}
        gnn_spatial_map = None
        try:
            if self._use_gnn_fusion:
                graph_pack, hw = self._build_temporal_graph(p0_list, self.hm, self.wh, self.reg, topk=self._temp_topk, T=max(1, self._temp_T))
                nf = graph_pack['node_feat'].to(p0_list[-1].device).float()
                ed = graph_pack['edges'].to(p0_list[-1].device)
                
                xy = graph_pack['node_xy'].to(p0_list[-1].device).float()
                node_types = graph_pack.get('node_types', None)
                if node_types is not None:
                    node_types = node_types.to(p0_list[-1].device)
                assoc_logits = self._edge_scorer(nf, ed, node_xy=xy, node_types=node_types)
                node_feat = self._real_gnn(nf, ed, node_xy=xy, edge_logits=assoc_logits, node_time=graph_pack['node_time'].to(p0_list[-1].device)) if self._use_real_gnn else nf
                gnn_spatial_map = self._g2m_fuser(node_feat, xy, hw, node_types=node_types).contiguous()
                lam = float(os.environ.get('G2M_L2', '0.0'))
                if lam > 0.0:
                    gate = torch.sigmoid(self.g2m_gate).view(1, -1, 1, 1).to(gnn_spatial_map.device)
                    ret_temp['_aux_g2m_l2'] = lam * (gate * gnn_spatial_map).pow(2).mean()
                ret_temp['assoc'] = {
                    'logits': assoc_logits,
                    'edges':  ed,
                    'node_xy': xy,
                    'node_types': node_types,
                    'node_time': graph_pack['node_time'].to(p0_list[-1].device),
                    't_slices': graph_pack['t_slices']
                }
        except Exception as e:
            print('[G2M] WARN:', str(e))

        gate = torch.sigmoid(self.g2m_gate).view(1, -1, 1, 1).to(p0_list[-1].device)
        p0_last = p0_list[-1] if gnn_spatial_map is None else (p0_list[-1] + gate * gnn_spatial_map)
        if hasattr(self, 'hm'):  ret_temp['hm']  = self.hm(p0_last)
        if hasattr(self, 'wh'):  ret_temp['wh']  = self.wh(p0_last)
        if hasattr(self, 'reg'): ret_temp['reg'] = self.reg(p0_last)

        if 'dis' in self.heads:
            dis_seq = []
            for i in range(N - 1):
                p0_i = p0_list[i]
        ret = {1: ret_temp}
        if ('dis' in self.heads) and ('dis' not in ret_temp):
            B, N, C, H, W = img_input.shape
            ret_temp['dis'] = p0_list[-1].new_zeros(B, max(N-1, 0), self.heads['dis'], H, W)
        return [temp_feat, ret]

def dla_dcn_net(heads, head_conv=128):
    model = THGSL(heads, final_kernel=1, head_conv=head_conv)
    return model
