import torch
import torch.nn as nn
import torch.nn.functional as F
import math, random
from utils import activation_method
from collections import defaultdict


def create_module(module_type, **config):
    module_type = module_type.lower()
    if module_type == 'mlp':
        return MLP(**config)
    elif module_type == 'gcn':
        return AttentionGCN(**config)
    elif module_type == 'empty':
        return nn.Sequential()
    else:
        raise NotImplementedError


class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers, final_size=0, final_activation="none", normalization="batch_norm",
                 activation='relu'):
        """
        :param input_size:
        :param hidden_layers: [(unit_num, normalization, dropout_rate)]
        :param final_size:
        :param final_activation:
        """
        nn.Module.__init__(self)
        self.input_size = input_size
        fcs = []
        last_size = self.input_size
        for size, to_norm, dropout_rate in hidden_layers:
            linear = nn.Linear(last_size, size)
            linear.bias.data.fill_(0.0)
            fcs.append(linear)
            last_size = size
            if to_norm:
                if normalization == 'batch_norm':
                    fcs.append(nn.BatchNorm1d(last_size))
                elif normalization == 'layer_norm':
                    fcs.append(nn.LayerNorm(last_size))
            fcs.append(activation_method(activation))
            if dropout_rate > 0.0:
                fcs.append(nn.Dropout(dropout_rate))
        self.fc = nn.Sequential(*fcs)
        if final_size > 0:
            linear = nn.Linear(last_size, final_size)
            linear.bias.data.fill_(0.0)
            finals = [linear, activation_method(final_activation)]
        else:
            finals = []
        self.final_layer = nn.Sequential(*finals)

    def forward(self, x):
        out = self.fc(x)
        out = self.final_layer(out)
        return out


class MultiheadAttention(nn.Module):
    def __init__(self, input_size, query_size, value_size, head_num, dropout=0.0, concatenate=True, configurable=False,
                 use_dot=True):
        nn.Module.__init__(self)
        self.use_dot = use_dot
        if use_dot is True:
            self.query_heads = nn.Linear(input_size, head_num * query_size, bias=True)
        else:
            self.query_heads = nn.Linear(query_size + input_size, head_num, bias=False)
        self.head_num = head_num
        self.concatenate = concatenate
        self.input_size = input_size
        self.value_size = value_size
        if concatenate:
            self.value_proj = nn.Linear(value_size, input_size)
        else:
            self.value_proj = nn.Linear(value_size, input_size * head_num)
        if configurable:
            self.param_divide(self.query_heads, with_query=True)
            self.param_divide(self.value_proj, with_query=True)
        if dropout > 0.0:
            self.attn_dropout = nn.Dropout(dropout)
        else:
            self.attn_dropout = None
        self.attn = None

    @staticmethod
    def param_divide(linear_module, with_query):
        weight = getattr(linear_module, 'weight')
        del linear_module._parameters['weight']
        linear_module.register_parameter('share_weight', weight)
        setattr(linear_module, 'weight', weight.data)
        if with_query:
            input_size, output_size = linear_module.in_features, linear_module.out_features
            bound = math.sqrt(6.0 / input_size)
            query_vector = torch.empty(output_size, dtype=torch.float)
            nn.init.uniform_(query_vector, -bound, bound)
            linear_module.register_parameter('query', nn.Parameter(query_vector))

    def configure(self, in_vector):
        """
        :param in_vector: (2, in_features)
        :return:
        """
        setattr(self.query_heads, 'weight',
                torch.matmul(self.query_heads.query.unsqueeze(-1), in_vector[0:1]) + self.query_heads.share_weight)
        setattr(self.value_proj, 'weight',
                torch.matmul(self.value_proj.query.unsqueeze(-1), in_vector[1:2]) + self.value_proj.share_weight)

    @staticmethod
    def attention(scores, value, mask=None, dropout=None):
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        p_attn_org = p_attn
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn_org

    def forward(self, query, key, value, mask=None):
        """
        :param query: (batch_size, input_size)
        :param key: (batch_size, max_len, query_size)
        :param value: (batch_size, max_len, value_size)
        :return:
        """
        batch_size, max_len = key.size(0), key.size(1)
        value_size = self.value_proj.out_features // self.head_num
        value = self.value_proj(value)
        # batch_size, attnhead_num, max_len, out_features
        value = value.view(batch_size, max_len, self.head_num, value_size).transpose(1, 2)
        # (*, output_features) (*, max_len, out_features)
        if self.use_dot:
            attnhead_size = self.query_heads.out_features // self.head_num
            query = self.query_heads(query)
            query = query.view(batch_size, self.head_num, 1, attnhead_size)
            # batch_size attnhead_num, max_len, query_size
            key = key.unsqueeze(1).expand(-1, self.head_num, -1, -1)
            # batch_size, query_num, 1, dict_size
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(attnhead_size)
        else:
            # batch_size, max_len, query_size + input_size
            query = torch.cat((query.unsqueeze(1).expand(-1, max_len, -1), key), dim=-1)
            # batch_size, head_size, 1, max_len
            scores = self.query_heads(query).transpose(-1, -2).unsqueeze(-2)
        attn_value, attn = self.attention(scores, value, mask=mask, dropout=self.attn_dropout)
        self.attn = attn.detach()
        if self.concatenate:
            attn_value = attn_value.view(batch_size, -1)
        else:
            attn_value = attn_value.mean(dim=1).squeeze(1)
        return attn_value


class AttentionGCN(nn.Module):
    def __init__(self, input_size, layer_num, attnhead_num, attention=True, activation='relu', layer_norm=False,
                 attn_dropout=0.0, configurable=False, use_dot=True):
        nn.Module.__init__(self)
        self.layer_num = layer_num
        self.input_size = input_size
        self.attnhead_num = attnhead_num
        self.layers = []
        attn_heads, node_projs, activations, normalizations = [], [], [], []
        self.d_k = input_size // attnhead_num
        self.use_attention = attention
        for i in range(layer_num):
            if attention:
                attn_head = MultiheadAttention(input_size=input_size, query_size=input_size, value_size=input_size,
                                               head_num=attnhead_num, use_dot=use_dot,
                                               configurable=configurable, dropout=attn_dropout,
                                               concatenate=i != layer_num - 1)
            else:
                attn_head = nn.Linear(input_size, input_size)
            attn_heads.append(attn_head)
            node_projs.append(nn.Linear(input_size, input_size, bias=True))
            if i != layer_num - 1:
                activations.append(activation_method(activation))
            else:
                activations.append(activation_method('none'))
            if layer_norm and i != layer_num - 1:
                normalizations.append(nn.LayerNorm(input_size))
            else:
                normalizations.append(nn.Sequential())
        if len(attn_heads) > 0:
            self.attn_heads = nn.ModuleList(attn_heads)
        self.node_projs = nn.ModuleList(node_projs)
        self.activations = nn.ModuleList(activations)
        self.normalizations = nn.ModuleList(normalizations)
        self.attns = []
        if attn_dropout > 0.0:
            self.atten_dropout = nn.Dropout(attn_dropout)
        else:
            self.atten_dropout = None

    @staticmethod
    def generate_mask(lengths, max_len):
        batch_size = lengths.size(0)
        masks = torch.arange(0, max_len, device=lengths.device).unsqueeze(0).expand(batch_size, -1)
        masks = masks < lengths.unsqueeze(1)
        return masks

    def forward(self, node_embeds, neighbor_embeds, node_degrees=None):
        """
        :param node_embeds: (batch_size, embed_dim)
        :param neighbor_embeds: (batch_size, max_len, embed_dim)
        :param node_degrees: (batch_size,)
        :return:
        """
        batch_size, max_len, embed_dim = neighbor_embeds.size()
        if node_degrees is not None:
            neighbor_mask = self.generate_mask(node_degrees, max_len).view(batch_size, 1, 1, max_len)
            neighbor_embeds = neighbor_embeds.clone().masked_fill_(neighbor_mask.view(batch_size, max_len, 1) == 0, 0.0)
        else:
            neighbor_mask = None
        if not self.use_attention:
            if node_degrees is None:
                neighbor_embeds = neighbor_embeds.mean(dim=1)
            else:
                neighbor_embeds = neighbor_embeds.sum(dim=1)
                nonzeros = node_degrees.nonzero().squeeze(-1)
                neighbor_embeds[nonzeros] /= node_degrees[nonzeros].unsqueeze(-1).type(dtype=torch.float)
        for i in range(self.layer_num):
            node_proj, activation, normalization = self.node_projs[i], \
                                                   self.activations[i], self.normalizations[i]
            attn_head = self.attn_heads[i]
            if self.use_attention:
                attn_value = attn_head(node_embeds, neighbor_embeds, neighbor_embeds, mask=neighbor_mask)
            else:
                attn_value = attn_head(neighbor_embeds)
            node_embeds = node_proj(node_embeds) + attn_value
            node_embeds = normalization(node_embeds)
            node_embeds = activation(node_embeds)
        return node_embeds


class Recommender(nn.Module):
    def __init__(self, useritem_embeds, user_graph=False, item_graph=False):
        nn.Module.__init__(self)
        self.useritem_embeds = useritem_embeds
        self.user_graph = user_graph
        self.item_graph = item_graph

    def forward(self, query_users, query_items, with_attr=False):
        if query_users[0].dim() > 1:
            query_users = list(map(lambda x: x.squeeze(0), query_users))
        if query_items[0].dim() > 1:
            query_items = list(map(lambda x: x.squeeze(0), query_items))
        if not with_attr:
            query_users = self.useritem_embeds(*query_users, is_user=True, with_neighbor=self.user_graph)
            query_items = self.useritem_embeds(*query_items, is_user=False, with_neighbor=self.item_graph)
        return query_users, query_items


class InteractionRecommender(Recommender):
    def __init__(self, useritem_embeds, mlp_config):
        super(InteractionRecommender, self).__init__(useritem_embeds)
        self.mlp = MLP(**mlp_config)

    def forward(self, query_users, query_items, support_users=None, support_items=None, with_attr=False):
        query_users, query_items = super(InteractionRecommender, self).forward(query_users, query_items,
                                                                               with_attr=with_attr)
        query_users, query_items = query_users[0], query_items[0]
        if query_users.size(0) == 1:
            query_users = query_users.expand(query_items.size(0), -1)
        query_embeds = torch.cat((query_users, query_items), dim=1)
        return self.mlp(query_embeds).squeeze(1)


class EmbedRecommender(Recommender):
    def __init__(self, useritem_embeds, user_config, item_config, user_graph=True, item_graph=True):
        super(EmbedRecommender, self).__init__(useritem_embeds, user_graph, item_graph)
        self.user_model = create_module(**user_config)
        self.item_model = create_module(**item_config)

    def forward(self, query_users, query_items, with_attr=False):
        """
        :param with_attr:
        :param query_users: (batch_size,)
        :param query_items: (batch_size)
        :return:
        """
        query_users, query_items = Recommender.forward(self, query_users, query_items, with_attr=with_attr)
        query_users = self.user_model(*query_users)
        query_items = self.item_model(*query_items)
        return (query_users * query_items).sum(dim=1)


class CoNet(nn.Module):
    def __init__(self, useritem_embeds, source_ratings, item_padding_idx, input_size, hidden_layers):
        nn.Module.__init__(self)
        self.useritem_embeds = useritem_embeds
        self.source_ratings = source_ratings
        self.item_padding_idx = item_padding_idx
        last_size = input_size * 2
        layers1, layers2, transfer_layers = [], [], []
        for hidden_size in hidden_layers:
            layers1.append(nn.Linear(last_size, hidden_size))
            layers2.append(nn.Linear(last_size, hidden_size))
            transfer_layers.append(nn.Linear(last_size, hidden_size))
            last_size = hidden_size
        self.target_layers = nn.ModuleList(layers1)
        self.auxiliary_layers = nn.ModuleList(layers2)
        self.transfer_layers = nn.ModuleList(transfer_layers)
        self.target_output = nn.Linear(last_size, 1)
        self.auxiliary_output = nn.Linear(last_size, 1)

    def forward(self, query_users, target_items, auxiliary_items=None):
        only_target = False
        if auxiliary_items is None:
            only_target = True
            auxiliary_items = [
                random.choice(self.source_ratings[user_id.item()]) if len(
                    self.source_ratings[user_id.item()]) > 0 else self.item_padding_idx for user_id in query_users[0]]
            auxiliary_items = (torch.tensor(auxiliary_items, dtype=torch.long, device=query_users[0].device),)
        query_users = list(map(lambda x: x.expand(target_items[0].size(0)), query_users))
        auxiliary_items = list(map(lambda x: x.expand(target_items[0].size(0)), auxiliary_items))
        query_users = self.useritem_embeds(*query_users, is_user=True)
        target_items, auxiliary_items = self.useritem_embeds(*target_items, is_user=False), self.useritem_embeds(
            *auxiliary_items, is_user=False)
        target_x = torch.cat((*query_users, *target_items), dim=1)
        auxiliary_x = torch.cat((*query_users, *auxiliary_items), dim=1)
        for target_layer, auxiliary_layer, transfer_layer in zip(self.target_layers, self.auxiliary_layers,
                                                                 self.transfer_layers):
            new_target_x = target_layer(target_x) + transfer_layer(auxiliary_x)
            new_auxiliary_x = auxiliary_layer(auxiliary_x) + transfer_layer(target_x)
            target_x, auxiliary_x = new_target_x, new_auxiliary_x
            target_x, auxiliary_x = torch.relu_(target_x), torch.relu_(auxiliary_x)
        if only_target:
            return self.target_output(target_x).squeeze(-1)
        else:
            return self.target_output(target_x).squeeze(-1), self.auxiliary_output(auxiliary_x).squeeze(-1)


class HybridRecommender(Recommender):
    def __init__(self, useritem_embeds, input_size, hidden_layers, final_size, activation='relu',
                 normalization="batch_norm"):
        super(HybridRecommender, self).__init__(useritem_embeds, False, False)
        self.interaction_model = MLP(input_size=2 * input_size, hidden_layers=hidden_layers, activation=activation,
                                     normalization=normalization, final_activation='none', final_size=final_size)
        self.final_layer = nn.Linear(input_size + final_size, 1)

    def forward(self, query_users, query_items, with_attr=False):
        query_users, query_items = Recommender.forward(self, query_users, query_items, with_attr=with_attr)
        query_users, query_items = query_users[0], query_items[0]
        if query_users.size(0) == 1:
            query_users = query_users.expand(query_items.size(0), -1)
        interactions = torch.cat((query_users, query_items), dim=-1)
        interactions = self.interaction_model(interactions)
        product = query_users * query_items
        concatenation = torch.cat((interactions, product), dim=-1)
        return self.final_layer(concatenation).squeeze(-1)
