import torch
import torch.nn as nn
import torch.nn.functional as F
import random, math, itertools
from modules import MLP
from collections import defaultdict


class AdversarialSampler(nn.Module):
    def __init__(self, loss_function, step=10, lr=0.01, gamma=0.01, normalization=False):
        nn.Module.__init__(self)
        self.step = step
        self.lr = lr
        self.gamma = gamma
        self.loss_function = loss_function
        self.normalization = normalization
        self.register_buffer('loss', torch.zeros(step))
        self.register_buffer('penalty', torch.zeros(step))
        self.detach = lambda x: x.detach_().requires_grad_(False)

    @staticmethod
    def penal(attributes, old_attributes, length=None):
        penalty = ((attributes[0] - old_attributes[0]) ** 2).mean()
        if len(attributes) > 1:
            penalty += (((attributes[1] - old_attributes[1]) ** 2).sum(dim=1) / attributes[2].view(-1, 1).type(
                torch.float)).mean()
        return penalty

    def forward(self, model, query_users, positive_items, negative_items):
        if len(query_users) > 1:
            user_embeds, user_neighbors, user_degrees = query_users
            new_user_embeds = user_embeds.clone().requires_grad_(True)
            new_user_neighbors = user_neighbors.clone().requires_grad_(True)
            new_users = (new_user_embeds, new_user_neighbors, user_degrees)
        else:
            new_users = (query_users[0].clone().requires_grad_(True),)
        new_positive_items = (positive_items[0].clone().requires_grad_(True),)
        new_negative_items = (negative_items[0].clone().requires_grad_(True),)
        self.loss.zero_()
        self.penalty.zero_()
        for step in range(self.step):
            positive_values = model(new_users, new_positive_items, with_attr=True)
            negative_values = model(new_users, new_negative_items, with_attr=True)
            loss = self.loss_function(positive_values, negative_values)
            self.loss[step] += loss.detach()
            if self.gamma > 0.0:
                penalty = self.penal(new_users, query_users)
                penalty += self.penal(new_positive_items, positive_items)
                penalty += self.penal(new_negative_items, negative_items)
                self.penalty[step] += penalty.detach()
                loss -= self.gamma * penalty
            grads = torch.autograd.grad(loss, itertools.chain(new_users[:2], new_positive_items, new_negative_items))
            with torch.no_grad():
                for grad, attr in zip(grads, itertools.chain(new_users[:2], new_positive_items, new_negative_items)):
                    if self.normalization:
                        attr += self.lr * grad / grad.norm()
                    else:
                        attr += self.lr * grad
        return list(map(self.detach, new_users)), list(map(self.detach, new_positive_items)), list(
            map(self.detach, new_negative_items))


class GradientModel(nn.Module):
    class StopControl(nn.Module):
        def __init__(self, input_size, hidden_size):
            nn.Module.__init__(self)
            self.lstm = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)
            self.output_layer = nn.Linear(hidden_size, 1)
            self.output_layer.bias.data.fill_(0.0)
            self.h_0 = nn.Parameter(torch.randn((hidden_size,), requires_grad=True))
            self.c_0 = nn.Parameter(torch.randn((hidden_size,), requires_grad=True))

        def forward(self, inputs, hx):
            if hx is None:
                hx = (self.h_0.unsqueeze(0), self.c_0.unsqueeze(0))
            h, c = self.lstm(inputs, hx)
            return torch.sigmoid(self.output_layer(h).squeeze()), (h, c)

    def __init__(self, user_neighbors, item_neighbors, useritem_embeds, model, loss_function, attack_model=None,
                 step=10, min_step=None, flexible_step=False, hidden_input=True, addition_params=None,
                 batch_size=64, learn_bias=False,
                 user_graph=True, item_graph=False):
        nn.Module.__init__(self)
        self.flexible_step = flexible_step
        if min_step is None:
            min_step = step
        self.min_step = min_step
        if addition_params is None:
            addition_params = []
        self.user_neighbors = user_neighbors
        self.item_neighbors = item_neighbors
        self.useritem_embeds = useritem_embeds
        self.model = model
        self.learned_params = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                self.learned_params.append((name, 'weight'))
                self.init_learned_param(module, 'weight')
                if learn_bias and module.bias is not None:
                    self.learned_params.append((name, 'bias'))
                    self.init_learned_param(module, 'bias')
        named_modules = dict(self.model.named_modules())
        for layer_name, param_name in addition_params:
            if not (layer_name, param_name) in self.learned_params:
                self.learned_params.append((layer_name, param_name))
                module = named_modules[layer_name]
                self.init_learned_param(module, param_name)
        self.loss_function = loss_function
        # gradient, parameter, last state, loss
        self.max_step = step
        self.batch_size = batch_size
        self.user_graph = user_graph
        self.item_graph = item_graph
        self.attack_model = attack_model
        self.aim_parameters = []
        if self.flexible_step:
            self.hidden_input = hidden_input
            if hidden_input:
                stop_input_size = len(self.learned_params) * lstm_config['hidden_size']
            else:
                stop_input_size = len(self.learned_params) + 1
            hidden_size = stop_input_size * 2
            self.stop_gate = self.StopControl(stop_input_size, hidden_size)

    @staticmethod
    def init_learned_param(module, param_name):
        param = getattr(module, param_name)
        del module._parameters[param_name]
        module.register_parameter('share_' + param_name, param)
        setattr(module, param_name, param.data.clone())

    def init(self):
        self.current_step = 0
        self.store = {
            'loss': [],
            'grads': [],
            'stop_gates': []
        }
        self.aim_parameters = []
        named_modules = dict(self.model.named_modules())
        for layer_name, param_name in self.learned_params:
            layer = named_modules[layer_name]
            setattr(layer, param_name, getattr(layer, 'share_' + param_name).clone().requires_grad_(True))
            self.aim_parameters.append(getattr(layer, param_name))
        if self.flexible_step:
            self.stop_hx = None

    def update(self, parameters, loss, grads):
        raise NotImplementedError

    def stop(self, step, loss, grads):
        if self.flexible_step:
            if step < self.max_step:
                if self.hidden_input:
                    hxs = list(map(lambda x: x[0].mean(dim=0), self.hxs))
                    # 1, param_num * hidden_size
                    inputs = torch.cat(hxs, dim=0).unsqueeze(0)
                else:
                    grad_norms = list(map(lambda x: x.detach().norm(), grads))
                    inputs = grad_norms + [loss.detach()]
                    inputs = torch.stack(inputs, dim=0).unsqueeze(0)
                    inputs = self.smooth(inputs)[0]
                stop_gate, self.stop_hx = self.stop_gate(inputs, self.stop_hx)
                return stop_gate
        return loss.new_zeros(1, dtype=torch.float)

    def forward(self, *input, **kwargs):
        for _ in self.step_forward(*input, **kwargs):
            pass

    @staticmethod
    def smooth(weight, p=10, eps=1e-20):
        weight_abs = weight.abs()
        less = (weight_abs < math.exp(-p)).type(torch.float)
        noless = 1.0 - less
        log_weight = less * -1 + noless * torch.log(weight_abs + eps) / p
        sign = less * math.exp(p) * weight + noless * weight.sign()
        return log_weight, sign

    def step_forward(self, support_pairs, candidates, step_forward=False):
        """
        :param query_items: (1, batch_size,)
        :param query_users: (1, batch_size,)
        :param support_pairs: (1, few_size, 2)
        :param candidates: list of python
        :return:
        """
        named_modules = dict(self.model.named_modules())
        support_users, support_items = support_pairs[:, 0], support_pairs[:, 1]
        support_users = self.useritem_embeds(*self.user_neighbors(support_users), is_user=True)
        support_items = self.useritem_embeds(*self.item_neighbors(support_items), is_user=False)
        for step in range(self.max_step):
            if len(support_pairs) >= self.batch_size:
                rand_index = random.sample(range(len(support_pairs)), self.batch_size)
            else:
                rand_index = [random.randrange(len(support_pairs)) for _ in range(self.batch_size)]
            selected_users, positive_items = list(map(lambda x: x[rand_index], support_users)), list(
                map(lambda x: x[rand_index], support_items))
            negative_items = []
            for idx in rand_index:
                negative_item = random.choice(candidates)
                while negative_item == support_pairs[idx, 1]:
                    negative_item = random.choice(candidates)
                negative_items.append(negative_item)
            negative_items = self.useritem_embeds(*self.item_neighbors(negative_items), is_user=False,
                                                  with_neighbor=self.item_graph)
            if self.attack_model is not None:
                attack_users, attack_positives, attack_negatives = self.attack_model(self.model, selected_users,
                                                                                     positive_items, negative_items)
                selected_users = list(map(lambda x: torch.cat((x[0], x[1]), dim=0),
                                          zip(selected_users, attack_users)))
                positive_items = list(map(lambda x: torch.cat((x[0], x[1]), dim=0),
                                          zip(positive_items, attack_positives)))
                negative_items = list(map(lambda x: torch.cat((x[0], x[1]), dim=0),
                                          zip(negative_items, attack_negatives)))
            positive_values = self.model(selected_users, positive_items, with_attr=True)
            negative_values = self.model(selected_users, negative_items, with_attr=True)
            loss = self.loss_function(positive_values, negative_values)
            grads = torch.autograd.grad(loss, self.aim_parameters)
            self.store['loss'].append(loss.item())
            self.store['grads'].append(list(map(lambda x: x.norm().item(), grads)))
            stop_gate = self.stop(self.current_step, loss, grads)
            self.store['stop_gates'].append(stop_gate.item())
            if step >= self.min_step and random.random() < stop_gate:
                break
            if step_forward:
                yield stop_gate
            self.aim_parameters = self.update(self.aim_parameters, loss, grads)
            for weight, (layer_name, param_name) in zip(self.aim_parameters, self.learned_params):
                layer = named_modules[layer_name]
                setattr(layer, param_name, weight)
        # if query_users is not None and query_items is not None:
        #     if not with_attr:
        #         query_users, query_items = query_users.squeeze(0), query_items.squeeze(0)
        #         query_users = self.user_neighbors(query_users)
        #         query_items = self.item_neighbors(query_items)
        #     return self.model(query_users, query_items, with_attr=with_attr)


class SGDModel(GradientModel):
    def __init__(self, *params, lr, **meta_config):
        super(SGDModel, self).__init__(*params, **meta_config)
        self.lr = lr

    def update(self, parameters, loss, grads):
        new_parameters = []
        for num, (param, grad) in enumerate(zip(parameters, grads)):
            weight = param - self.lr * grad
            new_parameters.append(weight)
        return new_parameters


class LSTMLearner(GradientModel):
    class MetaLSTM(nn.Module):
        def __init__(self, hidden_size, layer_norm=False, input_gate=True, forget_gate=True):
            nn.Module.__init__(self)
            self.hidden_size = hidden_size
            # gradient(2), param(2), loss
            self.lstm = nn.LSTMCell(input_size=5, hidden_size=hidden_size)
            if layer_norm:
                self.layer_norm = nn.LayerNorm(hidden_size)
            else:
                self.layer_norm = None
            self.input_gate = input_gate
            self.forget_gate = forget_gate
            if self.input_gate:
                self.lr_layer = nn.Linear(hidden_size, 1)
                self.lrs = []
            else:
                self.output_layer = nn.Linear(hidden_size, 1)
                self.dets = []
            if forget_gate:
                self.fg_layer = nn.Linear(hidden_size, 1)
                self.fgs = []
            self.h_0 = nn.Parameter(torch.randn((hidden_size,), requires_grad=True))
            self.c_0 = nn.Parameter(torch.randn((hidden_size,), requires_grad=True))

        def weight_init(self):
            if self.input_gate:
                nn.init.xavier_normal_(self.lr_layer.weight)
                self.lr_layer.bias.data.fill_(0.0)
            else:
                nn.init.xavier_normal_(self.output_layer.weight)
                self.output_layer.weight.data /= 1000.0
                self.output_layer.bias.data.fill_(0.0)
            if self.forget_gate:
                nn.init.xavier_normal_(self.fg_layer.weight)
                self.fg_layer.bias.data.fill_(5.0)
            self.lstm.reset_parameters()
            hidden_size = self.lstm.hidden_size
            self.lstm.bias_ih.data[hidden_size // 4: hidden_size // 2].fill_(1.0)
            self.lstm.bias_hh.data[hidden_size // 4: hidden_size // 2].fill_(0.0)
            self.h_0.data = torch.randn((self.hidden_size,), requires_grad=True)
            self.c_0.data = torch.randn((self.hidden_size,), requires_grad=True)

        def forward(self, grad_norm, grad_sign, param_norm, param_sign, loss_norm, hx):
            batch_size = grad_norm.size(0)
            inputs = torch.stack((grad_norm, grad_sign, param_norm, param_sign, loss_norm.expand(grad_norm.size(0))),
                                 dim=1)
            if hx is None:
                self.lrs = []
                if self.forget_gate:
                    self.fgs = []
                hx = (self.h_0.expand((batch_size, -1)), self.c_0.expand((batch_size, -1)))
            h, c = self.lstm(inputs, hx)
            if self.layer_norm is not None:
                h = self.layer_norm(h)
            if self.input_gate:
                lr = torch.sigmoid(self.lr_layer(h))
            else:
                lr = self.output_layer(h)
            self.lrs.append(lr.mean().item())
            if self.forget_gate:
                fg = torch.sigmoid(self.fg_layer(h))
                self.fgs.append(fg.mean().item())
                return lr, fg, (h, c)
            else:
                return lr, (h, c)

    def __init__(self, *params, input_gate, forget_gate, lstm_config, **meta_config):
        super(LSTMLearner, self).__init__(*params, **meta_config)
        meta_lstms = []
        for _ in self.learned_params:
            lstm = self.MetaLSTM(input_gate=input_gate, forget_gate=forget_gate, **lstm_config)
            lstm.weight_init()
            meta_lstms.append(lstm)
        self.input_gate = input_gate
        self.forget_gate = forget_gate
        self.hxs = []
        self.meta_lstms = nn.ModuleList(meta_lstms)

    def init(self):
        GradientModel.init(self)
        self.store['input_gates'] = []
        self.store['forget_gates'] = []
        self.hxs = []
        for _ in self.learned_params:
            self.hxs.append(None)

    def update(self, parameters, loss, grads):
        loss = loss.detach()
        smooth_loss = self.smooth(loss)[0]
        new_parameters = []
        lrs, fgs = [], []
        for num, (meta_lstm, param, grad, hx) in enumerate(
                zip(self.meta_lstms, parameters, grads, self.hxs)):
            grad.clamp_(-1.0, 1.0)
            flat_grad, flat_param = grad.view(-1), param.detach().view(-1)
            smooth_grad, smooth_param = self.smooth(flat_grad), self.smooth(flat_param)
            if self.forget_gate:
                lr, fg, hx = meta_lstm(*smooth_grad, *smooth_param, smooth_loss, hx)
                lrs.append(lr.mean().item())
                fgs.append(fg.mean().item())
                lr, fg = lr.view_as(grad), fg.view_as(grad)
                weight = fg * param
            else:
                lr, hx = meta_lstm(*smooth_grad, *smooth_param, smooth_loss, hx)
                lr = lr.view_as(grad)
                weight = param
            if self.input_gate:
                weight -= lr * grad
            else:
                weight += lr
            new_parameters.append(weight)
            self.hxs[num] = hx
        self.store['input_gates'].append(lrs)
        self.store['forget_gates'].append(fgs)
        return new_parameters


class PopularityModel(nn.Module):
    def __init__(self, *params, **kwparams):
        nn.Module.__init__(self)
        self.flag = nn.Parameter(torch.empty(0), requires_grad=False)
        self.popularity = {}

    def init(self):
        self.popularity = defaultdict(int)

    def forward(self, support_pairs, support_users, support_items, candidates, with_attr=False):
        self.popularity = defaultdict(int)
        support_pairs = support_pairs.squeeze(0)
        for user_id, item_id in support_pairs.tolist():
            self.popularity[item_id] += 1

    def value(self, query_users, query_items):
        result = []
        for item_id in query_items[0].squeeze(0).tolist():
            result.append(self.popularity[item_id])
        result = torch.tensor(result, device=self.flag.device, dtype=torch.float)
        return result


class SimilarityLearner(nn.Module):
    def __init__(self, user_neighbors, item_neighbors, useritem_embeds):
        super(SimilarityLearner, self).__init__()
        self.user_neighbors = user_neighbors
        self.item_neighbors = item_neighbors
        self.useritem_embeds = useritem_embeds

    def forward(self, query_users, query_items, support_users, support_items):
        """
        :param query_users: (batch_size,)
        :param query_items: (batch_size,)
        :param support_users: (few_size,)
        :param support_items: (few_size,)
        :return: (batch_size, )
        :return:
        """
        query_users, query_items = self.user_embeds(query_users), self.item_embeds(query_items)
        support_users, support_items = self.user_embeds(support_users), self.item_embeds(support_items)
        similarity = F.softmax(F.cosine_similarity(query_users.unsqueeze(-1), support_users.t().unsqueeze(0)), dim=1)
        item_embeds = torch.matmul(similarity, support_items)
        return F.cosine_similarity(query_items, item_embeds, dim=1)


class CoNetTrain(nn.Module):
    def __init__(self, source_ratings, source_candidates, item_padding_idx, model, step, lr, batch_size, negative_ratio,
                 sparse_ratio):
        nn.Module.__init__(self)
        self.model = model
        self.source_ratings = source_ratings
        self.source_candidates = source_candidates
        self.item_padding_idx = item_padding_idx
        self.step = step
        self.lr = lr
        self.negative_ratio = negative_ratio
        self.sparse_ratio = sparse_ratio
        self.criterion = nn.BCEWithLogitsLoss()
        self.batch_size = batch_size

    @staticmethod
    def transfer_generator(source_itemset, target_data, source_candidates, target_candidates, item_padding_idx,
                           batch_size, negative_ratio=1):
        while True:
            if len(target_data) >= batch_size:
                target_batch = random.sample(target_data, batch_size)
            else:
                target_batch = random.choices(target_data, k=batch_size)
            target_users = list(map(lambda x: x[0], target_batch))
            target_items = list(map(lambda x: x[1], target_batch))
            target_labels = [1.0] * batch_size
            for user_id, item_id in target_batch:
                for i in range(negative_ratio):
                    neg_item = random.choice(target_candidates)
                    while neg_item == item_id:
                        neg_item = random.choice(target_candidates)
                    target_items.append(neg_item)
                    target_users.append(user_id)
                    target_labels.append(0.0)
            source_users = target_users[:]
            source_items, source_labels = [], []
            for user_id in source_users:
                if random.random() < negative_ratio / (negative_ratio + 1):
                    neg_item = random.choice(source_candidates)
                    while neg_item in source_itemset[user_id]:
                        neg_item = random.choice(source_candidates)
                    source_items.append(neg_item)
                    source_labels.append(0.0)
                else:
                    if source_itemset[user_id]:
                        source_items.append(random.choice(source_itemset[user_id]))
                    else:
                        source_items.append(item_padding_idx)
                    source_labels.append(1.0)
            yield target_users, target_items, source_items, target_labels, source_labels

    def init(self):
        self.loss = []
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()

    def forward(self, support_pairs, support_users, support_items, support_candidates, with_attr=False):
        self.loss = []
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()
        parameters = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        optimizer = torch.optim.Adam(parameters, lr=self.lr)
        user_set = set(map(lambda x: x[0], support_pairs.tolist()))
        source_data = {user_id: self.source_ratings[user_id] for user_id in user_set}
        for batch_id, data in enumerate(
                self.transfer_generator(source_data, support_pairs.tolist(), self.source_candidates, support_candidates
                    , self.item_padding_idx, self.batch_size, self.negative_ratio)):
            data = list(map(lambda x: torch.tensor(x, dtype=torch.long, device=support_pairs.device), data))
            target_users, target_items, source_items, target_labels, source_labels = data
            target_values, source_values = self.model((target_users,), (target_items,), (source_items,))
            loss = self.criterion(target_values, target_labels.type(torch.float)) + self.criterion(source_values,
                                                                                                   source_labels.type(
                                                                                                       torch.float))
            for module in self.model.transfer_layers.modules():
                if isinstance(module, nn.Linear):
                    loss += self.sparse_ratio * torch.abs(module.weight).sum()
            self.loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(parameters, 0.25)
            optimizer.step()
            if batch_id > self.step:
                break


class SimpleModel(nn.Module):
    def __init__(self, model, device):
        super(SimpleModel, self).__init__()
        self.model = model
        self.device = device

    def forward(self, *data):
        data = list(map(lambda x: torch.tensor(x, dtype=torch.long, device=self.device), data))
        support_users, support_items, query_users, query_items = data
        if query_users.size(0) != query_items.size(0):
            query_users = query_users.expand(query_items.size(0))
        values = self.model(query_users, query_items, support_users, support_items)
        return values


class EmbedMatcher(nn.Module):
    def __init__(self, user_embeds, item_embeds):
        super(EmbedMatcher, self).__init__()
        self.user_embeds = user_embeds
        self.item_embeds = item_embeds
        self.similarity = nn.CosineSimilarity(dim=1)

    def forward(self, query_users, query_items, support_users=None, support_items=None):
        user_embeds = self.user_embeds(query_users)
        item_embeds = self.item_embeds(query_items)
        return self.similarity(user_embeds, item_embeds)


class ConfigureModel(nn.Module):
    def __init__(self, useritem_embeds, model, hidden_size, bidirectional, input_size,
                 layer_num):
        nn.Module.__init__(self)
        self.useritem_embeds = useritem_embeds
        self.model = model
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=False,
                            bidirectional=bidirectional)
        self.support_proj = MLP(input_size=2 * input_size, hidden_layers=[(input_size, True, 0.2)],
                                normalization='layer_norm', activation='relu')
        self.bidirectional = bidirectional
        self.layer_num = layer_num
        if bidirectional:
            hidden_size *= 2
        self.user_attn = nn.Linear(hidden_size, layer_num * input_size * 2)
        self.item_attn = nn.Linear(hidden_size, layer_num * input_size * 2)

    def forward(self, support_users, support_items):
        support_users = self.useritem_embeds(*support_users, is_user=True, with_neighbor=False)
        support_items = self.useritem_embeds(*support_items, is_user=False, with_neighbor=False)
        support_embeds = torch.cat((support_users, support_items), dim=1)  # (batch_size, embed_size * 2)
        support_embeds = self.support_proj(support_embeds)
        support_embeds = F.relu(support_embeds)
        _, (h, c) = self.lstm(support_embeds.unsqueeze(1))
        h = h.view(1, -1)
        user_attn = self.user_attn(h).view(self.layer_num, 2, -1)
        for i in range(self.layer_num):
            self.model.user_gcn.attn_heads[i].configure(user_attn[i])
        item_attn = self.item_attn(h).view(self.layer_num, 2, -1)
        for i in range(self.layer_num):
            self.model.item_gcn.attn_heads[i].configure(item_attn[i])

# class Matcher(nn.Module):
#     def __init__(self, user_embeds, item_embeds, support_encoder_config, query_encoder_config):
#         super(Matcher, self).__init__()
#         self.user_embeds = user_embeds
#         self.item_embeds = item_embeds
#         self.support_encoder = SupportEncoder(**support_encoder_config)
#         self.query_encoder = QueryEncoder(**query_encoder_config)
#         self.similarity = nn.CosineSimilarity(dim=1)
#
#     def forward(self, query_users, query_items, support_users, support_items):
#         """
#         :param query_users: (batch_size,)
#         :param query_items: (batch_size,)
#         :param support_users: (few_size,)
#         :param support_items: (few_size,)
#         :return: (batch_size, )
#         """
#         query_embeds = torch.cat((self.user_embeds(query_users), self.item_embeds(query_items)), dim=1)
#         support_embeds = torch.cat((self.user_embeds(support_users), self.item_embeds(support_items)), dim=1)
#         support_embeds = self.support_encoder(support_embeds)
#         query_embeds = self.query_encoder(query_embeds, support_embeds)
#         query_embeds = query_embeds.unsqueeze(-1)
#         support_embeds = support_embeds.transpose(0, 1).unsqueeze(0)
#         return self.similarity(query_embeds, support_embeds).mean(dim=1)
