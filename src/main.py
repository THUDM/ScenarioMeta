import torch
import torch.nn as nn
import torch.optim as optim
import time, functools, itertools, os, argparse
import logging
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils import unserialize, serialize, divide_dataset
from evaluate import topNRecall, multi_mean_measure
from utils import NeighborDict, UserItemEmbeds, filter_statedict
import loss
from dataset import train_generator, evaluate_generator, task_preprocess, divide_support
from modules import EmbedRecommender, InteractionRecommender, HybridRecommender
from meta import LSTMLearner

logging.basicConfig(format='%(asctime)s - %(levelname)s -   '
                           '%(message)s',
                    datefmt='%m/%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_embedding(user_embedding, item_embedding):
    user_padding_idx = user_embedding.size(0)
    user_embedding = torch.cat(
        (user_embedding, torch.zeros(1, user_embedding.size(1))), dim=0)
    item_padding_idx = item_embedding.size(0)
    item_embedding = torch.cat(
        (item_embedding, torch.zeros(1, item_embedding.size(1))), dim=0)

    user_embedding = torch.nn.Embedding.from_pretrained(user_embedding)
    item_embedding = torch.nn.Embedding.from_pretrained(item_embedding)
    useritem_embeds = UserItemEmbeds(user_embedding, item_embedding)
    return useritem_embeds, user_padding_idx, item_padding_idx


def get_neighbor_dict(back_ratings, user_dict, item_dict, user_padding_idx, item_padding_idx):
    user_neighbor_dict = [set() for _ in range(len(user_dict))]
    item_neighbor_dict = [set() for _ in range(len(item_dict))]
    for user_id, item_id, _, _ in back_ratings:
        user_id = user_dict[user_id]
        item_id = item_dict[item_id]
        user_neighbor_dict[user_id].add(item_id)
    user_neighbor_dict = [list(items) for items in user_neighbor_dict]
    user_neighbor_dict = NeighborDict(user_neighbor_dict, max_degree=192, padding_idx=item_padding_idx)
    item_neighbor_dict = [list(users) for users in item_neighbor_dict]
    item_neighbor_dict = NeighborDict(item_neighbor_dict, max_degree=192, padding_idx=user_padding_idx)
    return user_neighbor_dict, item_neighbor_dict


def get_data(train_data, test_data, user_dict, item_dict):
    train_data = [
        list(
            map(lambda x: (user_dict[x[0]], item_dict[x[1]]),
                filter(lambda x: x[0] in user_dict and x[1] in item_dict, value))) for key, value in train_data
    ]
    train_data, valid_data, _ = divide_dataset(train_data, valid_ratio=0.05, test_ratio=0.0)
    logger.info(
        "train tasks: {} ratings: {} valid tasks: {} ratings: {}".format(len(train_data), sum(map(len, train_data)),
                                                                         len(valid_data), sum(map(len, valid_data))))
    train_data = task_preprocess(train_data)
    valid_data = divide_support(valid_data)
    valid_support, valid_eval = list(map(lambda x: x[0], valid_data)), list(map(lambda x: x[1], valid_data))
    valid_candidates = [
        list({item_id
              for user_id, item_id in task})
        for task in map(lambda x: itertools.chain(*x), zip(valid_support, valid_eval))
    ]
    _, _, valid_truth = task_preprocess(valid_eval)
    test_support = [
        list(
            map(lambda x: (user_dict[x[0]], item_dict[x[1]]),
                filter(lambda x: x[0] in user_dict and x[1] in item_dict, support)))
        for key, support, evaluate in test_data
    ]
    test_eval = [
        list(
            map(lambda x: (user_dict[x[0]], item_dict[x[1]]),
                filter(lambda x: x[0] in user_dict and x[1] in item_dict, evaluate)))
        for key, support, evaluate in test_data
    ]

    test_candidates = [
        list({item_id
              for user_id, item_id in task})
        for task in map(lambda x: itertools.chain(*x), zip(test_support, test_eval))
    ]

    _, _, test_truth = task_preprocess(test_eval)
    return train_data, (valid_support, valid_candidates, valid_truth), (test_support, test_candidates, test_truth)


def get_model(useritem_embeds, user_neighbor_dict, item_neighbor_dict, criterion, config):
    model_type = config['recommender'].pop('model_type').lower()
    if model_type == 'mapping':
        model = EmbedRecommender(useritem_embeds, **config['recommender'])
    elif model_type == 'interaction':
        model = InteractionRecommender(useritem_embeds, config['recommender'])
    elif model_type == 'hybrid':
        model = HybridRecommender(useritem_embeds, **config['recommender'])
    else:
        raise NotImplemented
    if 'support' in config:
        gradient_model = LSTMLearner(user_neighbor_dict, item_neighbor_dict, useritem_embeds, model, criterion,
                                     attack_model=None, **config['support'])
        return gradient_model, model
    else:
        return model


def get_optimizer(gradient_model, config):
    if config['support'].get('flexible_step', False):
        stop_parameters = list(filter(lambda p: p.requires_grad, gradient_model.stop_gate.parameters()))
    else:
        stop_parameters = []
    init_parameters = list(filter(lambda p: p.requires_grad, gradient_model.model.parameters()))
    update_parameters = list(filter(lambda p: p.requires_grad, gradient_model.meta_lstms.parameters()))
    parameters = [
        {'params': init_parameters, 'lr': config['lr']['init_lr']},
        {'params': update_parameters, 'lr': config['lr']['update_lr']},
        {'params': stop_parameters, 'lr': config['lr']['stop_lr']}
    ]
    optimizer = optim.Adam(parameters, **config['optim'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=2,
                                                           verbose=True, min_lr=1e-6)
    return optimizer, scheduler


def evaluate(data_loader, matcher, configure, useritem_embeds, user_neighbor_dict, item_neighbor_dict, device):
    for data in data_loader:
        support_pairs, candidates, task_iterator = data
        support_pairs = torch.tensor(support_pairs, device=device)
        support_users, support_items = support_pairs[:, 0], support_pairs[:, 1]
        support_users = useritem_embeds(*user_neighbor_dict(support_users), is_user=True)
        support_items = useritem_embeds(*item_neighbor_dict(support_items), is_user=False)
        if configure is not None:
            configure.init()
            configure(support_pairs, candidates)
        with torch.no_grad():
            for positive_users, positive_items, negative_users, negative_items in task_iterator:
                query_users = positive_users[:1]
                query_items = positive_items + negative_items
                positive_num = len(positive_users)
                values = matcher(user_neighbor_dict(query_users), item_neighbor_dict(query_items))
                _, index = values.sort(dim=0, descending=True)
                index = index.tolist()
                yield (range(positive_num), index)


measure_dict = {
    'Recall_1': functools.partial(topNRecall, topn=1),
    'Recall_3': functools.partial(topNRecall, topn=3),
    'Recall_5': functools.partial(topNRecall, topn=5),
    'Recall_10': functools.partial(topNRecall, topn=10),
    'Recall_20': functools.partial(topNRecall, topn=20),
    'Recall_50': functools.partial(topNRecall, topn=50),
    'Recall_100': functools.partial(topNRecall, topn=100)
}

measure_keys, measure_funcs = list(map(lambda x: x[0], measure_dict.items())), list(
    map(lambda x: x[1], measure_dict.items()))


def train(modules, criterion, optimizer, scheduler, task_iter, valid_data, test_data, device, few_num, step_penalty,
          train_steps,
          data_directory, writer=None):
    gradient_model, model, useritem_embeds, user_neighbor_dict, item_neighbor_dict = modules
    parameters = list(filter(lambda p: p.requires_grad, gradient_model.parameters()))
    test_support, test_candidates, test_truth = test_data
    valid_support, valid_candidates, valid_truth = valid_data
    running_loss, running_steps, loss_descend = 0.0, 0, 0.0
    best_values = {}
    gradient_model.train()
    for batch_id, data in tqdm(task_iter):
        if batch_id > train_steps:
            break
        support_pairs, candidates, positive_users, positive_items, negative_users, negative_items = data
        step_losses, stop_gates, positive_num = [], [], len(positive_items)
        support_pairs = torch.tensor(support_pairs, device=device)
        query_users = user_neighbor_dict(torch.tensor(positive_users + negative_users, device=device))
        query_items = item_neighbor_dict(torch.tensor(positive_items + negative_items, device=device))
        if gradient_model is not None:
            gradient_model.init()
            for stop_gate in gradient_model.step_forward(support_pairs, candidates, step_forward=True):
                stop_gates.append(stop_gate)
                with torch.no_grad():
                    values = model(query_users, query_items, with_attr=False)
                positive_values, negative_values = values[:positive_num], values[positive_num:]
                step_loss = criterion(positive_values, negative_values)
                step_losses.append(step_loss)
        values = model(query_users, query_items, with_attr=False)
        positive_values, negative_values = values[:positive_num], values[positive_num:]
        final_loss = criterion(positive_values, negative_values)
        total_loss = 0.0
        for step, (step_loss, stop_gate) in enumerate(zip(step_losses, stop_gates)):
            log_prob = torch.log(1 - stop_gate)
            total_loss = -log_prob * ((step_loss - final_loss -
                                       (len(step_losses) - step) * step_penalty).detach()) + total_loss
        total_loss = final_loss + total_loss
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_value_(parameters, 0.25)
        optimizer.step()

        running_loss += final_loss.item()
        running_steps += len(step_losses)
        loss_descend += (final_loss - step_losses[0]).item()

        if (batch_id + 1) % 1000 == 0:
            if writer is not None:
                writer.add_scalar('loss', running_loss / 1000, batch_id)
                writer.add_scalar('step', running_steps / 1000, batch_id)
                writer.add_scalar('descend', loss_descend / 1000, batch_id)
            print("loss@%5d: %.4f" % (batch_id + 1, running_loss / 1000))
            running_loss, running_steps, loss_descend = 0.0, 0, 0.0
        if (batch_id + 1) % 5000 == 0:
            gradient_model.eval()
            valid_values = multi_mean_measure(
                evaluate(
                    evaluate_generator(valid_support, valid_truth, valid_candidates,
                                       few_num=few_num), model, gradient_model, useritem_embeds, user_neighbor_dict,
                    item_neighbor_dict, device), measure_funcs)
            test_values = multi_mean_measure(
                evaluate(
                    evaluate_generator(test_support, test_truth, test_candidates,
                                       few_num=few_num), model, gradient_model, useritem_embeds, user_neighbor_dict,
                    item_neighbor_dict, device), measure_funcs)
            test_output, valid_output = [], []
            update = False
            for i, key in enumerate(measure_keys):
                valid_value, test_value = valid_values[i], test_values[i]
                if key not in best_values or valid_value > best_values[key]:
                    best_values[key] = valid_value
                    if key == 'Recall_20':
                        update = True
                if writer is not None:
                    writer.add_scalar(key, test_value, batch_id)
                valid_output.append("%s:%.4f" % (key, valid_value))
                test_output.append("%s:%.4f" % (key, test_value))
            logger.info("  ".join(valid_output))
            if update:
                logger.info("  ".join(test_output))
                torch.save(
                    filter_statedict(gradient_model), os.path.join(data_directory,
                                                                   str(batch_id + 1) + ".dict"))
            scheduler.step(test_values[measure_keys.index('Recall_10')])
            gradient_model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--root_directory', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--comment', type=str, default='init')
    args = parser.parse_args()

    config = unserialize(args.config)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda:0')
    root_directory = args.root_directory
    # load data
    user_embedding = torch.from_numpy(
        unserialize(os.path.join(root_directory, "embeddings/user_embeddings.npy")).astype(np.float32))
    item_embedding = torch.from_numpy(
        unserialize(os.path.join(root_directory, "embeddings/item_embeddings.npy")).astype(np.float32))
    train_data = unserialize(os.path.join(root_directory, "train_data/train_data"))
    user_dict = unserialize(os.path.join(root_directory, "embeddings/user_dict"))
    item_dict = unserialize(os.path.join(root_directory, "embeddings/item_dict"))
    test_data = unserialize(os.path.join(root_directory, "train_data/test_data"))
    if config['recommender'].get('user_graph', False) or config['recommender'].get('item_graph', False):
        back_ratings = unserialize(os.path.join(root_directory, "train_data/back_ratings"))
        user_neighbor_dict, item_neighbor_dict = get_neighbor_dict(back_ratings, user_dict, item_dict)
    # data preprocess
    useritem_embeds, user_padding_idx, item_padding_idx = get_embedding(user_embedding, item_embedding)
    train_data, valid_data, test_data = get_data(train_data, test_data, user_dict, item_dict)
    if not config['recommender'].get('user_graph', False):
        user_neighbor_dict = NeighborDict(None)
    if not config['recommender'].get('item_graph', False):
        item_neighbor_dict = NeighborDict(None)
    # define model
    criterion = loss.__getattribute__(
        config['training']['loss'])(**config['training']['loss_config'])
    gradient_model, model = get_model(useritem_embeds, user_neighbor_dict, item_neighbor_dict, criterion, config)
    gradient_model.to(device)
    model.to(device)
    # optimizer
    config.setdefault('lr', {})
    for key in ['stop_lr', 'init_lr', 'update_lr']:
        config['lr'].setdefault(key, config['optim']['lr'])
    optimizer, scheduler = get_optimizer(gradient_model, config)
    # save
    if config['save']:
        project_name = args.comment
        data_directory = os.path.join(root_directory, "log", "-".join((project_name, time.strftime("%m-%d-%H"))))
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)
        log_file = os.path.join(data_directory, "_".join(("log", time.strftime("%m-%d-%H-%M"))))
        writer = SummaryWriter(log_file, comment='Normal')
        serialize(config, os.path.join(data_directory, "config.json"), in_json=True)
    else:
        data_directory, writer = None, None
    batch_size = config['training']['batch_size']
    task_iter = enumerate(
        train_generator(*train_data, batch_size, negative_ratio=config['training']['negative_ratio'],
                        few_num=config['few_num']))
    modules = (gradient_model, model, useritem_embeds, user_neighbor_dict, item_neighbor_dict)
    train(modules, criterion, optimizer, scheduler, task_iter, valid_data, test_data, device, few_num=config['few_num'],
          step_penalty=config['training']['step_penalty'], train_steps=config['training']['max_steps'],
          data_directory=data_directory, writer=writer)
