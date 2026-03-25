from torchvision import datasets
from torchvision.transforms import transforms
from options import args_parser
from Dataset.dataset import classify_label, show_clients_data_distribution, Indices2Dataset_labeled, \
    Indices2Dataset_unlabeled_fixmatch, partition_train
from Dataset.sample_dirichlet import clients_indices, clients_indices_homo
import numpy as np
import pandas as pd
from torch import max, eq, no_grad
from Dataset.CINIC10 import CINIC10
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch import nn
import torch.nn.functional as F
from Model.resnet import ResNet, ResNet_PC
from tqdm import tqdm
import copy
import torch
import random
from torch.utils.data import DataLoader, RandomSampler
import logging
import os
import time

worker_num = 4


class Global(object):
    def __init__(self, args):
        self.model = ResNet_PC(resnet_size=8, scaling=4,
                            save_activations=False, group_norm_num_groups=None,
                            freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes)

        self.model.cuda(args.gpu_id)
        self.num_classes = args.num_classes
        self.gpu_id = args.gpu_id

        self.GPT = nn.Linear(
            in_features=int(64 * self.model.scaling * self.model.block_fn.expansion),
            out_features=self.num_classes,
        )
        self.GPT.cuda(args.gpu_id)
        self.GPT_opt = SGD(self.GPT.parameters(), lr=args.lr_server)

        self.server_epochs = args.total_server_epochs // args.num_rounds
        self.ema_beta = 0.99

    def initialize_for_model_fusion(self, list_dicts_local_params: list, list_nums_local_data: list):

        fedavg_global_params = copy.deepcopy(list_dicts_local_params[0])
        fedavg_mlp_params = {}
        for name_param in list_dicts_local_params[0]:
            list_values_param = []
            for dict_local_params, num_local_data in zip(list_dicts_local_params, list_nums_local_data):
                list_values_param.append(dict_local_params[name_param] * num_local_data)

            value_global_param = sum(list_values_param) / sum(list_nums_local_data)
            fedavg_global_params[name_param] = value_global_param

            if "classifier" in name_param:
                new_key = name_param[len("classifier."):]
                fedavg_mlp_params[new_key] = value_global_param
            
        self.update_GPT(args, list_dicts_local_params, fedavg_mlp_params)
        for name, param in self.GPT.named_parameters():
            full_param_name = f"classifier.{name}"
            if full_param_name in fedavg_global_params:
                fedavg_global_params[full_param_name] = param.data.clone()
            else:
                logging.warning(f"Parameter {full_param_name} not found in global params")
        
        all_classifier_weights = []
        # 遍历每个客户端的参数
        for client_params in list_dicts_local_params:
            # 提取分类器权重 (假设键名为 'classifier.weight')
            if 'classifier.weight' in client_params:
                classifier_weights = client_params['classifier.weight']
                all_classifier_weights.append(classifier_weights)

        return fedavg_global_params, all_classifier_weights
    

    def update_GPT(self, args, list_dicts_local_params: list, fedavg_mlp_params: dict):
        self.GPT.load_state_dict(fedavg_mlp_params)
        self.GPT.train()
        uploaded_proxies = self.upload_proxies(list_dicts_local_params, fedavg_mlp_params)
        max_dist = self.update_max_dist(fedavg_mlp_params)

        # cdw 
        proxy_loader = DataLoader(uploaded_proxies, args.bs_server, drop_last=False, shuffle=True)
        for e in range(self.server_epochs):
            for proxy, y in proxy_loader:
                y = y.cuda(args.gpu_id)
                # print(y.shape, len(proxy_loader))
                proxy_g = self.GPT.weight
                dist = torch.cdist(proxy, proxy_g)

                one_hot = F.one_hot(y, self.num_classes).cuda(args.gpu_id)
                dist = dist + one_hot * min(max_dist.item(), args.gpt_threshold)
                loss = F.cross_entropy(-dist, y)

                self.GPT_opt.zero_grad()
                loss.backward()
                self.GPT_opt.step()
            
            logging.info(f'Server loss: {loss.item()}')

        self.GPT.eval()


    def upload_proxies(self, list_dicts_local_params: list, fedavg_mlp_params: dict):
        # 1. 从客户端参数中提取分类器权重并构建训练数据集
        all_classifier_weights = []
        all_class_labels = []
        client_proxies_dicts = []  # 存储每个客户端的代理字典

        # 遍历每个客户端的参数
        for client_params in list_dicts_local_params:
            # 提取分类器权重 (假设键名为 'classifier.weight')
            if 'classifier.weight' in client_params:
                classifier_weights = client_params['classifier.weight']
                
                # 检查权重形状 [num_classes, feature_dim]
                if classifier_weights.dim() == 2:
                    num_classes, feat_dim = classifier_weights.shape
                    client_proxies = {}

                    # 为每个类别创建样本
                    for class_id in range(num_classes):
                        weight_vector = classifier_weights[class_id]
                        all_classifier_weights.append(weight_vector)
                        all_class_labels.append(class_id)
                        client_proxies[class_id] = weight_vector
                    
                    client_proxies_dicts.append(client_proxies)
        
        # 转换为张量
        classifier_tensors = torch.stack(all_classifier_weights)
        label_tensors = torch.tensor(all_class_labels, dtype=torch.long)
        
        # 创建数据集 (特征向量 + 类别标签)
        uploaded_proxies = list(zip(classifier_tensors, label_tensors))
        return uploaded_proxies

    def update_max_dist(self, fedavg_mlp_params):
        avg_proxies = fedavg_mlp_params['weight']
        dist_mat = torch.cdist(avg_proxies, avg_proxies, p=2)  # (C, C)
        dist_mat.fill_diagonal_(float('inf'))
        min_dist_cls = torch.min(dist_mat, dim=-1)[0].cuda(self.gpu_id)  
        max_dist_all_cls = torch.max(min_dist_cls)

        return max_dist_all_cls
      
    def fedavg_eval(self, fedavg_params, data_test, batch_size_test):
        self.model.load_state_dict(fedavg_params)
        self.model.eval()
        with no_grad():
            test_loader = DataLoader(data_test, batch_size_test)
            num_corrects = 0
            for data_batch in test_loader:
                images, labels = data_batch
                images, labels = images.cuda(args.gpu_id), labels.cuda(args.gpu_id)
                _, outputs = self.model(images)
                _, predicts = max(outputs, -1)
                num_corrects += sum(eq(predicts.cpu(), labels.cpu())).item()
            accuracy = num_corrects / len(data_test)
        return accuracy

    def download_params(self):
        return self.model.state_dict()
    
    def update_global_distribution(self, current_global_dist=None, list_class_counts=None):
        """
        输入: 所有客户端上传的 class counts 列表 (list of numpy arrays)
        输出: 经过 EMA 平滑后的全局分布 (numpy array)
        """
        if current_global_dist is None:
            logging.info("Initialized Global EMA Distribution.")
            return np.ones(self.num_classes) / self.num_classes
        
        # 1. 计算本轮的实时分布 (Current Round Distribution)
        total_counts = np.sum(np.stack(list_class_counts), axis=0)
        total_sum = np.sum(total_counts)
        
        current_round_dist = total_counts / total_sum
        current_global_dist = (self.ema_beta * current_global_dist + (1 - self.ema_beta) * current_round_dist)
            
        logging.info(f'Current Round Dist: {current_round_dist}')
        logging.info(f'Updated EMA Global Dist: {current_global_dist}')

        return current_global_dist
        

class Local(object):
    def __init__(self, args):

        self.local_model = ResNet_PC(resnet_size=8, scaling=4,
                                  save_activations=False, group_norm_num_groups=None,
                                  freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes)


        self.local_G = ResNet_PC(resnet_size=8, scaling=4,
                              save_activations=False, group_norm_num_groups=None,
                              freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes)

        self.local_model.cuda(args.gpu_id)
        self.local_G.cuda(args.gpu_id)

        self.criterion = CrossEntropyLoss().cuda(args.gpu_id)
        self.optimizer = SGD(self.local_model.parameters(), lr=args.lr_local_training, momentum=0.9, weight_decay=1e-4)

        self.num_classes = args.num_classes        

    def fixmatch_train(self, args, data_client_labeled, data_client_unlabeled, global_params, r, client_idx, global_class_dist=None):

        self.labeled_trainloader = DataLoader(
            dataset=data_client_labeled,
            sampler=RandomSampler(data_client_labeled),
            batch_size=args.batch_size_local_labeled_fixmatch,
            drop_last=True,
            num_workers = 2,
            pin_memory=True
        )

        self.unlabeled_trainloader = DataLoader(
            dataset=data_client_unlabeled,
            sampler=RandomSampler(data_client_unlabeled),
            batch_size=args.batch_size_local_labeled_fixmatch * args.mu,
            drop_last=True,
            num_workers=2,
            pin_memory=True
        )

        self.local_model.load_state_dict(global_params)
        self.local_model.train()

        self.local_G.load_state_dict(global_params)
        self.local_G.eval()

        # 初始化本地类别计数器
        local_class_counts = torch.zeros(args.num_classes).cuda(args.gpu_id)

        # default: E = 5
        for local_epoch in range(args.local_epochs):

            labeled_iter = iter(self.labeled_trainloader)
            unlabeled_iter = iter(self.unlabeled_trainloader)

            local_iter = int(len(data_client_unlabeled) / args.batch_size_local_labeled_fixmatch)

            # cdw
            if local_epoch + 1 == args.local_epochs:
                num_pseudo_corrects = 0
                num_pseudo_total = 0
                num_u_valid = 0

            for epoch in range(local_iter):

                try:
                    inputs_x, targets_x = labeled_iter.__next__()
                except:
                    labeled_iter = iter(self.labeled_trainloader)
                    inputs_x, targets_x = labeled_iter.__next__()

                try:
                    inputs_u_w, inputs_u_s, targets_u_groundtruth = unlabeled_iter.__next__()
                except:
                    unlabeled_iter = iter(self.unlabeled_trainloader)
                    inputs_u_w, inputs_u_s, targets_u_groundtruth = unlabeled_iter.__next__()

                inputs_x, inputs_u_w, inputs_u_s = inputs_x.cuda(args.gpu_id), inputs_u_w.cuda(args.gpu_id), inputs_u_s.cuda(args.gpu_id)

                batch_size = inputs_x.shape[0]
                inputs = self.interleave(
                    torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2 * args.mu + 1).cuda(args.gpu_id) # 将有标签和无标签数据交错组合成一个大的批次
                targets_x = targets_x.cuda(args.gpu_id)

                targets_x_one_hot = F.one_hot(targets_x, args.num_classes).float()
                targets_u_groundtruth = targets_u_groundtruth.cuda(args.gpu_id)

                # local logits
                feats, logits = self.local_model(inputs)
                logits = self.de_interleave(logits, 2 * args.mu + 1)
                feats = self.de_interleave(feats, 2 * args.mu + 1) # cdw

                logits_x = logits[:batch_size]
                logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
                
                feats_x = feats[:batch_size]
                feats_u = feats[batch_size:]

                del logits, feats

                Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')
                _, logits_x_glob = self.local_G(inputs_x)
                probs_x_glob = torch.softmax(logits_x_glob.detach() / args.T, dim=-1)
                max_probs_x_glob, _ = torch.max(probs_x_glob, dim=-1)

                # global pseudo-labeling
                _, logits_u_w_global = self.local_G(inputs_u_w.cuda(args.gpu_id))
                pseudo_label_global = torch.softmax(logits_u_w_global.detach() / args.T, dim=-1)
                max_probs_global, targets_u_global = torch.max(pseudo_label_global, dim=-1)

                # local pseudo-labeling
                pseudo_label_local = torch.softmax(logits_u_w.detach() / args.T, dim=-1)
                max_probs_local, targets_u_local = torch.max(pseudo_label_local, dim=-1)

                targets_u_local_one_hot = F.one_hot(targets_u_local, args.num_classes).float()
                targets_u_global_one_hot = F.one_hot(targets_u_global, args.num_classes).float()

                mask_local = max_probs_local.ge(args.threshold).float()
                mask_global = max_probs_global.ge(args.threshold).float()
                mask_valid = torch.max(mask_local, mask_global)
                mask_x = max_probs_x_glob.ge(args.threshold).float()

                logits_u_s_probs = torch.softmax(logits_u_s, dim=-1)
                logits_u_s_probs = torch.softmax(logits_u_s, dim=-1) + 1e-10
                final_targets_u = targets_u_global_one_hot + 1e-10 # final_targets_u + 1e-10
                
                Lu = (F.kl_div(logits_u_s_probs.log(), final_targets_u, reduction='none').sum(-1) * mask_valid).mean()
                
                # print(mask_x.shape)
                projs_u = self.local_model.feat_proj(feats_u)
                projs_x = self.local_model.feat_proj(feats_x)
                projs = torch.cat([projs_x, projs_u], dim=0)
                
                projs_prob = torch.cat([probs_x_glob, pseudo_label_global, pseudo_label_global], dim=0).detach()
                conf_mask = torch.cat([mask_x, mask_valid, mask_valid], dim=0).bool()
                # print(projs.shape, projs_prob.shape, conf_mask.shape)
                current_dist_prior = global_class_dist if global_class_dist is not None else None
                Lc = self.ICPL(projs, projs_prob, conf_mask, conf_x_num=projs_x.shape[0], prior_distribution=current_dist_prior)

                loss = Lx + args.lambda_u * Lu + args.lambda_u * Lc

                # cdw
                if local_epoch + 1 == args.local_epochs:

                    ##############

                    # 1. 统计 Labeled 数据 (targets_x)
                    labels_one_hot = F.one_hot(targets_x, args.num_classes).float().sum(dim=0)
                    local_class_counts += labels_one_hot
                    
                    # 2. 统计 High-confidence Unlabeled 数据 (targets_u_global masked by mask_valid)
                    # 这里我们统计那些实际参与训练(被mask选中)的样本的伪标签类别
                    if mask_valid.sum() > 0:
                        valid_indices = mask_valid.bool()
                        pseudo_labels_selected = targets_u_global[valid_indices]
                        pseudo_one_hot = F.one_hot(pseudo_labels_selected, args.num_classes).float().sum(dim=0)
                        local_class_counts += pseudo_one_hot

                    ##############
                    num_pseudo_corrects += sum(eq(targets_u_global.cpu(), targets_u_groundtruth.cpu())).item()
                    num_pseudo_total += len(targets_u_global)
                    num_u_valid += sum(mask_valid).item()
                    
                    
                # logging
                logging.info(
                    f'Round {r}, Local Epoch {local_epoch}, Batch {epoch}: Lx = {Lx.item():.4f}, Lu = {Lu.item():.4f}, Lc = {Lc.item():.4f}')

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # cdw
            if local_epoch + 1 == args.local_epochs:
                pseudo_client_acc = num_pseudo_corrects / num_pseudo_total
                u_client_valid = num_u_valid / num_pseudo_total

                # logging
                logging.info(
                    f'Round {r}, Local Epoch {local_epoch}, Client {client_idx}, pseudo_acc = {pseudo_client_acc: .4f}, pseudo_num_valid = {num_u_valid}, valid_ratio = {u_client_valid}')
                
        pseudo_status = [num_pseudo_total, num_pseudo_corrects, num_u_valid, pseudo_client_acc, u_client_valid]
        return copy.deepcopy(self.local_model.state_dict()), pseudo_status, local_class_counts.cpu().numpy()
    
    def ICPL(self, feature, projs_prob, conf_mask, prior_distribution, conf_x_num=None): 
        feature = F.normalize(feature, p=2, dim=1)
        proxy = self.local_model.classifier.weight 
        proxy = F.normalize(self.local_model.proxy_proj(proxy), p=2.0, dim=1)   

        if not isinstance(prior_distribution, torch.Tensor):
            prior_tensor = torch.tensor(prior_distribution, device=projs_prob.device, dtype=projs_prob.dtype)
        else:
            prior_tensor = prior_distribution.to(projs_prob.device)
        # 维度对齐 (1, C)
        if prior_tensor.dim() == 1:
            prior_tensor = prior_tensor.unsqueeze(0)
        sets_class = (projs_prob > prior_tensor)

        top1_class = F.one_hot(projs_prob.argmax(dim=1), num_classes=projs_prob.size(1)).bool()
        pred_class = sets_class * ~conf_mask.unsqueeze(1) + top1_class * conf_mask.unsqueeze(1)

        # candidate-weighted proxy similarity
        candidate_weight = projs_prob * sets_class
        candidate_proxy = (candidate_weight.unsqueeze(2) * proxy.unsqueeze(0)).sum(dim=1)
        candidate_sim = (feature * candidate_proxy).sum(dim=1)

        # top1 proxy similarity
        pred = feature @ proxy.T
        target = projs_prob.argmax(dim=1)
        proxy_sim = pred[torch.arange(feature.size(0), device=feature.device), target]

        pos_pair = torch.where(conf_mask, proxy_sim, candidate_sim)

        # negative mask: samples with no overlapping predicted classes are negatives
        overlap = (pred_class.unsqueeze(1) & pred_class.unsqueeze(0)).any(dim=2)
        neg_matrix = ~overlap

        pairwise_sim = feature @ feature.T
        neg_pair = pairwise_sim.masked_fill(~neg_matrix | (pairwise_sim < 1e-6), float('-inf'))

        logits = torch.cat([pos_pair.unsqueeze(1), neg_pair], dim=1)

        if conf_x_num is not None:
            logits = logits[conf_x_num:]

        label = torch.zeros(logits.size(0), dtype=torch.long, device=feature.device)
        loss = F.cross_entropy(logits, label)

        return loss

    def interleave(self, x, size):
        s = list(x.shape)
        return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

    def de_interleave(self, x, size):
        s = list(x.shape)
        return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def fixmatch(alpha):

    args = args_parser()
    args.method = 'ProxyFL_{}k'.format(args.total_server_epochs // 1000)
    
    log_dir = f'./results/{args.dataset}/logs'
    os.makedirs(log_dir, exist_ok=True)
    cr_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    log_file = os.path.join(log_dir, '{method}_α={alpha}_{cr_time}.log'.format(method=args.method, alpha = alpha, cr_time=cr_time))

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=log_file
                        )

    if args.dataset == 'CIFAR10':
        args.num_classes = 10
        args.num_labeled = 500
        args.num_rounds = 300
        args.total_server_epochs = 30000
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        data_local_training = datasets.CIFAR10(args.path_cifar10, train=True, download=True, transform=None)
        data_global_test = datasets.CIFAR10(args.path_cifar10, train=False, transform=transform_test)

    elif args.dataset == 'CIFAR100': # training:50k; testing:10k; for training, each class includes 500 images
        args.num_classes = 100
        args.num_labeled = 50
        args.num_rounds = 500
        args.total_server_epochs = 5000
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        data_local_training = datasets.CIFAR100(args.path_cifar100, train=True, download=True, transform=None)
        data_global_test = datasets.CIFAR100(args.path_cifar100, train=False, transform=transform_test)

    elif args.dataset == 'SVHN':
        args.num_classes = 10
        args.num_labeled = 460
        args.num_rounds = 150
        args.total_server_epochs = 15000
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])
        data_local_training = datasets.SVHN(args.path_svhn, split='train', download=True, transform=None)
        data_global_test = datasets.SVHN(args.path_svhn, split='test', transform=transform_test, download=True)

    elif args.dataset == 'CINIC10':
        args.num_classes = 10
        args.num_labeled = 900
        args.num_rounds = 400
        args.total_server_epochs = 40000
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4789, 0.4723, 0.4305), (0.2421, 0.2383, 0.2587)),
        ])
        data_local_training = CINIC10(root=args.path_cinic10, split='train', transform=None)
        data_global_test = CINIC10(root=args.path_cinic10, split='test', transform=transform_test)

    else:
        print(
            f"Error: Unsupported dataset {args.dataset}. Please specify one of the following: CIFAR10, CIFAR100, CINIC10 or SVHN.")
        exit(1)


    logging.info(
        'dataset:{dataset}\n'
        'num_classes:{num_classes}\n'
        'num_labeled:{num_labeled}\n'
        'non_iid:{alpha}\n'
        'mu:{mu}\n'
        'num_rounds:{num_rounds}\n'
        'batch_label:{batch_label}, batch_unlabel:{batch_unlabel}'.format(
            dataset=args.dataset,
            num_classes=args.num_classes,
            num_labeled=args.num_labeled,
            alpha=alpha,
            mu=args.mu,
            num_rounds=args.num_rounds,
            batch_label=args.batch_size_local_labeled,
            batch_unlabel=args.batch_size_local_unlabeled,
        ))

    random_state = np.random.RandomState(args.seed)

    list_label2indices = classify_label(data_local_training, args.num_classes)

    list_label2indices_labeled, list_label2indices_unlabeled = partition_train(list_label2indices, args.num_labeled)

    # IID
    if alpha == 0:
        list_client2indices_labeled = clients_indices_homo(list_label2indices=list_label2indices_labeled,
                                                           num_classes=args.num_classes,
                                                           num_clients=args.num_clients)
        list_client2indices_unlabeled = clients_indices_homo(list_label2indices=list_label2indices_unlabeled,
                                                             num_classes=args.num_classes,
                                                             num_clients=args.num_clients)
    # Non-IID
    else:
        list_client2indices_labeled = clients_indices(list_label2indices=list_label2indices_labeled,
                                                      num_classes=args.num_classes,
                                                      num_clients=args.num_clients, non_iid_alpha=alpha, seed=0)
        list_client2indices_unlabeled = clients_indices(list_label2indices=list_label2indices_unlabeled,
                                                        num_classes=args.num_classes,
                                                        num_clients=args.num_clients, non_iid_alpha=alpha, seed=0)

    # Show data distribubution
    show_clients_data_distribution(data_local_training, list_client2indices_labeled,
                                   list_client2indices_unlabeled, args.num_classes)

    # add labeled samples without labels into the unlabeled dataset
    for client in range(args.num_clients):
        list_client2indices_unlabeled[client].extend(list_client2indices_labeled[client])

    global_model = Global(args)
    local_model = Local(args)

    total_clients = list(range(args.num_clients))


    indices2data_labeled = Indices2Dataset_labeled(data_local_training)
    indices2data_unlabeled = Indices2Dataset_unlabeled_fixmatch(data_local_training)

    fedavg_acc = []
    fedavg_pseudo_acc = []
    fedavg_num_valid = []
    fedavg_valid_ratio = []

    current_global_dist = global_model.update_global_distribution()
    
    # FL training
    for r in tqdm(range(1, args.num_rounds + 1), desc='Server'):

        list_local_class_counts = []

        dict_global_params = global_model.download_params()

        online_clients = random_state.choice(total_clients, args.num_online_clients, replace=False)
        list_dicts_local_params = []
        list_nums_local_data = []

        # client training
        num_clients_u_total = 0
        num_clients_u_corrects = 0
        num_clients_u_valid = 0

        for client in online_clients:
            indices2data_labeled.load(list_client2indices_labeled[client])
            data_client_labeled = indices2data_labeled
            indices2data_unlabeled.load(list_client2indices_unlabeled[client])
            data_client_unlabeled = indices2data_unlabeled


            list_nums_local_data.append(len(data_client_labeled) + len(data_client_unlabeled))
            local_params, pseudo_status, local_class_counts = local_model.fixmatch_train(args, data_client_labeled, data_client_unlabeled,
                                                                     copy.deepcopy(dict_global_params), r, client, 
                                                                     global_class_dist=current_global_dist)
            
            list_dicts_local_params.append(copy.deepcopy(local_params))
            # cdw: calculate the participating clients' acc and valid ratio
            num_clients_u_total += pseudo_status[0]
            num_clients_u_corrects += pseudo_status[1]
            num_clients_u_valid += pseudo_status[2]

            list_local_class_counts.append(local_class_counts) # 收集计数

        pseudo_acc = num_clients_u_corrects / num_clients_u_total
        pseudo_valid_ratio = num_clients_u_valid / num_clients_u_total
        fedavg_pseudo_acc.append(pseudo_acc)
        fedavg_valid_ratio.append(pseudo_valid_ratio)
        fedavg_num_valid.append(num_clients_u_valid)

        current_global_dist = global_model.update_global_distribution(current_global_dist, list_local_class_counts)
        
        fedavg_params, all_classifier_weights = global_model.initialize_for_model_fusion(list_dicts_local_params, list_nums_local_data)

        global_acc = global_model.fedavg_eval(copy.deepcopy(fedavg_params), data_global_test, args.batch_size_test)
        fedavg_acc.append(global_acc)
        
        print('round {round}, accuracy:{global_acc}, pseudo_acc:{fedavg_pseudo_acc}, num_valid:{fedavg_num_valid}, valid_ratio:{fedavg_valid_ratio}'.format(
            round = r,
            global_acc = global_acc,
            fedavg_pseudo_acc = fedavg_pseudo_acc[-1],
            fedavg_num_valid = fedavg_num_valid[-1],
            fedavg_valid_ratio = fedavg_valid_ratio[-1]))

        result_dir = f'./results/{args.dataset}'
        os.makedirs(result_dir, exist_ok=True)

        # make specific result dir by cdw
        result_dir_spec = f'{result_dir}/{args.method}_α={alpha}_{cr_time}'
        os.makedirs(result_dir_spec, exist_ok=True)

        if r == 1 or r == args.num_rounds or (r % 50 == 0 and r > 0.8 * args.num_rounds):
            # 保存 fedavg_params
            torch.save(fedavg_params, f'{result_dir_spec}/fedavg_params_round_{r}.pth')
            # 保存 all_classifier_weights
            torch.save(all_classifier_weights, f'{result_dir_spec}/all_classifier_weights_round_{r}.pt')
            # 同时保存为 numpy 格式以便后续分析
            classifier_weights_numpy = [weight.cpu().numpy() for weight in all_classifier_weights]
            np.save(f'{result_dir_spec}/all_classifier_weights_round_{r}.npy', classifier_weights_numpy)
            print(f"Saved model and proxy for round {r}")

        result_file = f'{result_dir}/{args.method}_α={alpha}_{cr_time}.csv'
        acc_num_pseudo_label_csv_col_name = ['acc', 'num_pseudo_label']
        acc_num_pseudo_label_csv_index = list(range(1, len(fedavg_acc)+1))
        acc_num_pseudo_label_csv_df = pd.DataFrame({'acc':fedavg_acc}, index = acc_num_pseudo_label_csv_index)
        # 保存文件
        acc_num_pseudo_label_csv_df.to_csv(result_file, encoding='utf8')

        result_pseudo_file = f'{result_dir}/{args.method}_α={alpha}_pseudo_{cr_time}.csv'
        # min_length = min(len(fedavg_pseudo_acc), len(fedavg_valid_ratio), len(fedavg_num_valid))
        min_length = min(len(fedavg_pseudo_acc), len(fedavg_valid_ratio), len(fedavg_num_valid), len(fedavg_acc),)
                        #  len(fedavg_low_conf_gt_in_candidates_ratio), len(fedavg_low_conf_pred_correct_ratio))
        
        # 创建 DataFrame，包含所有指标
        metrics_df = pd.DataFrame({
            'round': list(range(1, min_length + 1)),
            'acc': fedavg_acc[:min_length],
            'pseudo_acc': fedavg_pseudo_acc[:min_length],
            'valid_ratio': fedavg_valid_ratio[:min_length],
            'num_valid': fedavg_num_valid[:min_length],
        })
        # 设置轮次为索引
        metrics_df.set_index('round', inplace=True)

        # 保存文件
        metrics_df.to_csv(result_pseudo_file, encoding='utf8')
        print(f"Metrics saved to {result_pseudo_file}")




if __name__ == '__main__':

    torch.manual_seed(7)  # cpu
    torch.cuda.manual_seed(7)  # gpu
    np.random.seed(7)  # numpy
    random.seed(7)
    torch.backends.cudnn.deterministic = True  # cudnn

    args = args_parser()
    fixmatch(args.alpha)
