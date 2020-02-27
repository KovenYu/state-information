from resnet import *
from utils import *
import torch.nn as nn
import torch
import os
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from collections import Counter


class Trainer(object):
    def __init__(self):
        super(Trainer, self).__init__()

    def train(self, *names):
        """
        set the given attributes in names to the training state.
        if names is empty, call the train() method for all attributes which are instances of nn.Module.
        :param names:
        :return:
        """
        if not names:
            modules = []
            for attr_name in dir(self):
                attr = getattr(self, attr_name)
                if isinstance(attr, nn.Module):
                    modules.append(attr_name)
        else:
            modules = names

        for m in modules:
            getattr(self, m).train()

    def eval(self, *names):
        """
        set the given attributes in names to the evaluation state.
        if names is empty, call the eval() method for all attributes which are instances of nn.Module.
        :param names:
        :return:
        """
        if not names:
            modules = []
            for attr_name in dir(self):
                attr = getattr(self, attr_name)
                if isinstance(attr, nn.Module):
                    modules.append(attr_name)
        else:
            modules = names

        for m in modules:
            getattr(self, m).eval()


class ReidTrainer(Trainer):
    def __init__(self, args, logger, loader):
        super(ReidTrainer, self).__init__()
        self.args = args
        self.logger = logger

        self.disc_loss = nn.CrossEntropyLoss().cuda()
        self.align_loss = AlignLoss(args.batch_size).cuda()
        
        self.net = resnet50(pretrained=False, num_classes=args.pseudo_class).cuda()
        if args.pretrain_path is None:
            self.logger.print_log('do not use pre-trained model. train from scratch.')
        elif os.path.isfile(args.pretrain_path):
            checkpoint = torch.load(args.pretrain_path)
            state_dict = parse_pretrained_checkpoint(checkpoint, args.pseudo_class)
            state_dict = self.add_fc_dim(state_dict, loader)
            self.net.load_state_dict(state_dict, strict=False)
            self.logger.print_log('loaded pre-trained model from {}'.format(args.pretrain_path))
        else:
            self.logger.print_log('{} is not a file. train from scratch.'.format(args.pretrain_path))
        self.net = nn.DataParallel(self.net).cuda()

        bn_params, other_params = partition_params(self.net, 'bn')
        self.optimizer = torch.optim.SGD([{'params': bn_params, 'weight_decay': 0},
                                          {'params': other_params}], lr=args.lr, momentum=0.9, weight_decay=args.wd)
        self.lr_scheduler = MultiStepLR(self.optimizer, milestones=[int(args.epochs/8*5), int(args.epochs/8*7)])

        self.init_losses(loader)
        self.prior = torch.ones(args.pseudo_class).cuda()
        self.n_total = torch.full((args.pseudo_class,), len(loader.dataset)/args.pseudo_class)
        self.max_predominance_index = torch.zeros(args.pseudo_class)
        self.pseudo_label_memory = torch.full((len(loader.dataset),), -1, dtype=torch.long)
        self.view_memory = np.asarray(loader.dataset.views)

    def train_epoch(self, loader, epoch):
        self.lr_scheduler.step()
        batch_time_meter = AverageMeter()
        stats = ('loss_surrogate', 'loss_align', 'loss_total')
        meters_trn = {stat: AverageMeter() for stat in stats}
        self.train()

        end = time.time()
        for i, tuple in enumerate(loader):
            if i % self.args.prior_update_freq == 0:
                self.update_state()
                self.update_prior()
            imgs = tuple[0].cuda()
            views = tuple[2].cuda()
            idx_img = tuple[3]

            classifer = self.net.module.fc.weight.renorm(2, 0, 1e-5).mul(1e5)

            features, similarity, _ = self.net(imgs)
            scores = similarity * 30
            logits = F.softmax(features.mm(classifer.detach().t() * 30), dim=1)
            loss_align = self.align_loss(features, views)

            pseudo_labels = get_pseudo_labels(logits.detach()*self.prior)
            self.pseudo_label_memory[idx_img] = pseudo_labels.cpu()
            loss_surrogate = self.disc_loss(scores, pseudo_labels)

            self.optimizer.zero_grad()
            loss_total = loss_surrogate + self.args.lamb_align * loss_align
            loss_total.backward()
            self.optimizer.step()

            for k in stats:
                v = locals()[k]
                if v.item() > 0:
                    meters_trn[k].update(v.item(), self.args.batch_size)

            batch_time_meter.update(time.time() - end)
            freq = self.args.batch_size / batch_time_meter.avg
            end = time.time()
            if self.args.print_freq != 0 and i % self.args.print_freq == 0:
                self.logger.print_log('  Iter: [{:03d}/{:03d}]   Freq {:.1f}   '.format(
                    i, len(loader), freq) + create_stat_string(meters_trn) + time_string())

        save_checkpoint(self, epoch, os.path.join(self.args.save_path, "checkpoints.pth"))
        return meters_trn

    def eval_performance(self, loader, gallery_loader, probe_loader):
        stats = ('r1', 'r5', 'r10', 'MAP')
        meters_val = {stat: AverageMeter() for stat in stats}
        self.eval()

        gallery_features, gallery_labels, gallery_views = extract_features(gallery_loader, self.net, index_feature=0)
        probe_features, probe_labels, probe_views = extract_features(probe_loader, self.net, index_feature=0)
        dist = cdist(gallery_features, probe_features, metric='cosine')
        CMC, MAP = eval_cmc_map(dist, gallery_labels, probe_labels, gallery_views, probe_views, ignore_MAP=False)
        r1 = CMC[0]
        r5 = CMC[4]
        r10 = CMC[9]

        for k in stats:
            v = locals()[k]
            meters_val[k].update(v.item(), self.args.batch_size)
        return meters_val

    def eval_performance_mpie(self, target_loader, gallery_loader, probe_loader):
        stats = ('overall', 'd0', 'd15', 'd30', 'd45', 'd60')
        meters_val = {stat: AverageMeter() for stat in stats}
        self.eval()

        gallery_features, gallery_labels, gallery_views = extract_features(gallery_loader, self.net, index_feature=0)
        probe_features, probe_labels, probe_views = extract_features(probe_loader, self.net, index_feature=0)
        accuracy = []
        for v in np.unique(probe_views):
            idx = probe_views == v
            f = probe_features[idx]
            l = probe_labels[idx]
            dist = cdist(gallery_features, f, metric='cosine')
            acc = eval_acc(dist, gallery_labels, l)
            accuracy.append(acc)
            self.logger.print_log('view : {}, acc : {:2f}'.format(v, acc))
        accuracy = np.array(accuracy)
        overall = accuracy.mean()
        d0 = accuracy[4]
        d15 = (accuracy[3]+accuracy[5])/2
        d30 = (accuracy[2]+accuracy[6])/2
        d45 = (accuracy[1]+accuracy[7])/2
        d60 = (accuracy[0] + accuracy[8]) / 2
        for k in stats:
            v = locals()[k]
            meters_val[k].update(v.item(), self.args.batch_size)
        return meters_val

    def eval_performance_cfp(self, test_loader, protocol):
        stats = ('acc_FP', 'EER_FP', 'AUC_FP')
        meters_val = {stat: AverageMeter() for stat in stats}
        self.eval()

        test_features, _, _ = extract_features(test_loader, self.net, index_feature=0, return_numpy=False)
        similarity = (test_features.matmul(test_features.t())+1)/2
        FP_same_idx = protocol['FP_same']
        FP_same = []
        for pair in FP_same_idx:
            sim = similarity[pair[0], pair[1]+500]
            FP_same.append(sim)
        FP_same = torch.stack(FP_same).cpu()
        FP_diff_idx = protocol['FP_diff']
        FP_diff = []
        for pair in FP_diff_idx:
            sim = similarity[pair[0], pair[1]+500]
            FP_diff.append(sim)
        FP_diff = torch.stack(FP_diff).cpu()
        acc_FP, EER_FP, AUC_FP = eval_CFP(FP_same, FP_diff)

        for k in stats:
            v = locals()[k]
            meters_val[k].update(v.item())
        return meters_val

    def init_losses(self, loader):
        if os.path.isfile(self.args.align_path):
            features, views = torch.load(self.args.align_path)
            self.logger.print_log('loaded features from {}'.format(self.args.align_path))
        else:
            self.logger.print_log('not found {}. computing features...'.format(self.args.align_path))
            features, _, views = extract_features(loader, self.net, index_feature=0, return_numpy=False)
            torch.save((features, views), self.args.align_path)
        self.align_loss.init_centers(features, views)
        self.logger.print_log('initializing align loss centers done.')

    def add_fc_dim(self, state_dict, loader, fc_layer_name='fc'):
        fc_weight_name = '{}.weight'.format(fc_layer_name)
        fc_weight = state_dict[fc_weight_name] if fc_weight_name in state_dict else torch.empty(0, 2048).cuda()
        if os.path.isfile(self.args.centroids_path):
            renorm_centroids = torch.load(self.args.centroids_path)
            self.logger.print_log('loaded centroids from {}.'.format(self.args.centroids_path))
        else:
            self.logger.print_log('Not found {}. Evaluating centroids ..'.format(self.args.centroids_path))
            self.net.load_state_dict(state_dict, strict=False)
            self.eval()

            features, _, _ = extract_features(loader, self.net, index_feature=0)
            kmeans = KMeans(n_clusters=self.args.pseudo_class, n_init=2)
            kmeans.fit(features)
            centroids_np = kmeans.cluster_centers_
            centroids = torch.Tensor(centroids_np).cuda()

            fc_weights = self.net.fc.weight.data
            mean_norm = fc_weights.pow(2).sum(dim=1).pow(0.5).mean()
            renorm_centroids = centroids.renorm(p=2, dim=0, maxnorm=(1e-5) * mean_norm).mul(1e5)
            torch.save(renorm_centroids, self.args.centroids_path)
        new_fc_weight = torch.cat([fc_weight, renorm_centroids], dim=0)
        state_dict[fc_weight_name] = new_fc_weight
        self.logger.print_log('FC dimensions added.')
        return state_dict

    def update_state(self, moment=0.5):
        """
        according to self.pseudo_label_memory and self.view_memory,
        update self.n_total and self.max_predominance_index
        :return:
        """
        n_total = torch.zeros(self.args.pseudo_class)
        max_predominance_index = torch.ones(self.args.pseudo_class)
        for i in range(self.args.pseudo_class):
            idx = self.pseudo_label_memory == i
            n_total[i] = idx.sum()
            views = self.view_memory[idx.nonzero().squeeze(dim=1).numpy()]
            t = tuple(views)
            if t:
                c = Counter(t)
                _, max_count = c.most_common(1)[0]
                max_predominance_index[i] = max_count/n_total[i]
        self.n_total = moment * self.n_total + (1-moment) * n_total
        self.max_predominance_index = moment * self.max_predominance_index + (1-moment) * max_predominance_index

    def update_prior(self):
        for i in range(self.args.pseudo_class):
            a = self.args.a
            b = self.args.b
            x = self.max_predominance_index[i]
            self.prior[i] = 1/(1+np.exp(a*(x-b)))


class AlignLoss(torch.nn.Module):
    def __init__(self, batch_size):
        super(AlignLoss, self).__init__()
        self.moment = batch_size / 10000
        self.initialized = False

    def init_centers(self, variables, views):
        """
        :param variables: shape=(N, n_class)
        :param views: (N,)
        :return:
        """
        univiews = torch.unique(views)
        mean_ml = []
        std_ml = []
        for v in univiews:
            ml_in_v = variables[views == v]
            mean = ml_in_v.mean(dim=0)
            std = ml_in_v.std(dim=0)
            mean_ml.append(mean)
            std_ml.append(std)
        center_mean = torch.mean(torch.stack(mean_ml), dim=0)
        center_std = torch.mean(torch.stack(std_ml), dim=0)
        self.register_buffer('center_mean', center_mean)
        self.register_buffer('center_std', center_std)
        self.initialized = True

    def _update_centers(self, variables, views):
        """
        :param variables: shape=(BS, n_class)
        :param views: shape=(BS,)
        :return:
        """
        univiews = torch.unique(views)
        means = []
        stds = []
        for v in univiews:
            ml_in_v = variables[views == v]
            if len(ml_in_v) == 1:
                continue
            mean = ml_in_v.mean(dim=0)
            means.append(mean)
            std = ml_in_v.std(dim=0)
            stds.append(std)
        new_mean = torch.mean(torch.stack(means), dim=0)
        self.center_mean = self.center_mean*(1-self.moment) + new_mean*self.moment
        new_std = torch.mean(torch.stack(stds), dim=0)
        self.center_std = self.center_std*(1-self.moment) + new_std*self.moment

    def forward(self, variables, views):
        """
        :param variables: shape=(BS, n_class)
        :param views: shape=(BS,)
        :return:
        """
        self._update_centers(variables.detach(), views)

        univiews = torch.unique(views)
        loss_terms = []
        for v in univiews:
            ml_in_v = variables[views == v]
            if len(ml_in_v) == 1:
                continue
            mean = ml_in_v.mean(dim=0)
            loss_mean = (mean - self.center_mean).pow(2).sum()
            loss_terms.append(loss_mean)
            std = ml_in_v.std(dim=0)
            loss_std = (std - self.center_std).pow(2).sum()
            loss_terms.append(loss_std)
        loss_total = torch.mean(torch.stack(loss_terms))
        return loss_total


def get_pseudo_labels(similarity):
    """
    :param similarity: torch.Tensor, shape=(BS, n_classes)
    :return:
    """
    sim = similarity
    max_entries = torch.argmax(sim, dim=1)
    pseudo_labels = max_entries
    return pseudo_labels.cuda()
