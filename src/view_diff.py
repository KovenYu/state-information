from trainers import *
import time
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import sys
sys.path.append("pycharm-debug-py3k.egg")
import pydevd
pydevd.settrace('172.18.217.207', port=8008, stdoutToServer=True, stderrToServer=True)


def main():
    opts = BaseOptions()
    args = opts.parse()
    # occupy_gpu_memory(args.gpu)
    logger = Logger(args.save_path)
    opts.print_options(logger)
    target_loader, gallery_loader, probe_loader = \
        get_reid_dataloaders(args.target, args.img_size,
                             args.crop_size, args.padding, args.batch_size)

    if args.resume:
        trainer, start_epoch = load_checkpoint(args, logger)
    else:
        trainer = ReidTrainer(args, logger, target_loader)
        start_epoch = 0

    trainer.eval()
    features, labels, views = extract_features(target_loader, trainer.net, index_feature=0, return_numpy=False)
    univiews = torch.unique(views)
    view_features = []
    for v in univiews:
        idx = v == views
        v_feature = features[idx].mean(dim=0)
        view_features.append(v_feature)
    view_features = torch.stack(view_features)
    sim_matrix = view_features.matmul(view_features.t())
    avg_sim = 0
    n_views = len(univiews)
    for i in range(1, n_views):
        for j in range(i):
            avg_sim = avg_sim + sim_matrix[i, j]
    avg_sim = avg_sim / (n_views*(n_views-1)/2)
    print(avg_sim)

    target_loader, gallery_loader, probe_loader = \
        get_reid_dataloaders('Duke_384_128', args.img_size,
                             args.crop_size, args.padding, args.batch_size)
    trainer.eval()
    features, labels, views = extract_features(target_loader, trainer.net, index_feature=0, return_numpy=False)
    univiews = torch.unique(views)
    view_features = []
    for v in univiews:
        idx = v == views
        v_feature = features[idx].mean(dim=0)
        view_features.append(v_feature)
    view_features = torch.stack(view_features)
    sim_matrix_duke = view_features.matmul(view_features.t())
    avg_sim_duke = 0
    n_views = len(univiews)
    for i in range(1, n_views):
        for j in range(i):
            avg_sim_duke = avg_sim_duke + sim_matrix_duke[i, j]
    avg_sim_duke = avg_sim_duke / (n_views * (n_views - 1) / 2)
    print(avg_sim_duke)


if __name__ == '__main__':
    main()
