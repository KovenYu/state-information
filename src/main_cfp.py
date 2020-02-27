from trainers import *
import time
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")


def main():
    opts = BaseOptions()
    args = opts.parse()
    logger = Logger(args.save_path)

    acc_FP = []
    EER_FP = []
    AUC_FP = []
    for test_set in range(10):
        args.centroids_path = 'data/renorm_centroids_CFP_resnet50-vggface2_K500_{}.dat'.format(test_set)
        args.align_path = 'data/feature_CFP_resnet50-vggface2_{}.dat'.format(test_set)
        args.test_set = test_set
        opts.print_options(logger)
        train_loader, test_loader, protocol = \
            get_cfp_dataloaders(args.dataset_path, args.img_size, args.crop_size,
                                args.padding, args.batch_size, args.test_set)

        if args.resume:
            trainer, start_epoch = load_checkpoint(args, logger)
        else:
            trainer = ReidTrainer(args, logger, train_loader)
            start_epoch = 0

        total_epoch = args.epochs

        start_time = time.time()
        epoch_time = AverageMeter()

        for epoch in range(start_epoch, total_epoch):

            logger.print_log(
                '\n==>>{:s} [Epoch={:03d}/{:03d}]'.format(time_string(), epoch, total_epoch))

            meters_trn = trainer.train_epoch(train_loader, epoch)
            logger.print_log('  **Train**  ' + create_stat_string(meters_trn))

            epoch_time.update(time.time() - start_time)
            start_time = time.time()

        meters_val = trainer.eval_performance_cfp(test_loader, protocol)
        acc_FP.append(meters_val['acc_FP'].val)
        EER_FP.append(meters_val['EER_FP'].val)
        AUC_FP.append(meters_val['AUC_FP'].val)

    acc_m_FP = torch.Tensor(acc_FP).mean()
    acc_std_FP = torch.Tensor(acc_FP).std()
    EER_m_FP = torch.Tensor(EER_FP).mean()
    EER_std_FP = torch.Tensor(EER_FP).std()
    AUC_m_FP = torch.Tensor(AUC_FP).mean()
    AUC_std_FP = torch.Tensor(AUC_FP).std()

    logger.print_log(
        'acc_FP: {:.2%}({:.2%}), EER_FP: {:.2%}({:.2%}), AUC_FP: {:.2%}({:.2%})'.format(acc_m_FP, acc_std_FP, EER_m_FP,
                                                                                        EER_std_FP, AUC_m_FP,
                                                                                        AUC_std_FP))


if __name__ == '__main__':
    main()
