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
    opts.print_options(logger)
    loader, gallery_loader, probe_loader = \
        get_mpie_dataloaders(args.dataset_path, args.img_size,
                             args.crop_size, args.padding, args.batch_size)

    if args.resume:
        trainer, start_epoch = load_checkpoint(args, logger)
    else:
        trainer = ReidTrainer(args, logger, loader)
        start_epoch = 0

    total_epoch = args.epochs

    start_time = time.time()
    epoch_time = AverageMeter()

    for epoch in range(start_epoch, total_epoch):
        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (total_epoch - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
        logger.print_log(
            '\n==>>{:s} [Epoch={:03d}/{:03d}] {:s}'.format(time_string(), epoch, total_epoch, need_time))

        meters_trn = trainer.train_epoch(loader, epoch)
        logger.print_log('  **Train**  ' + create_stat_string(meters_trn))

        epoch_time.update(time.time() - start_time)
        start_time = time.time()

    meters_val = trainer.eval_performance_mpie(loader, gallery_loader, probe_loader)
    logger.print_log('  **Test**  ' + create_stat_string(meters_val))


if __name__ == '__main__':
    main()
