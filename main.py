import torch
from options  import train_opts
from utils    import logger, recorders
from datasets import custom_data_loader
from models   import custom_model, solver_utils, model_utils

import train_utils
import test_utils
import os
os.environ['CUDA_VISIBLE_DEVICES']='2'
args = train_opts.TrainOpts().parse()
log  = logger.Logger(args)

def main(args):
    train_loader, val_loader = custom_data_loader.customDataloader(args)

    GNet,RNet = custom_model.buildModel(args)
    G_optimizer, G_scheduler, G_records = solver_utils.configOptimizer(args, GNet)
    R_optimizer, R_scheduler, R_records = solver_utils.configOptimizer(args, RNet)

    G_criterion = solver_utils.Criterion(args)

    R_criterion = solver_utils.Criterion(args)
    
    R_criterion.normal_w=args.normal_w
    R_criterion.normal_loss = 'mse'
    R_criterion.n_crit = torch.nn.L1Loss()
    R_criterion.n_crit = R_criterion.n_crit.cuda()
    recorder = recorders.Records(args.log_dir, G_records)

    for epoch in range(1, args.epochs+1):
        recorder.insertRecord('train', 'lr', epoch, G_scheduler.get_last_lr()[0])
        G_scheduler.step()
        R_scheduler.step()
        print(G_scheduler.get_last_lr()[0])
        print(R_scheduler.get_last_lr()[0])
        if epoch<args.start_epoch:continue
        train_utils.train(args, train_loader, GNet,RNet, G_criterion,R_criterion, G_optimizer,R_optimizer, log, epoch, recorder)



        if epoch % args.save_intv == 0:
            model_utils.saveCheckpoint(args.cp_dir, epoch, GNet, G_optimizer, recorder.records, args,"GNet")
            model_utils.saveCheckpoint(args.cp_dir, epoch, RNet, R_optimizer, recorder.records, args,"RNet")

        if epoch % args.val_intv == 0:test_utils.test(args, 'val', val_loader, GNet,RNet, log, epoch, recorder)


if __name__ == '__main__':
    torch.manual_seed(args.seed)
    main(args)
