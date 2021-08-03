import torch
import os
import torch_optimizer as optim

class Criterion(object):
    def __init__(self, args):
        self.setupNormalCrit(args)

    def setupNormalCrit(self, args):
        #默认cos
        print('=> Using {} for criterion normal'.format(args.normal_loss))
        self.normal_loss = args.normal_loss
        self.normal_w    = args.normal_w
        if args.normal_loss == 'mse':
            self.n_crit = torch.nn.MSELoss()
        elif args.normal_loss == 'cos':
            self.n_crit = torch.nn.CosineEmbeddingLoss()
        else:
            raise Exception("=> Unknown Criterion '{}'".format(args.normal_loss))
        if args.cuda:
            self.n_crit = self.n_crit.cuda()

    def forward(self, output, target):
        dim2=output.shape[1]
        if self.normal_loss == 'cos':
            num = target.nelement() / target.shape[1]
            num=int(num)
            if not hasattr(self, 'flag') or num != self.flag.nelement():
                self.flag = torch.autograd.Variable(target.data.new().resize_(num).fill_(1))

            self.out_reshape = output.permute(0, 2, 3, 1).contiguous().view(-1, dim2)
            self.gt_reshape  = target.permute(0, 2, 3, 1).contiguous().view(-1, dim2)
            # print(self.out_reshape.shape, self.gt_reshape.shape, self.flag.shape)
            self.loss        = self.n_crit(self.out_reshape, self.gt_reshape, self.flag)


        elif self.normal_loss == 'mse':
            # print(output, target)
            self.loss = self.normal_w * self.n_crit(output, target)

        # print(self.loss)
        out_loss = {'N_loss': self.loss.item()}
        return out_loss

    def backward(self,retain_graph=False):
        self.loss.backward(retain_graph=retain_graph)

def getOptimizer(args, params):
    print('=> Using %s solver for optimization' % (args.solver))
    if args.solver == 'adam':
        optimizer = torch.optim.Adam(params, args.init_lr, betas=(args.beta_1, args.beta_2))
    elif args.solver == 'sgd':
        optimizer = torch.optim.SGD(params, args.init_lr, momentum=args.momentum)
    elif args.solver == 'RAdam':
        optimizer = optim.RAdam(params,args.init_lr, betas=(args.beta_1, args.beta_2))
    else:
        raise Exception("=> Unknown Optimizer %s" % (args.solver))
    return optimizer

def getLrScheduler(args, optimizer):
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
            milestones=args.milestones, gamma=args.lr_decay)
    scheduler.last_epoch = -1
    # scheduler.last_epoch =args.start_epoch-2
    return scheduler

def loadRecords(path, model, optimizer):
    records = None
    if os.path.isfile(path):
        records = torch.load(path[:-8] + '_rec' + path[-8:])
        optimizer.load_state_dict(records['optimizer'])
        start_epoch = records['epoch'] + 1
        records = records['records']
        print("=> loaded Records")
    else:
        raise Exception("=> no checkpoint found at '{}'".format(path))
    return records, start_epoch

def configOptimizer(args, model):
    records = None
    optimizer = getOptimizer(args, model.parameters())
    if args.resume:
        print("=> Resume loading checkpoint '{}'".format(args.resume))
        records, start_epoch = loadRecords(args.resume, model, optimizer)
        args.start_epoch = start_epoch
    scheduler = getLrScheduler(args, optimizer)
    return optimizer, scheduler, records
