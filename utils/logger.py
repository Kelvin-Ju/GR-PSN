import datetime, time
import os
import numpy as np
import torch
import torchvision.utils as vutils
from . import utils
from utils import eval_utils, time_utils 


class Logger(object):
    def __init__(self, args):
        self.times = {'init': time.time()}
        self._checkPath(args)
        self.args = args
        self.printArgs()

    def printArgs(self):
        strs = '------------ Options -------------\n'
        strs += '{}'.format(utils.dictToString(vars(self.args)))
        strs += '-------------- End ----------------\n'
        print(strs)

    def _checkPath(self, args): 
        if hasattr(args, 'run_model') and args.run_model:
            print(args.retrain)
            if args.benchmark=="Dragon":
                log_root = os.path.join('run_model_Dragon', f"mtrl_{args.mtrl_type}")
            else:
                # log_root = os.path.join('run_model',f"mtrl_{args.mtrl_type}")
                log_root = os.path.join(args.bm_dir,f'mtrl_{args.mtrl_type}')
            utils.makeFiles([os.path.join(log_root)])
        else:
            if args.resume and os.path.isfile(args.resume):
                log_root = os.path.join(os.path.dirname(os.path.dirname(args.resume)), 'resume')
            else:
                log_root = os.path.join(args.save_root, args.item)
            for sub_dir in ['train', 'val']: 
                utils.makeFiles([os.path.join(log_root, sub_dir)])
            args.cp_dir  = os.path.join(log_root, 'train')
        args.log_dir = log_root

    def getTimeInfo(self, epoch, iters, batch):
        time_elapsed = (time.time() - self.times['init']) / 3600.0
        total_iters  = (self.args.epochs - self.args.start_epoch + 1) * batch
        cur_iters    = (epoch - self.args.start_epoch) * batch + iters
        time_total   = time_elapsed * (float(total_iters) / cur_iters)
        return time_elapsed, time_total

    def printItersSummary(self, opt):
        epoch, iters, batch = opt['epoch'], opt['iters'], opt['batch']
        strs = ' | {}'.format(str.upper(opt['split']))
        strs += ' Iter [{}/{}] Epoch [{}/{}]'.format(iters, batch, epoch, self.args.epochs)
        if opt['split'] == 'train': 
            time_elapsed, time_total = self.getTimeInfo(epoch, iters, batch) 
            strs += ' Clock [{:.2f}h/{:.2f}h]'.format(time_elapsed, time_total)
            strs += ' LR [{}]'.format(opt['recorder'].records[opt['split']]['lr'][epoch][0])
        print(strs) 
        if 'timer' in opt.keys(): 
            print(opt['timer'].timeToString())
        if 'recorder' in opt.keys(): 
            print(opt['recorder'].iterRecToString(opt['split'], epoch))

    def printEpochSummary(self, opt):
        split = opt['split']
        epoch = opt['epoch']
        print('---------- {} Epoch {} Summary -----------'.format(str.upper(split), epoch))
        print(opt['recorder'].epochRecToString(split, epoch))

    def saveNormalResults(self, results, split, epoch, iters) :

        save_dir = os.path.join(self.args.log_dir, iters)
        os.makedirs(save_dir, exist_ok=True)
        save_name = 'Normal.png'
        print(save_dir)
        vutils.save_image(results, os.path.join(save_dir, save_name))

    def saveErrorMap(self, error_map, split, epoch, iters) :
        save_dir = os.path.join(self.args.log_dir, iters)
        os.makedirs(save_dir, exist_ok = True)
        save_name = 'errormap.png'
        vutils.save_image(error_map, os.path.join(save_dir, save_name))

    def saveRenderingResults(self, results, split, epoch, iters) :

        img_split = torch.split(results, 3, 1)
        print(img_split[0].shape)
        save_dir = os.path.join(self.args.log_dir, iters)
        os.makedirs(save_dir, exist_ok = True)

        for i, img in enumerate(img_split) :
            # M = 255. / img.mean()
            # if M>1:img =img*M
            save_name = '%03d.png' % (i+1)
            vutils.save_image(img, os.path.join(save_dir, save_name))
    # def saveNormalResults(self, results, split, epoch, iters):
    #     save_dir = os.path.join(self.args.log_dir, split)
    #     save_name = 'Normal%d_%d.png' % (epoch, iters)
    #     vutils.save_image(results, os.path.join(save_dir, save_name))
    #
    # def saveErrorMap(self,error_map,split,epoch,iters):
    #     save_dir = os.path.join(self.args.log_dir, split)
    #     save_name = 'errormap%d_%d.png' % (epoch, iters)
    #     vutils.save_image(error_map, os.path.join(save_dir, save_name))
    #
    # def saveRenderingResults(self,results,split,epoch,iters):
    #
    #     img_split = torch.split(results, 3, 1)
    #     print(img_split[0].shape)
    #     save_dir = os.path.join(self.args.log_dir, split)
    #
    #     for i,img in enumerate(img_split):
    #         # M = 255. / img.mean()
    #         # if M>1:img =img*M
    #         save_name = 'Rendering%d_%d_%d.png' % (epoch, iters,i)
    #         vutils.save_image(img, os.path.join(save_dir, save_name))