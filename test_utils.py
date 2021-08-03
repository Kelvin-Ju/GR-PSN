import os
import torch
import torchvision.utils as vutils
import numpy as np
from models import model_utils
from utils import eval_utils, time_utils 

def get_itervals(args, split):
    args_var = vars(args)
    disp_intv = args_var[split+'_disp']
    save_intv = args_var[split+'_save']
    return disp_intv, save_intv


def test(args, split, loader, model,RNet, log, epoch, recorder,index=0):
    model.eval()
    print('---- Start %s Epoch %d: %d batches ----' % (split, epoch, len(loader)))
    timer = time_utils.Timer(args.time_sync);

    disp_intv, save_intv = get_itervals(args, split)
    with torch.no_grad():
        for i, sample in enumerate(loader):
            data = model_utils.parseData(args, sample, timer, split,index)
            input = model_utils.getInput(args, data)

            out_var = model(input); timer.updateTime('Forward')
            acc,error_map = eval_utils.calNormalAcc(data['tar'].data, out_var.data, data['m'].data) 
            recorder.updateIter(split, acc.keys(), acc.values())

            RNet_input = [out_var, data['l'], data['mtrl']]
            out_Rendering = RNet(RNet_input)
            masked_rendering = out_Rendering * data['m'].data.expand_as(out_Rendering.data)
            iters = i + 1

            if iters % disp_intv == 0:
                opt = {'split':split, 'epoch':epoch, 'iters':iters, 'batch':len(loader), 
                        'timer':timer, 'recorder': recorder}
                log.printItersSummary(opt)
                if args.benchmark=="Dragon":
                    mtrl_name=sample['obj'][0].split("/")[-2]
                    L1=torch.nn.L1Loss()(data["input"],masked_rendering).item()
                    print(f"L1 loss of {mtrl_name} is: {L1}")
                    with open("MAEandREL.csv","a+") as f:
                        print(",".join([mtrl_name,str(acc['n_err_mean']),str(L1)]),file = f)
                        print(",".join([mtrl_name,str(acc['n_err_mean']),str(L1)]))

            if iters % save_intv == 0:
                mtrl_name = sample['obj'][0]
                pred = (out_var.data + 1) / 2
                masked_pred = pred * data['m'].data.expand_as(out_var.data)
                # log.saveNormalResults(masked_pred, split, epoch, iters)
                # log.saveErrorMap(error_map ,split, epoch, iters)
                # log.saveRenderingResults(masked_rendering, split, epoch, iters)
                log.saveNormalResults(masked_pred, split, epoch, mtrl_name)
                log.saveErrorMap(error_map ,split, epoch,mtrl_name)
                log.saveRenderingResults(masked_rendering, split, epoch, mtrl_name)


    opt = {'split': split, 'epoch': epoch, 'recorder': recorder}
    log.printEpochSummary(opt)

