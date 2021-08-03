from models import model_utils
from utils  import time_utils 

def train(args, loader,GNet,RNet, G_criterion,R_criterion, G_optimizer,R_optimizer, log, epoch, recorder):
    GNet.train()
    RNet.train()
    print('---- Start Training Epoch %d: %d batches ----' % (epoch, len(loader)))
    timer = time_utils.Timer(args.time_sync);

    for i, sample in enumerate(loader):
        dataA = model_utils.parseData(args, sample, timer, 'train', index = 0)
        dataB = model_utils.parseData(args, sample, timer, 'train', index = 1)

        inputA = model_utils.getInput(args, dataA)
        inputB = model_utils.getInput(args, dataB)

        G_optimizer.zero_grad()
        out_var_A = GNet(inputA)
        loss_N_A = G_criterion.forward(out_var_A, dataA['tar'])
        G_criterion.backward()
        G_optimizer.step()

        G_optimizer.zero_grad()
        out_var_B = GNet(inputB)
        loss_N_B = G_criterion.forward(out_var_B, dataB['tar'])
        G_criterion.backward()
        G_optimizer.step()

        lossG = (loss_N_A['N_loss'] + loss_N_B['N_loss'])/2




        if epoch<0:
            R_optimizer.zero_grad()
            # G_optimizer.zero_grad()
            # out_var_A = GNet(inputA)
            A2A = [dataA['tar'], dataA['l'], dataA['mtrl']]
            out_A2A = RNet(A2A)
            # print(out_A2A.shape)
            # print(dataA['input'].shape)
            loss_A2A = R_criterion.forward(out_A2A, dataA['input'])
            R_criterion.backward()
            R_optimizer.step()
            # G_optimizer.step()

            R_optimizer.zero_grad()
            # G_optimizer.zero_grad()
            # out_var_B = GNet(inputB)
            B2B = [dataB['tar'], dataB['l'], dataB['mtrl']]
            out_B2B = RNet(B2B)
            loss_B2B = R_criterion.forward(out_B2B, dataB['input'])
            R_criterion.backward()
            R_optimizer.step()
            # G_optimizer.step()

            lossR = (loss_A2A['N_loss'] + loss_B2B['N_loss'] )/2


        else:
            R_optimizer.zero_grad()
            G_optimizer.zero_grad()
            out_var_A = GNet(inputA)
            A2A = [out_var_A, dataA['l'], dataA['mtrl']]
            out_A2A = RNet(A2A)
            loss_A2A = R_criterion.forward(out_A2A, dataA['input'])
            R_criterion.backward()
            R_optimizer.step()
            G_optimizer.step()

            R_optimizer.zero_grad()
            G_optimizer.zero_grad()
            out_var_A = GNet(inputA)
            A2B = [out_var_A, dataB['l'], dataB['mtrl']]
            out_A2B = RNet(A2B)
            loss_A2B = R_criterion.forward(out_A2B, dataB['input'])
            R_criterion.backward()
            R_optimizer.step()
            G_optimizer.step()

            R_optimizer.zero_grad()
            # G_optimizer.zero_grad()
            # out_var_A = GNet(inputA)
            A2A = [dataA['tar'], dataA['l'], dataA['mtrl']]
            out_A2A = RNet(A2A)
            loss_A2A = R_criterion.forward(out_A2A, dataA['input'])
            R_criterion.backward()
            R_optimizer.step()
            # G_optimizer.step()

            R_optimizer.zero_grad()
            G_optimizer.zero_grad()
            out_var_B = GNet(inputB)
            B2B = [out_var_B, dataB['l'], dataB['mtrl']]
            out_B2B = RNet(B2B)
            loss_B2B = R_criterion.forward(out_B2B, dataB['input'])
            R_criterion.backward()
            R_optimizer.step()
            G_optimizer.step()

            R_optimizer.zero_grad()
            G_optimizer.zero_grad()
            out_var_B = GNet(inputB)
            B2A = [out_var_B, dataA['l'], dataA['mtrl']]
            out_B2A = RNet(B2A)
            loss_B2A = R_criterion.forward(out_B2A, dataA['input'])
            R_criterion.backward()
            R_optimizer.step()
            G_optimizer.step()

            R_optimizer.zero_grad()
            # G_optimizer.zero_grad()
            # out_var_B = GNet(inputB)
            B2B = [dataB['tar'], dataB['l'], dataB['mtrl']]
            out_B2B = RNet(B2B)
            loss_B2B = R_criterion.forward(out_B2B, dataB['input'])
            R_criterion.backward()
            R_optimizer.step()
            # G_optimizer.step()

            lossR = (loss_A2A['N_loss'] + loss_A2B['N_loss'] + loss_B2B['N_loss'] + loss_B2A['N_loss'])/4

        print(lossG, lossR)
        # print(lossG)
        recorder.updateIter('train', ['G_loss'], [lossG])
        recorder.updateIter('train', ['R_loss'], [lossR])
        # dataA  = model_utils.parseData(args, sample, timer, 'train',index =0)
        # dataB = model_utils.parseData(args, sample, timer, 'train', index =1)
        #
        # inputA = model_utils.getInput(args, dataA)
        # inputB = model_utils.getInput(args, dataB)
        # out_var_A = GeometryNet(inputA)
        # # timer.updateTime('Forward')
        # out_var_B = GeometryNet(inputB)
        #
        # G_optimizer.zero_grad()
        #
        # loss = criterion.forward(out_var, data['tar'])
        #
        # # timer.updateTime('Crit');
        # criterion.backward(); timer.updateTime('Backward')
        #
        # recorder.updateIter('train', loss.keys(), loss.values())
        #
        # optimizer.step();
        timer.updateTime('Solver')

        iters = i + 1
        if iters % args.train_disp == 0:
            opt = {'split':'train', 'epoch':epoch, 'iters':iters, 'batch':len(loader), 
                    'timer':timer, 'recorder': recorder}
            log.printItersSummary(opt)

    opt = {'split': 'train', 'epoch': epoch, 'recorder': recorder}
    log.printEpochSummary(opt)
