from models import model_utils
#默认psfcn
def buildModel(args):
    print('Creating Model %s' % (args.model))
    in_c = model_utils.getInputChanel(args)
    other = {'img_num': args.in_img_num, 'in_light': args.in_light}
    if args.model == 'GR_PSN':
        from models.GR_PSN import GeometryNet
        model = GeometryNet(args.fuse_type, args.use_BN, in_c, other)
    elif args.model == 'GR_PSN_run':
        from models.GR_PSN_run import GeometryNet
        model = GeometryNet(args.fuse_type, args.use_BN, in_c, other)
    else:
        raise Exception("=> Unknown Model '{}'".format(args.model))

    if args.in_mtrl:
        from models.GR_PSN import RenderingNet
        Rmodel=RenderingNet(args.fuse_type, args.use_BN, 3, other)

    if args.cuda: 
        model = model.cuda()
        Rmodel = Rmodel.cuda()

    if args.retrain: 
        print("=> using pre-trained model %s" % (args.retrain))
        model_utils.loadCheckpoint(args.retrain, model, cuda=args.cuda)
        print("=> using pre-trained Rendering model %s" % (args.retrain.replace("GNet", "RNet")))
        model_utils.loadCheckpoint(args.retrain.replace("GNet", "RNet"), Rmodel, cuda = args.cuda)
        return model, Rmodel

    if args.retrain_R:
        # print("=> using pre-trained model %s" % (args.retrain))
        # model_utils.loadCheckpoint(args.retrain, model, cuda=args.cuda)
        print("=> using pre-trained Rendering model %s" % (args.retrain_R.replace("GNet", "RNet")))
        model_utils.loadCheckpoint(args.retrain_R.replace("GNet", "RNet"), Rmodel, cuda = args.cuda)
        return model, Rmodel

    if args.resume:
        # print("=> Resume loading checkpoint %s" % (args.resume))
        # model_utils.loadCheckpoint(args.resume, model, cuda=args.cuda)
        print("=> using pre-trained Rendering model %s" % (args.resume.replace("GNet", "RNet")))
        model_utils.loadCheckpoint(args.resume.replace("GNet", "RNet"), Rmodel, cuda = args.cuda)
        return model, Rmodel
    print(model)
    print("=> Model Parameters: %d" % (model_utils.get_n_params(model)))

    if args.in_mtrl:
        return model, Rmodel
    else:
        return model
