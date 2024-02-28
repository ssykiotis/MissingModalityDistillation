def main():

    args = get_args()
    num_cls      = args.num_cls
    num_channels = args.num_channels
    max_epoch    = args.max_epoch
    batch_size   = args.batch_size

    set_random(seed_id=args.seed)

    snapshot_path = args.log_dir
    create_if_not(snapshot_path)
    save_model_path = snapshot_path + '/model'
    create_if_not(save_model_path)
    create_if_not(save_model_path)
    
    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    
    #model
    model = VNet(n_channels = num_channels, n_classes = num_cls, n_filters = 16, normalization='batchnorm')
    model.train()
    model.cuda()
    start_epoch = 0
    best_epoch  = 0
    best_dice   = 0
    best_wt     = 0
    best_co     = 0
    best_ec     = 0

    if args.resume:
        print('load %s'%args.ckpt_path)
        ckpt = torch.load(args.ckpt_path)
        start_epoch = ckpt['epoch'] + 1
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        best_dice  = ckpt['best_dice']
        best_epoch = ckpt['best_epoch']
    #dataset
    train_dataset = FloodsDataset('train',num_channels,args.data_dir)
    print('Training set includes %d data.' % len(train_dataset))
    train_loader = DataLoader(dataset     = train_dataset,
                              batch_size  = batch_size,
                              shuffle     = True,
                              pin_memory  = True)

    val_image_paths = [f for f in os.listdir(args.data_dir) if 'val' in f] 

    print('Val set includes %d data.' % len(val_image_paths))
    # val_list = imglist
    #optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=1e-5)
    
    writer = SummaryWriter(snapshot_path+'/tensorboard')
    iter_num = 0
    loss_criterion = DiceCeLoss(num_cls)
    train_time1 = time.time()
    
    print('---Start training.')
    for epoch in range(start_epoch,max_epoch):
        loss_values = []
        time1 = time.time()
        #change lr
        curr_lr = args.lr * (1.0-np.float32(epoch)/np.float32(args.max_epoch))**(0.9)
        for parm in optimizer.param_groups:
            parm['lr'] = curr_lr
        for idx, sampled_batch in enumerate(train_loader):
            image,label = sampled_batch
            image,label = image.float().cuda(), label.float().cuda()
            
            features,logits = model(image)

            # loss = loss_criterion(logits,label)
            # loss_values.append(loss.item())

            # if loss<0:
            #     print('negative loss')
            
            dice_loss,ce_loss,loss = loss_criterion(logits,label)
            loss_values.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #record loss
            iter_num = iter_num + 1
            writer.add_scalar('loss/loss', loss, iter_num)
            # writer.add_scalar('loss/ce_loss',ce_loss,iter_num)
            # writer.add_scalar('loss/dice_loss',dice_loss,iter_num)
        epoch_loss = sum(loss_values)/len(loss_values)
        
        logging.info('Epoch:[%d/%d],iteration:%d, loss: %f' % (epoch, max_epoch, iter_num, epoch_loss))
            
        # logging.info('Epoch:[%d/%d],iteration:%d, loss: %f' % (epoch,max_epoch,iter_num, sum(loss_values)/len(loss_values)))
        time2 = time.time()
        logging.info('Epoch %d training time :%f minutes' % (epoch, (time2-time1)/60))
        #val
        if epoch%25 == 0:
            state = {
                'epoch':epoch,
                'state_dict':model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'best_dice':best_dice,
                'best_epoch':best_epoch
            }
            torch.save(state,save_model_path+'/checkpoint_{}.pth.tar'.format(epoch))
        #if epoch is smaller than 250, do not val, save time

        if epoch < max_epoch//4:
            continue

        model.eval()
        dice_all_wt = []
        dice_all_co = []
        dice_all_ec = []
        dice_all_mean = []
        print('---Start epoch %d validation' % epoch)
        val_dataset = FloodsDataset('val',num_channels,args.data_dir,min_ = train_dataset.x_min, max_ = train_dataset.x_max)
        val_loader  = DataLoader(dataset    = val_dataset,
                                batch_size  = batch_size,
                                shuffle     = True,
                                pin_memory  = True)

        time1 = time.time()
        dice_evaluator = Dice(num_classes= num_cls, average = 'macro').cuda()
        with torch.no_grad():
            for idx, sampled_batch in enumerate(val_loader):
                image,label = sampled_batch
                image,label = image.float().cuda(), label.float().cuda()
                        
                # predict,_ = test_single_case(model,image,STRIDE,CROP_SIZE,num_cls)
                _,predict = model(image)
                predict = predict.argmax(dim = 1)
                dice_evaluator.update(predict,label.squeeze().int())

                dice_mean = dice_evaluator.compute().item()


                # dice_wt,dice_co,dice_ec,dice_mean = eval_one_dice(predict,label)
                # dice_all_wt.append(dice_wt)
                # dice_all_co.append(dice_co)
                # dice_all_ec.append(dice_ec)
                # dice_all_mean.append(dice_mean)
                # logging.info('Sample [%d], average dice : %f' % (idx, dice_mean))
        time2 = time.time()
        logging.info('Epoch %d validation time : %f minutes' % (epoch, (time2-time1)/60))
        logging.info('Epoch %d validation Dice : %f ' % (epoch, dice_mean))
        # dice_all_wt = np.mean(np.array(dice_all_wt))
        # dice_all_co = np.mean(np.array(dice_all_co))
        # dice_all_ec = np.mean(np.array(dice_all_ec))
        # dice_all_mean = np.mean(np.array(dice_all_mean))
        # logging.info('epoch %d val dice, wt_dice:%f, co_dice:%f, ec_dice:%f'%(epoch,dice_all_wt,dice_all_co,dice_all_ec))
        # writer.add_scalar('val/dice_wt',   dice_all_wt,  epoch)
        # writer.add_scalar('val/dice_co',   dice_all_co,  epoch)
        # writer.add_scalar('val/dice_ec',   dice_all_ec,  epoch)
        writer.add_scalar('val/dice_mean', dice_mean,epoch)
        if dice_mean>=best_dice:
            best_epoch = epoch
            best_dice = dice_mean
            # best_wt = dice_all_wt
            # best_co = dice_all_co
            # best_ec = dice_all_ec
            torch.save(model.state_dict(), save_model_path+'/best_model.pth')
        model.train()
        logging.info('Best dice is: %f'%best_dice)
        logging.info('Best epoch is: %d'%best_epoch)
    writer.close()
    train_time2 = time.time()
    training_time = (train_time2 - train_time1) / 3600
    logging.info('Training finished, tensorboardX writer closed')
    logging.info('Best epoch is %d, best mean dice is %f'%(best_epoch,best_dice))
    logging.info('Dice of wt/co/ec is %f,%f,%f'%(best_wt,best_co,best_ec))
    logging.info('Training total time: %f hours.' % training_time)