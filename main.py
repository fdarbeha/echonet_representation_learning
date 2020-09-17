import os
import argparse
import tqdm
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data as data

from pytorch_metric_learning import losses
from torchvideotransforms import video_transforms, volume_transforms

import sklearn.metrics
import numpy as np
from torch.cuda.amp import GradScaler, autocast


from losses.NTXentLoss import NTXentLoss
from utils.helper_functions import get_next_model_folder, inspect_model, reshape_videos_cnn_input, reshape_videos_cnn_input_eval
from utils.helper_functions import get_image_patch_tensor_from_video_batch, write_csv_stats
from utils.helper_classes import AverageMeter, GaussianBlur
from data.echonet_dataset import get_train_and_test_echonet_datasets
from data.oasis3_dataset import get_oasis3_datasets, collate_fn, collate_fn_ssl
from models.models_3d import construct_3d_enc, construct_linear_regressor 
import models.cpc as cpc
import models.autoencoder as autoencoder

print(torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

s=1
video_transform_list = [video_transforms.RandomRotation(15),
                        video_transforms.RandomCrop((50, 50)),
                        video_transforms.Resize((112, 112)),
                        video_transforms.RandomHorizontalFlip(),
                        video_transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s),
                        volume_transforms.ClipToTensor(3, 3)]

data_augment = video_transforms.Compose(video_transform_list)


def dataloader(batch_size, mode, args):
    trainloader, valloader, testloader = None, None, None
    if args.dataset.lower() == 'echonet':
        ssl = False
        if args.eval == False and (mode == "ssl" or mode == "multi" or mode == "cpc"):
            ssl = True

        trainset, valset, testset = get_train_and_test_echonet_datasets("EF", frames=args.frame_num,\
         period=args.period, ssl=ssl)

        print('TRAIN DATASET: ', len(trainset), trainset[0][0].shape)
        print('TEST DATASET: ', len(testset), testset[0][0].shape)
        
        trainloader = data.DataLoader(trainset, batch_size=batch_size, \
                                    shuffle = True, num_workers=args.num_workers, drop_last=True)
        valloader  = data.DataLoader(valset, batch_size=batch_size, \
                                    shuffle = False, num_workers=args.num_workers, drop_last=True)
        testloader  = data.DataLoader(testset, batch_size=batch_size, \
                                    shuffle = False, num_workers=args.num_workers, drop_last=True)
    if args.dataset.lower() == 'oasis3':
        trainset, valset, testset = get_oasis3_datasets()

        print('TRAIN DATASET: ', len(trainset), trainset[0][0].shape)
        print('TEST DATASET: ', len(valset), valset[0][0].shape)
        
        if ssl == False:

            trainloader = data.DataLoader(trainset, batch_size=batch_size, \
                                        shuffle = True, num_workers=args.num_workers, drop_last=True,\
                                        collate_fn=collate_fn)
            valloader = data.DataLoader(valset, batch_size=batch_size, \
                                        shuffle = True, num_workers=args.num_workers, drop_last=True,\
                                        collate_fn=collate_fn)
            testloader  = data.DataLoader(testset, batch_size=batch_size, \
                                        shuffle = False, num_workers=args.num_workers, drop_last=True,\
                                        collate_fn=collate_fn)
        else:
            trainloader = data.DataLoader(trainset, batch_size=batch_size, \
                                        shuffle = True, num_workers=args.num_workers, drop_last=True,\
                                        collate_fn=collate_fn_ssl)

    return trainloader, valloader, testloader

def optimizer(net, args):
    assert args.optimizer.lower() in ["sgd", "adam", "radam"], "Invalid Optimizer"

    if args.optimizer.lower() == "sgd":
           return optim.SGD(net.parameters(), lr=args.lr, momentum=args.beta1, weight_decay=1e-4)
    elif args.optimizer.lower() == "adam":
           return optim.Adam(net.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    elif args.optimizer.lower() == "radam":
            return radam.RAdam(net.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))



def train_regressor(net, regressor, epoch, criterion, reg_optimizer, con_optimizer, trainloader, scaler):
    loss_meter = AverageMeter()
    running_loss = 0
    net.train()
    regressor.train()

    yhat = []
    y = []

    for (i, (inputs, labels)) in enumerate(tqdm.tqdm(trainloader)): 
        inputs = inputs.to(device)        
        labels = labels.to(device)

        reg_optimizer.zero_grad()
        if con_optimizer != None:
            con_optimizer.zero_grad()
        
        with torch.set_grad_enabled(True):
            outputs, _ = net(inputs)
            outputs = regressor(outputs)
            loss = criterion(outputs.view(-1).float(), labels.float())

            loss.backward()
            reg_optimizer.step()
            if con_optimizer != None:
                con_optimizer.step()

        loss_meter.update(loss.item())
        yhat.append(outputs.view(-1).to("cpu").detach().numpy())
        y.append(labels.to("cpu").numpy())


        running_loss += loss.item()
    try:
        auc = sklearn.metrics.roc_auc_score(y, yhat)
    except:
        print(yhat)

    return loss_meter.average(), auc

def eval_regressor(net, regressor, epoch, criterion, testloader):
    loss_meter = AverageMeter()
    running_loss = 0
    net.eval()

    yhat = []
    y = []

    for (i, (inputs, labels)) in enumerate(tqdm.tqdm(testloader)): 
        inputs = inputs.to(device)        
        labels = labels.to(device)

        
        with torch.set_grad_enabled(False):
            with autocast(False):
                outputs, _ = net(inputs)
                outputs = regressor(outputs)
                loss = criterion(outputs.view(-1).float(), labels.float())

        loss_meter.update(loss.item())

        yhat.append(outputs.view(-1).float().to("cpu").detach().numpy())
        y.append(labels.float().to("cpu").numpy())

        running_loss += loss.item()

    auc = sklearn.metrics.roc_auc_score(y, yhat)
    return loss_meter.average(), auc


def CPC_train(model, epoch, optimizer, trainloader, args, scaler):
    loss_meter = AverageMeter()

    model.train()
    for (i, (b1, *_)) in enumerate(tqdm.tqdm(trainloader)):


        optimizer.zero_grad()
        x_1 = reshape_videos_cnn_input(b1)
        with torch.set_grad_enabled(True):

            loss, _, _, _ = model(x_1.to(device))
            # print('loss: ', loss)
            loss = loss.mean() # accumulate losses for all GPUs
        
            loss_meter.update(loss.item())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    return loss_meter.average()

def AutoEncoder(net, dec, epoch, criterion, optim, trainloader, args, scaler):
    loss_meter = AverageMeter()
    running_loss = 0
    net.train()
    if dec != None:
        dec.train()


    for (i, (b1, *_)) in enumerate(tqdm.tqdm(trainloader)):

        optim.zero_grad()

        with autocast():
            out_1 = net(b1.to(device))
            out_1 = dec(out_1)
            

        loss = criterion(out_1.float(), b1.to(device).float())
        loss_meter.update(loss.item())

        scaler.scale(loss).backward()
    
        scaler.step(optim)
        scaler.update()
       
        running_loss += loss.item()


    return loss_meter.average()

def train_PaSTSSL_echonet(net, epoch, criterion_ssl, optimizer, trainloader, args,\
             scaler, regressor=None, criterion_sup=None):
    loss_meter = AverageMeter()
    running_loss = 0
    net.train()
    if regressor != None:
        regressor.train()

    yhat = []
    y = []

    
    if args.mode == "ssl":
        # for (i, (b1, b2, _)) in enumerate(trainloader):
        for (i, (b1, b2, b3, *_)) in enumerate(tqdm.tqdm(trainloader)):

            optimizer.zero_grad()
            x_1 = torch.zeros_like(b1).cuda()
            x_2 = torch.zeros_like(b2).cuda()
            x_3 = torch.zeros_like(b3).cuda()
            with autocast():
                for idx, (x1, x2, x3) in enumerate(zip(b1, b2, b3)):
                    x1 = get_image_patch_tensor_from_video_batch(x1)
                    x2 = get_image_patch_tensor_from_video_batch(x2)
                    x3 = get_image_patch_tensor_from_video_batch(x3)

                    x_1[idx] = data_augment(x1)
                    x_2[idx] = data_augment(x2)
                    x_3[idx] = data_augment(x3)

                _, out_1 = net(x_1.to(device))
                _, out_2 = net(x_2.to(device))
                _, out_3 = net(x_3.to(device))

                out_1 = F.normalize(out_1, dim=1)
                out_2 = F.normalize(out_2, dim=1)
                out_3 = F.normalize(out_3, dim=1)
                
                
            loss = criterion_ssl(out_1.float(), out_2.float(), out_3.float())
            loss_meter.update(loss.item())

            scaler.scale(loss).backward()
           
            scaler.step(optimizer)
            scaler.update()
           
            running_loss += loss.item()


    return loss_meter.average(), running_loss

def train_PaSTSSL_oasis(net, epoch, criterion, optimizer, trainloader, scaler):
    loss_meter = AverageMeter()
    net.train()

    for (i, (b1, b2, b3, *_)) in enumerate(tqdm.tqdm(trainloader)):

        optimizer.zero_grad()
        x_1 = torch.zeros_like(b1).cuda()
        x_2 = torch.zeros_like(b2).cuda()
        x_3 = torch.zeros_like(b3).cuda()
        with autocast():
            for idx, (x1, x2, x3) in enumerate(zip(b1, b2, b3)):
                x1 = get_image_patch_tensor_from_volume_batch(x1)
                x2 = get_image_patch_tensor_from_volume_batch(x2)
                x3 = get_image_patch_tensor_from_volume_batch(x3)

                x_1[idx] = data_augment(x1)
                x_2[idx] = data_augment(x2)
                x_3[idx] = data_augment(x3)

            out_1 = net(x_1.to(device))
            out_2 = net(x_2.to(device))
            out_3 = net(x_3.to(device))
            # print(out_1.shape)
            # print(out_2.shape)
            
            out_1 = F.normalize(out_1, dim=1)
            out_2 = F.normalize(out_2, dim=1)
            out_3 = F.normalize(out_3, dim=1)
            

        loss = criterion(out_1.float(), out_2.float(), out_3.float())
        loss_meter.update(loss.item())

        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()

    return loss_meter.average()

    def train_supervise_oasis(cnn, rnn, epoch, criterion, optimizer, trainloader, scaler):
    loss_meter = AverageMeter()
    running_loss = 0
    cnn.train()
    rnn.train()

    yhat = []
    y = []

    for (i, (inputs, labels, lengths)) in enumerate(tqdm.tqdm(trainloader)):
        if inputs == None:
            continue
      
        labels = labels.to(device)
        lengths = lengths.to(device)

        _, C, D, H, W = inputs.shape
        batch_size = lengths.shape[0]

        optimizer.zero_grad()

        inputs = inputs.to(device)
                 
        with torch.set_grad_enabled(True):

            outputs = cnn(inputs)
            outputs = reshape_rnn_input(outputs, lengths, batch_size)

            outputs = rnn(outputs, lengths)

            loss = criterion(outputs.view(-1).float(), labels.float())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        outputs = outputs.to("cpu")

        loss_meter.update(loss.item())
        
        yhat.append(outputs.view(-1).detach().numpy())
        y.append(labels.to("cpu").numpy())

        running_loss += loss.item()

    y = np.reshape(y, len(y) * batch_size)
    yhat = np.reshape(yhat, len(yhat) * batch_size)


    auc = sklearn.metrics.roc_auc_score(y, yhat)
    return loss_meter.average(), auc

def eval_supervise_oasis(cnn, rnn, epoch, criterion, testloader):
    loss_meter = AverageMeter()
    running_loss = 0
    cnn.eval()
    rnn.eval()

    yhat = []
    y = []

    for (i, (inputs, labels, lengths)) in enumerate(tqdm.tqdm(testloader)):
        if inputs == None:
            continue
      
        labels = labels.to(device)
        lengths = lengths.to(device)

        _, C, D, H, W = inputs.shape
        batch_size = lengths.shape[0]


        inputs = inputs.to(device)
    
        with torch.set_grad_enabled(False):

            outputs = cnn(inputs)
            outputs = reshape_rnn_input(outputs, lengths, batch_size)

            outputs = rnn(outputs, lengths)
            loss = criterion(outputs.view(-1).float(), labels.float())


        outputs = outputs.to("cpu")

        loss_meter.update(loss.item())
        
        yhat.append(outputs.view(-1).detach().numpy())
        y.append(labels.to("cpu").numpy())

        running_loss += loss.item()

    y = np.reshape(y, len(y) * batch_size)
    yhat = np.reshape(yhat, len(yhat) * batch_size)

    auc = sklearn.metrics.roc_auc_score(y, yhat)
    return loss_meter.average(), auc


def checkpoint(model, model_store_folder, epoch_num, model_name, period, frames,\
                bestLoss, loss, auc, optim, scheduler = None):
    print('Saving checkpoints...')
    save = {
                'epoch': epoch_num,
                'state_dict': model.state_dict(),
                'period': period,
                'frames': frames,
                'best_loss': bestLoss,
                'loss': loss,
                'opt_dict': optim.state_dict(),
                'scheduler_dict': scheduler.state_dict() if scheduler != None else None,
            }

    suffix_latest = '{}.pth'.format(model_name)
    torch.save(save,
               '{}/{}'.format(model_store_folder, suffix_latest))

def load_model(output, epoch, model, model_name, optim = None, scheduler = None, csv_path=None):
    
    checkpoint_name = "{}.pth".format(model_name)
    try:
        print("checkpoint: ", os.path.join(output, checkpoint_name))
        checkpoint = torch.load(os.path.join(output, checkpoint_name))
        try:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            if optim != None:
                optim.load_state_dict(checkpoint['opt_dict'])
            if scheduler != None:
                scheduler.load_state_dict(checkpoint['scheduler_dict'])
            epoch_resume = checkpoint["epoch"] + 1
            bestLoss = checkpoint["best_loss"]
        except:
            model.load_state_dict(checkpoint)
            epoch_resume = 0

        if csv_path != None:
            stats = dict(
                    epoch_resume = "Resuming from epoch {}\n".format(epoch_resume)
                )
            write_csv_stats(csv_path, stats)
        return epoch_resume
    
    except FileNotFoundError:
        print("No checkpoint found\n")
    
    except: # saved model in nn.DataParallel
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

    return 0


def extract_features(net, testloader):
    features = []
    y = []

    net.eval()

    for (i, (inputs, labels)) in enumerate(tqdm.tqdm(testloader)): 
        inputs = inputs.to(device)        
        
        with torch.set_grad_enabled(False):
            with autocast(False):
                outputs, _ = net(inputs.float())


        features.extend(outputs.float().to("cpu").detach().numpy())
        y.extend(labels.float().to("cpu").numpy())

    return features, y

def scale_range01(x):
    drange = (np.max(x) - np.min(x))
    if drange == 0:
        drange = 1
    x = x - np.min(x)
    x = x / drange

    return x

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # model related arguments
    parser.add_argument('--encoder_model', default='r3d_18')
    parser.add_argument('--encoder_pretrained', default=False)
    parser.add_argument('--mode', default='ssl')
    parser.add_argument('--type', default='ssl')
    parser.add_argument('--projection_size', default=128, type=int)
    parser.add_argument('--similarity', default='cosine')
    parser.add_argument('--run', default=None, type=int, help='epoch to use weights')
    parser.add_argument('--checkpoint', default=None, type=int, help='epoch to start training from')
    parser.add_argument('--frame_num', default=32, type=int)
    parser.add_argument('--period', default=2, type=int)
    # optimization related arguments
    parser.add_argument('--batch_size', default=40, type=int,
                        help='input batch size')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='num workers')
    parser.add_argument('--epoch', default=45, type=int,
                        help='epochs to train for')
    parser.add_argument('--dataset', default='echonet')
    parser.add_argument('--optimizer', default='sgd', help='optimizer')
    parser.add_argument('--lr', default=1e-4, type=float, help='LR')
    parser.add_argument('--beta1', default=0.9, type=float,
                        help='momentum for sgd, beta1 for adam')
    parser.add_argument('--beta2', default=0.999, type=float)
    parser.add_argument('--nesterov', default=False)
    parser.add_argument('--tau', default=0.1, type=float)
    # mode related arguments
    parser.add_argument('--eval', default=False)
    parser.add_argument('--output', default=None)

    args = parser.parse_args()

    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))
    
    if args.output is None:
        output_folder = "output"
    else:
        output_folder = args.output

    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    pretrain_str = 'pretrained' if args.encoder_pretrained else 'random'
    model_store_folder = get_next_model_folder(\
        '{}_{}'.format(args.type, pretrain_str,), \
                    output_folder, args.run)
    try:
        os.mkdir(model_store_folder)
    except FileExistsError:
        print("Output folder exits")

    pastssl_stats_csv_path = os.path.join(model_store_folder, "pastssl_pred_stats.csv")
    regressor_stats_csv_path = os.path.join(model_store_folder, "regressor_pred_stats.csv")

    trainloader, valloader, testloader = dataloader(args.batch_size, args.mode, args)
    

    if (args.eval == False):

        if args.mode == 'cpc':
            args.prediction_step = 3
            args.negative_samples = 32
            args.subsample = True
            genc_hidden = 512
            gar_hidden = 128
            args.device = device
            args.calc_accuracy = False

            model = cpc.CPC(args, model='resnet18', pretrained=args.encoder_pretrained,\
                    genc_hidden=genc_hidden, gar_hidden=gar_hidden).to(device)

            # inspect_model(model)

            if device.type == "cuda":
                model = torch.nn.DataParallel(model)

            optim_ssl = optimizer(model, args)

            print("\nStart training CPC!\n")
            scaler = GradScaler()
            bestLoss = float("inf")
            if args.dataset.lower() == 'echonet':
                for epoch in range(0, 200):

                    epoch_loss = CPC_train(model, epoch, optim_ssl, trainloader, args, scaler)
                    
                    print('epoch {} average loss : {}'.format(epoch, epoch_loss))
                    # Write stats into csv file
                    stats = dict(
                            epoch      = epoch,
                            epoch_loss = epoch_loss
                        )
                    write_csv_stats(pastssl_stats_csv_path, stats)

                    if epoch_loss < bestLoss:
                        checkpoint(model, model_store_folder, epoch, "best_cpc", args.period, args.frame_num,\
                                    bestLoss, epoch_loss, 0, optim_ssl, None)
                        bestLoss = epoch_loss
            
            print("CPC training completed!")

        if args.mode == "autoencoder":
            net = autoencoder.Encoder()
            dec = autoencoder.Decoder()

            if device.type == "cuda":
                net = torch.nn.DataParallel(net)
                dec = torch.nn.DataParallel(dec)
        
           
            criterion = nn.MSELoss()
            params = list(net.parameters()) + list(dec.parameters())
            optim = torch.optim.Adam(params, lr=args.lr,
                             weight_decay=1e-5)

            scheduler = torch.optim.lr_scheduler.StepLR(optim, math.inf)

            epoch_start = 0
            
            net = net.to(device)
            dec = dec.to(device)
            # inspect_model(net)
            # inspect_model(dec)


            print("\nStart training autoencoder!\n")
            bestLoss = float("inf")
            scaler = GradScaler()
            if args.dataset.lower() == 'echonet':
                for epoch in range(epoch_start, 200):
                    train_epoch_loss = AutoEncoder(net, dec, epoch, criterion,\
                                                        optim, trainloader, args, scaler)
                    
                    print('epoch {} average train loss : {}'.format(epoch, train_epoch_loss))
            
                    
                    if train_epoch_loss < bestLoss:
                        checkpoint(dec, model_store_folder, epoch, "best_dec", args.period, args.frame_num,\
                                    bestLoss, train_epoch_loss, 0, optim, scheduler)
                        checkpoint(net, model_store_folder, epoch, "best_net", args.period, args.frame_num,\
                                    bestLoss, train_epoch_loss, 0, optim, scheduler)
                        bestLoss = train_epoch_loss

                # Write stats into csv file
                    stats = dict(
                            epoch_reg      = epoch,
                            train_epoch_loss = train_epoch_loss,
                        )
                    write_csv_stats(regressor_stats_csv_path, stats)
            
            print("AutoEncoder training completed!")

        if args.mode == "pastssl":
            net = construct_3d_enc(args.encoder_model, args.encoder_pretrained, \
                args.projection_size, 'projection_head')


            if device.type == "cuda":
                net = torch.nn.DataParallel(net)
        
            criterion = NTXentLoss(device, args.batch_size , args.tau,\
             args.similarity, args.projection_size).to(device)

            optim_ssl = optimizer(net, args)

            scheduler = torch.optim.lr_scheduler.StepLR(optim_ssl, math.inf)

            epoch_start = 0
            
            net = net.to(device)
            # inspect_model(net)

            if args.checkpoint != None:
                epoch_start = load_model(model_store_folder, args.checkpoint, net,\
                                    "best_pastssl",optim=optim_ssl, \
                                    scheduler=None, csv_path=pastsslstats_csv_path)
                print("Starting from epoch: ", epoch_start)

            print("\nStart training PaSTSSL 3D!\n")


            scaler = GradScaler()
            bestLoss = float("inf")
            if args.dataset.lower() == 'echonet':
                for epoch in range(epoch_start, 200):
                    epoch_loss, running_loss = train_PaSTSSL_echonet(net, epoch, criterion, optim_ssl, trainloader, args, scaler)
                    
                    print('epoch {} average loss : {}'.format(epoch, epoch_loss))


                    # Write stats into csv file
                    stats = dict(
                            epoch      = epoch,
                            epoch_loss = epoch_loss
                        )
                    write_csv_stats(pastssl_stats_csv_path, stats)

                    if epoch_loss < bestLoss:
                        checkpoint(net, model_store_folder, epoch, "best_pastssl_3d", args.period, args.frame_num,\
                                    bestLoss, epoch_loss, 0, optim_ssl, scheduler)
                        bestLoss = epoch_loss

                    if epoch % 50 == 0:
                        checkpoint(net, model_store_folder, epoch, "best_pastssl_3d_{}".format(epoch), args.period, args.frame_num,\
                                    bestLoss, epoch_loss, 0, optim_ssl, scheduler)

            if args.dataset.lower() == 'oasis3':
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50)
                for epoch in range(1, 201):
                    train_loss = train_PaSTSSL_oasis(net, epoch, criterion, optimizer, trainloader, scaler)
                    print('epoch {} average train loss : {}'.format(epoch, train_loss))
                    scheduler.step()

                    # Write stats into csv file
                    stats = dict(
                            epoch      = epoch,
                            epoch_loss = train_loss,
                        )
                    write_csv_stats(simclr_stats_csv_path, stats)

                    if train_loss < bestLoss:
                        checkpoint(net, model_store_folder, epoch, "best_simclr", optimizer, scheduler)
                        # checkpoint(rnn, model_store_folder, epoch, "best_supervised_rnn", optimizer, scheduler)
                        bestLoss = train_loss

                    if epoch % 50 == 0:
                            checkpoint(net, model_store_folder, epoch, "best_simclr_{}".format(epoch), optimizer, scheduler)

            
            print("PaSTSSL training completed!")
    
    else:
        if args.mode == 'fine-tune':
            net = construct_3d_enc(args.encoder_model, args.encoder_pretrained, \
                args.projection_size, 'representation')
            # inspect_model(net)

            if device.type == "cuda":
                net = torch.nn.DataParallel(net)
            
            net = net.to(device)
            scaler = GradScaler()
            
            criterion = torch.nn.BCELoss()
            # scheduler = torch.optim.lr_scheduler.StepLR(reg_optimizer, math.inf)
            epoch_start = 0
            bestLoss = float("inf")
            load_model(model_store_folder, args.checkpoint, net, "best_pastssl_3d_200", optim=con_optimizer, csv_path = regressor_stats_csv_path)


            print("\nStart Finetuning!\n")
            
            if args.dataset.lower() == 'echonet':
                regressor = construct_linear_regressor(net, args.projection_size)
                # inspect_model(regressor)
                if device.type == "cuda":
                    regressor = torch.nn.DataParallel(regressor)
                regressor = regressor.to(device)

                reg_optimizer = optimizer(regressor, args)
                con_optimizer = optimizer(net, args)

                
                for epoch in range(epoch_start, 45):
                    train_epoch_loss, train_auc = train_regressor(net, regressor, epoch, \
                        criterion, reg_optimizer, con_optimizer, trainloader, scaler)
                    print('epoch {} average train loss : {}, auc: {}'.format(epoch, train_epoch_loss, train_auc))

                    eval_epoch_loss, eval_auc = eval_regressor(net, regressor, epoch, \
                        criterion, valloader)

                    print('epoch {} average eval loss : {}, auc: {}'.format(epoch, eval_epoch_loss, eval_auc))
                    

                    if eval_epoch_loss < bestLoss:
                        checkpoint(regressor, model_store_folder, epoch, "best_regressor", args.period, args.frame_num,\
                                    bestLoss, eval_epoch_loss, eval_auc, reg_optimizer)
                        if con_optimizer != None:
                            checkpoint(net, model_store_folder, epoch, "best_net", args.period, args.frame_num,\
                                        bestLoss, eval_epoch_loss, eval_auc, con_optimizer)
                        bestLoss = eval_epoch_loss

                # Write stats into csv file
                    stats = dict(
                            epoch_reg      = epoch,
                            train_epoch_loss = train_epoch_loss,
                            train_auc = train_auc,
                            eval_epoch_loss = eval_epoch_loss,
                            eval_auc = eval_auc
                        )
                    write_csv_stats(regressor_stats_csv_path, stats)
            
            if args.dataset.lower() == 'oasis3':
                rnn = construct_rnn()
                if device.type == "cuda":
                    rnn = torch.nn.DataParallel(rnn)

                rnn = rnn.to(device)

                params = list(net.parameters()) + list(rnn.parameters())
                optimizer = optim.Adam(params, lr=args.lr, betas=(args.beta1, args.beta2))

                for epoch in range(1, 46):
                    train_loss, train_auc = train_supervise_oasis(net, rnn, epoch, criterion, optimizer, trainloader, scaler)
                    print('epoch {} average train loss : {}, auc: {}'.format(epoch, train_loss, train_auc))
                    scheduler.step()
                    
                    eval_loss, eval_auc = eval_supervise_oasis(net, rnn, epoch, criterion, valloader)
                    print('epoch {} average eval loss : {}, auc: {}'.format(epoch, eval_loss, eval_auc))
                    

                    # Write stats into csv file
                    stats = dict(
                            epoch      = epoch,
                            epoch_loss = train_loss,
                            train_auc = train_auc,
                            eval_loss = eval_loss,
                            eval_auc = eval_auc
                        )
                    write_csv_stats(regressor_stats_csv_path, stats)

                    if eval_loss < bestLoss:
                        checkpoint(net, model_store_folder, epoch, "best_supervised_cnn", optimizer, scheduler)
                        checkpoint(rnn, model_store_folder, epoch, "best_supervised_rnn", optimizer, scheduler)
                        bestLoss = eval_loss
                
                print("supervised training completed! Best loss: {}".format(bestLoss))

                load_model(model_store_folder, args.checkpoint, net, "best_supervised_cnn" ,csv_path = regressor_stats_csv_path)
                load_model(model_store_folder, args.checkpoint, rnn, "best_supervised_rnn" ,csv_path = regressor_stats_csv_path)

                test_loss, test_auc = eval_supervise_oasis(net, rnn, 0, criterion, testloader)
                print('Test loss : {}, auc: {}'.format(test_loss, test_auc))

                # Write stats into csv file
                stats = dict(
                        test_loss = test_loss,
                        test_auc = test_auc
                    )
                write_csv_stats(regressor_stats_csv_path, stats)


            
            print("Finetuning completed!")

            load_model(model_store_folder, args.checkpoint, regressor, "best_regressor",csv_path = regressor_stats_csv_path)
            load_model(model_store_folder, args.checkpoint, net, "best_net",csv_path = regressor_stats_csv_path)

            test_loss, test_auc = eval_regressor(net, regressor, epoch, \
                    criterion, testloader)

            stats = dict(
                        test_loss = test_loss,
                        test_auc = test_auc
                    )
            write_csv_stats(regressor_stats_csv_path, stats)

        elif args.mode == 'tsne':
            net = construct_3d_enc(args.encoder_model, args.encoder_pretrained, \
                args.projection_size, 'representation')
            if device.type == "cuda":
                net = torch.nn.DataParallel(net)
            net = net.to(device)


            load_model(model_store_folder, args.checkpoint, net, "best_net_3d_200",csv_path = regressor_stats_csv_path)
            features, y = extract_features(net, testloader)

            print(len(features), " ", features[0].shape)
            print(len(y), " ", y[0])

            import sklearn
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2, perplexity=50).fit_transform(features)

            tx = tsne[:, 0]
            ty = tsne[:, 1]

            tx = scale_range01(tx)
            ty = scale_range01(ty)

            # print(tx)
            # print(ty)

            target_ids = [0.0, 1.0]
            target_names = ['healthy', 'not-healthy']
            colors = ['r', 'g']

            from matplotlib import pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(111)


            for label, c, label_name in zip(target_ids, colors, target_names):
                indecis = [i for i, j in enumerate(y) if j == label]
                # print(indecis)

                current_tx = tx[indecis]
                current_ty = ty[indecis]

                # print(current_tx[0:5])

                ax.scatter(current_tx, current_ty, c=c, label=label_name)


            ax.legend(loc='best')
            plt.show()
            plt.savefig(os.path.join(model_store_folder, 'tsne.png'))






        



