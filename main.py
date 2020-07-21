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
# import torch.cuda
from torch.cuda.amp import GradScaler, autocast


# from models.models import ResNet18, ConvLSTM, ClassifierModule, ClassifierModuleDense
# from utils import utils 
# from lib import radam
from loss import NTXentLoss
from utils.helper_functions import get_next_model_folder, inspect_model, reshape_videos_cnn_input, reshape_videos_cnn_input_eval
from utils.helper_functions import get_image_patch_tensor_from_video_batch, write_csv_stats
from utils.helper_classes import AverageMeter, GaussianBlur
from data.echonet_dataset import get_train_and_test_echonet_datasets
from models.models_3d import construct_3d_enc, construct_linear_regressor 
import models.cpc as cpc
import models.models_2d as models_2d

print(torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

s=1
# video_transform_list = [video_transforms.RandomRotation(15),
#                         video_transforms.RandomCrop((50, 50)),
#                         video_transforms.Resize((112, 112)),
#                         video_transforms.RandomHorizontalFlip(),
#                         video_transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s),
#                         volume_transforms.ClipToTensor(3, 3)]

# data_augment = video_transforms.Compose(video_transform_list)

color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
data_augment = transforms.Compose([transforms.ToPILImage(),
                                   transforms.RandomResizedCrop(50),
                                   transforms.Resize((112, 112)),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.RandomApply([color_jitter], p=0.8),
                                   transforms.RandomGrayscale(p=0.2),
                                   GaussianBlur(),
                                   transforms.ToTensor()])

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
                                    shuffle = False, num_workers=5, drop_last=True)
        testloader  = data.DataLoader(testset, batch_size=batch_size, \
                                    shuffle = False, num_workers=5, drop_last=True)

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

    # for (i, (inputs, _, labels)) in enumerate(trainloader):
    for (i, (inputs, labels)) in enumerate(tqdm.tqdm(trainloader)): 
        inputs = inputs.to(device)        
        labels = labels.to(device)

        reg_optimizer.zero_grad()
        if con_optimizer != None:
            con_optimizer.zero_grad()
        
        with torch.set_grad_enabled(True):
            # with autocast(False):
            outputs, _ = net(inputs)
            outputs = regressor(outputs)
            loss = criterion(outputs.view(-1).float(), labels.float())

            
            # scaler.scale(loss).backward()
            # scaler.step(reg_optimizer)
            # scaler.update()
            # if con_optimizer != None:
            #     scaler.step(con_optimizer)
            # scaler.update()
            loss.backward()
            reg_optimizer.step()
            if con_optimizer != None:
                con_optimizer.step()

        loss_meter.update(loss.item())
        # print(outputs.view(-1).to("cpu").detach().numpy())
        # print(labels.to("cpu").numpy())
        yhat.append(outputs.view(-1).to("cpu").detach().numpy())
        y.append(labels.to("cpu").numpy())

        # print("train: ", loss.item())
        running_loss += loss.item()

    # r2 = sklearn.metrics.r2_score(yhat, y)
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

    # for (i, (inputs, _, labels)) in enumerate(testloader):
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

        # print("eval: ", loss.item())
        running_loss += loss.item()

    # yhat = np.stack(yhat, axis=0)
    # y = np.stack(y, axis=0)
    # print(yhat)
    # print()
    # print(y)

    # r2 = sklearn.metrics.r2_score(yhat, y)
    auc = sklearn.metrics.roc_auc_score(y, yhat)
    return loss_meter.average(), auc


def eval_multi(net, regressor, epoch, criterion, testloader):
    loss_meter = AverageMeter()
    running_loss = 0
    net.eval()

    yhat = []
    y = []

    # for (i, (inputs, labels)) in enumerate(testloader):
    for (i, (inputs, inputs2, labels)) in enumerate(tqdm.tqdm(testloader)): 
        inputs = inputs.to(device)        
        labels = labels.to(device)

        
        with torch.set_grad_enabled(False):
            with autocast():
                outputs, _ = net(inputs)
                outputs = regressor(outputs)
                # print(outputs.view(-1))
                # print(labels)
                loss = criterion(outputs.view(-1).float(), labels.float())

        loss_meter.update(loss.item())

        yhat.append(outputs.view(-1).float().to("cpu").detach().numpy())
        y.append(labels.float().to("cpu").numpy())

        # print("eval: ", loss.item())
        running_loss += loss.item()

    # yhat = np.stack(yhat, axis=0)
    # y = np.stack(y, axis=0)
    # print(yhat)
    # print()
    # print(y)

    r2 = sklearn.metrics.r2_score(yhat, y)
    return loss_meter.average(), r2


def CPC_train(model, epoch, optimizer, trainloader, args, scaler):
    loss_meter = AverageMeter()

    model.train()
    for (i, (b1, b2, *_)) in enumerate(tqdm.tqdm(trainloader)):


        optimizer.zero_grad()
        x_1 = reshape_videos_cnn_input(b1)#torch.zeros_like(b1).cuda()
        x_2 = b2#torch.zeros_like(b2).cuda()
        with torch.set_grad_enabled(True):
            # for idx, (x1, x2) in enumerate(zip(b1, b2)):
            #     x1 = get_image_patch_tensor_from_video_batch(x1)
            #     x2 = get_image_patch_tensor_from_video_batch(x2)

            #     x_1[idx] = data_augment(x1)
            #     x_2[idx] = data_augment(x2)

            loss, _, _, _ = model(x_1.to(device))
            # print('loss: ', loss)
            loss = loss.mean() # accumulate losses for all GPUs
        
            loss_meter.update(loss.item())

            scaler.scale(loss).backward()
            # loss.backward()
            scaler.step(optimizer)
            scaler.update()

    return loss_meter.average()

def train_2d_net(net, epoch, criterion, optimizer, trainloader, scaler):
    loss_meter = AverageMeter()
    net.train()

    yhat = []
    y = []

    # for (i, (inputs, _, labels)) in enumerate(trainloader):
    for (i, (inputs, labels)) in enumerate(tqdm.tqdm(trainloader)): 
        inputs = inputs.to(device)        
        labels = labels.to(device)

        optimizer.zero_grad()
        inputs = reshape_videos_cnn_input_eval(inputs)
        
        with torch.set_grad_enabled(True):
            # with autocast(False):
            outputs = net(inputs)
            loss = criterion(outputs.view(-1).float(), labels.float())

            loss.backward()
            optimizer.step()

        loss_meter.update(loss.item())
        # print(outputs.view(-1).to("cpu").detach().numpy())
        # print(labels.to("cpu").numpy())
        yhat.append(outputs.view(-1).to("cpu").detach().numpy())
        y.append(labels.to("cpu").numpy())

        # print("train: ", loss.item())

    try:
        auc = sklearn.metrics.roc_auc_score(y, yhat)
    except:
        print(yhat)

    return loss_meter.average(), auc
 
def eval_2d_net(net, epoch, criterion, testloader):
    loss_meter = AverageMeter()
    running_loss = 0
    net.eval()

    yhat = []
    y = []

    # for (i, (inputs, _, labels)) in enumerate(testloader):
    for (i, (inputs, labels)) in enumerate(tqdm.tqdm(testloader)): 
        inputs = inputs.to(device)        
        labels = labels.to(device)
        inputs = reshape_videos_cnn_input_eval(inputs)

        
        with torch.set_grad_enabled(False):
            with autocast(False):
                outputs = net(inputs)
                loss = criterion(outputs.view(-1).float(), labels.float())

        loss_meter.update(loss.item())

        yhat.append(outputs.view(-1).float().to("cpu").detach().numpy())
        y.append(labels.float().to("cpu").numpy())

        # print("eval: ", loss.item())

    auc = sklearn.metrics.roc_auc_score(y, yhat)
    return loss_meter.average(), auc


def Simclr_2d(net, epoch, criterion, optimizer, trainloader, scaler):
    loss_meter = AverageMeter()
    net.train()

    # for (i, (inputs, _, labels)) in enumerate(trainloader):
    for (i, (inputs1, inputs2, inputs3, labels)) in enumerate(tqdm.tqdm(trainloader)): 
        inputs1 = inputs1.to(device)
        inputs2 = inputs2.to(device)         
        labels = labels.to(device)


        optimizer.zero_grad()
        inputs1 = inputs1.squeeze()#reshape_videos_cnn_input(inputs1)
        inputs2 = inputs2.squeeze()#reshape_videos_cnn_input(inputs2)
        # inputs3 = inputs3.squeeze()

        
        # x_1 = torch.zeros_like(inputs1).cuda()
        # x_2 = torch.zeros_like(inputs2).cuda()
        # x_3 = torch.zeros_like(inputs3).cuda()

        # for idx, (x1, x2, x3) in enumerate(zip(inputs1, inputs2, inputs3)):
        #     x_1[idx] = data_augment(x1).to(device)
        #     x_2[idx] = data_augment(x2).to(device)
        #     x_3[idx] = data_augment(x3).to(device)



        with torch.set_grad_enabled(True):
            # with autocast(False):
            _, out_1 = net(inputs1)
            _, out_2 = net(inputs2)
            # _, out_3 = net(inputs3)

            out_1 = F.normalize(out_1, dim=1)
            out_2 = F.normalize(out_2, dim=1)
            # out_3 = F.normalize(out_3, dim=1)


            loss = criterion(out_1.float(), out_2.float())
            # loss2 = criterion(out_2.float(), out_3.float())
            # loss3 = criterion(out_1.float(), out_3.float())

            # loss = loss1 + loss2 + loss3

            loss.backward()
            optimizer.step()

        loss_meter.update(loss.item())

    return loss_meter.average()

def SimCLR(net, epoch, criterion_ssl, optimizer, trainloader, args,\
             scaler, regressor=None, criterion_sup=None):
    loss_meter = AverageMeter()
    running_loss = 0
    net.train()
    if regressor != None:
        regressor.train()

    yhat = []
    y = []

    if args.mode == "multi":
        # for (i, (b1, b2, labels)) in enumerate(trainloader):
        for (i, (b1, b2, targets)) in enumerate(tqdm.tqdm(trainloader)):
            # b1 = b1.to(device)
            # b2 = b2.to(device)      
            targets = targets.to(device)

            optimizer.zero_grad()
            x_1 = b1#torch.zeros_like(b1).cuda()
            x_2 = b2#torch.zeros_like(b2).cuda()
            with torch.set_grad_enabled(True):
                with autocast():
                    # for idx, (x1, x2) in enumerate(zip(b1, b2)):
                    #     x1 = get_image_patch_tensor_from_video_batch(x1)
                    #     x2 = get_image_patch_tensor_from_video_batch(x2)

                    #     x_1[idx] = data_augment(x1)
                    #     x_2[idx] = data_augment(x2)

                    _, out_1 = net(x_1.to(device))
                    _, out_2 = net(x_2.to(device))
                    out_1 = F.normalize(out_1, dim=1)
                    out_2 = F.normalize(out_2, dim=1)
                    
                    indices = torch.arange(0, out_1.size(0), device=out_1.device)
                    labels = torch.cat((indices, indices))
                out_3, _ = net(b1)
                out_3 = regressor(out_3)

                loss_ssl = criterion_ssl(torch.cat([out_1.float(), out_2.float()], dim=0), labels)
                loss_sup = criterion_sup(out_3.view(-1).float(), targets.float())
                loss = 10 * loss_ssl + loss_sup

                loss_meter.update(loss.item())

                scaler.scale(loss).backward()
                # loss.backward()
                scaler.step(optimizer)
                scaler.update()
                # optimizer.step()
                # print(loss.item())
                running_loss += loss.item()

                yhat.append(out_3.view(-1).to("cpu").detach().numpy())
                y.append(targets.to("cpu").numpy())

        r2 = sklearn.metrics.r2_score(yhat, y)

        return loss_meter.average(), r2

    elif args.mode == "ssl":
        # for (i, (b1, b2, _)) in enumerate(trainloader):
        for (i, (b1, b2, *_)) in enumerate(tqdm.tqdm(trainloader)):

            optimizer.zero_grad()
            x_1 = b1#torch.zeros_like(b1).cuda()
            x_2 = b2#torch.zeros_like(b2).cuda()
            with autocast():
                # for idx, (x1, x2) in enumerate(zip(b1, b2)):
                #     x1 = get_image_patch_tensor_from_video_batch(x1)
                #     x2 = get_image_patch_tensor_from_video_batch(x2)

                #     x_1[idx] = data_augment(x1)
                #     x_2[idx] = data_augment(x2)

                _, out_1 = net(x_1.to(device))
                _, out_2 = net(x_2.to(device))
                out_1 = F.normalize(out_1, dim=1)
                out_2 = F.normalize(out_2, dim=1)
                
                # indices = torch.arange(0, out_1.size(0), device=out_1.device)
                # labels = torch.cat((indices, indices))

            # loss = criterion_ssl(torch.cat([out_1.float(), out_2.float()], dim=0), labels)
            loss = criterion_ssl(out_1.float(), out_2.float())
            loss_meter.update(loss.item())

            scaler.scale(loss).backward()
            # loss.backward()
            scaler.step(optimizer)
            scaler.update()
            # optimizer.step()
            # print(loss.item())
            running_loss += loss.item()


    return loss_meter.average(), running_loss

def checkpoint(model, model_store_folder, epoch_num, model_name, period, frames,\
                bestLoss, loss, r2, optim, scheduler = None):
    print('Saving checkpoints...')
    save = {
                'epoch': epoch_num,
                'state_dict': model.state_dict(),
                'period': period,
                'frames': frames,
                'best_loss': bestLoss,
                'loss': loss,
                'r2': r2,
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # model related arguments
    parser.add_argument('--encoder_model', default='r3d_18') #r2plus1d_18
    parser.add_argument('--encoder_pretrained', default=False)
    parser.add_argument('--mode', default='ssl')
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

    pretrain_str = 'pretrained' if args.encoder_pretrained == True else 'random'
    model_store_folder = get_next_model_folder(\
        '{}_{}_{}_{}'.format(args.mode, pretrain_str, args.frame_num, args.period), \
                    output_folder, args.run)
    try:
        os.mkdir(model_store_folder)
    except FileExistsError:
        print("Output folder exits")

    simclr_stats_csv_path = os.path.join(model_store_folder, "simclr_pred_stats.csv")
    regressor_stats_csv_path = os.path.join(model_store_folder, "regressor_pred_stats.csv")

    trainloader, valloader, testloader = dataloader(args.batch_size, args.mode, args)
    

    if (args.eval == False):

        if args.mode == 'cpc':
            args.prediction_step = 2
            args.negative_samples = 32
            args.subsample = True
            genc_hidden = 512
            gar_hidden = 128
            args.device = device
            args.calc_accuracy = False

            model = cpc.CPC(args, model='resnet18', pretrained=args.encoder_pretrained,\
                    genc_hidden=genc_hidden, gar_hidden=gar_hidden).to(device)

            inspect_model(model)

            if device.type == "cuda":
                model = torch.nn.DataParallel(model)

            optim_ssl = optimizer(model, args)

            print("\nStart training CPC!\n")
            scaler = GradScaler()
            bestLoss = float("inf")
            for epoch in range(0, 1000):

                epoch_loss = CPC_train(model, epoch, optim_ssl, trainloader, args, scaler)
                
                print('epoch {} average loss : {}'.format(epoch, epoch_loss))
                # Write stats into csv file
                stats = dict(
                        epoch      = epoch,
                        epoch_loss = epoch_loss
                    )
                write_csv_stats(simclr_stats_csv_path, stats)

                if epoch_loss < bestLoss:
                    checkpoint(model, model_store_folder, epoch, "best_cpc", args.period, args.frame_num,\
                                bestLoss, epoch_loss, 0, optim_ssl, None)
                    bestLoss = epoch_loss
            
            print("CPC training completed!")

        
        
        if args.mode == "multi":
            net = construct_3d_enc(args.encoder_model, args.encoder_pretrained, \
                    args.projection_size, 'projection_head')
            if device.type == "cuda":
                net = torch.nn.DataParallel(net)
        
            # criterion = losses.NTXentLoss(temperature=args.tau)
            criterion = NTXentLoss(device, args.batch_size, args.tau,\
             args.similarity, args.projection_size).to(device)
            optim_ssl = optimizer(net, args)

            scheduler = torch.optim.lr_scheduler.StepLR(optim_ssl, math.inf)

            epoch_start = 0
            
            net = net.to(device)
            inspect_model(net)

            regressor = construct_linear_regressor(net, args.projection_size)
            if device.type == "cuda":
                regressor = torch.nn.DataParallel(regressor)
            regressor = regressor.to(device)

            criterion_ssl = losses.NTXentLoss(temperature=args.tau)
            criterion_sup = torch.nn.MSELoss()
            optim_multi = optim.SGD(list(net.parameters()) + list(regressor.parameters()),\
                                     lr=1e-4, momentum=0.9, weight_decay=1e-4)

            if args.checkpoint != None:
                epoch_start = load_model(model_store_folder, args.checkpoint, regressor, \
                                    "regressor",optim=optim_multi, \
                                    scheduler=None, csv_path=regressor_stats_csv_path)
                epoch_start = load_model(model_store_folder, args.checkpoint, net,\
                                    "encoder",optim=optim_multi, \
                                    scheduler=None, csv_path=regressor_stats_csv_path)
                print("Starting from epoch: ", epoch_start)

            print("\nStart training Multi-task!\n")
            bestLoss = float("inf")
            scaler = GradScaler()

            for epoch in range(epoch_start, 46):
                train_epoch_loss, train_r2 = SimCLR(net, epoch, criterion_ssl,\
                                                    optim_multi, trainloader, args, scaler, regressor, criterion_sup)
                
                print('epoch {} average train loss : {}, r2: {}'.format(epoch, train_epoch_loss, train_r2))

                eval_epoch_loss, eval_r2 = eval_multi(net, regressor, epoch, \
                                                    criterion_sup, valloader)

                print('epoch {} average eval loss : {}, r2: {}'.format(epoch, eval_epoch_loss, eval_r2))
                
                
                checkpoint(regressor, model_store_folder, epoch, "regressor", args.period, args.frame_num,\
                                bestLoss, eval_epoch_loss, eval_r2, optim_multi)
                checkpoint(net, model_store_folder, epoch, "encoder", args.period, args.frame_num,\
                                bestLoss, eval_epoch_loss, eval_r2, optim_multi)
                

                if eval_epoch_loss < bestLoss:
                    checkpoint(regressor, model_store_folder, epoch, "best_regressor", args.period, args.frame_num,\
                                bestLoss, eval_epoch_loss, eval_r2, optim_multi)
                    checkpoint(net, model_store_folder, epoch, "best_net", args.period, args.frame_num,\
                                bestLoss, eval_epoch_loss, eval_r2, optim_multi)
                    bestLoss = eval_epoch_loss

            # Write stats into csv file
                stats = dict(
                        epoch_reg      = epoch,
                        train_epoch_loss = train_epoch_loss,
                        train_r2 = train_r2,
                        eval_epoch_loss = eval_epoch_loss,
                        eval_r2 = eval_r2
                    )
                write_csv_stats(regressor_stats_csv_path, stats)
            
            print("Multi-task training completed!")
            load_model(model_store_folder, args.checkpoint, regressor, "best_regressor",csv_path = regressor_stats_csv_path)
            load_model(model_store_folder, args.checkpoint, net, "best_net",csv_path = regressor_stats_csv_path)

            test_loss, test_r2 = eval_multi(net, regressor, 0, \
                    criterion_sup, testloader)

            stats = dict(
                        test_loss = test_loss,
                        test_r2 = test_r2
                    )
            write_csv_stats(regressor_stats_csv_path, stats)





        else:
            # net = construct_3d_enc(args.encoder_model, args.encoder_pretrained, \
            #     args.projection_size, 'projection_head')

            net = models_2d.Encoder('resnet18', args.encoder_pretrained, 128)
            if device.type == "cuda":
                net = torch.nn.DataParallel(net)
        
            # criterion = losses.NTXentLoss(temperature=args.tau) * args.frame_num
            criterion = NTXentLoss(device, args.batch_size , args.tau,\
             args.similarity, args.projection_size).to(device)
            optim_ssl = optimizer(net, args)

            scheduler = torch.optim.lr_scheduler.StepLR(optim_ssl, math.inf)

            epoch_start = 0
            
            net = net.to(device)
            inspect_model(net)

            if args.checkpoint != None:
                epoch_start = load_model(model_store_folder, args.checkpoint, net,\
                                    "best_simclr",optim=optim_ssl, \
                                    scheduler=None, csv_path=simclr_stats_csv_path)
                print("Starting from epoch: ", epoch_start)

            print("\nStart training simCLR 2D!\n")
            scaler = GradScaler()
            bestLoss = float("inf")
            for epoch in range(epoch_start, 1000):
                epoch_loss = Simclr_2d(net, epoch, criterion, optim_ssl, trainloader, scaler)
                # epoch_loss, running_loss = SimCLR(net, epoch, criterion, optim_ssl, trainloader, args, scaler)
                
                print('epoch {} average loss : {}'.format(epoch, epoch_loss))
                # if epoch%5==0:
                #     checkpoint(net, model_store_folder, epoch, "simclr", args.period, args.frame_num,\
                #                 bestLoss, epoch_loss, 0, optim_ssl, scheduler)

                # Write stats into csv file
                stats = dict(
                        epoch      = epoch,
                        epoch_loss = epoch_loss
                    )
                write_csv_stats(simclr_stats_csv_path, stats)

                if epoch_loss < bestLoss:
                    checkpoint(net, model_store_folder, epoch, "best_simclr_2d", args.period, args.frame_num,\
                                bestLoss, epoch_loss, 0, optim_ssl, scheduler)
                    bestLoss = epoch_loss

                if epoch % 100 == 0:
                    checkpoint(net, model_store_folder, epoch, "best_simclr_2d_{}".format(epoch), args.period, args.frame_num,\
                                bestLoss, epoch_loss, 0, optim_ssl, scheduler)

            
            print("simCLR training completed!")
    
    else:
        if args.mode == '2d' or args.mode == 'cpc' or args.mode == 'ssl':
            net = models_2d.Classifier_2D('resnet18', args.encoder_pretrained)
            if device.type == "cuda":
                net = torch.nn.DataParallel(net)
                net = net.to(device)

            criterion = torch.nn.BCELoss()
            optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
            scaler = GradScaler()

            epoch_start = 0
            bestLoss = float("inf")
            load_model(model_store_folder, args.checkpoint, net, "best_simclr_2d_200", optim=None, csv_path = regressor_stats_csv_path)
            # epoch_start = load_model(model_store_folder, args.checkpoint, regressor, "best_regressor", optim=reg_optimizer, csv_path = regressor_stats_csv_path)

            print("\nStart training 2d classifier!\n")

            for epoch in range(epoch_start, 45):
                train_epoch_loss, train_auc = train_2d_net(net, epoch, criterion, optimizer, trainloader, scaler)
                print('epoch {} average train loss : {}, r2: {}'.format(epoch, train_epoch_loss, train_auc))

                eval_epoch_loss, eval_auc = eval_2d_net(net, epoch, criterion, valloader)

                print('epoch {} average eval loss : {}, r2: {}'.format(epoch, eval_epoch_loss, eval_auc))
                
                # checkpoint(regressor, model_store_folder, epoch, "regressor", args.period, args.frame_num,\
                #                 bestLoss, eval_epoch_loss, eval_r2, reg_optimizer)
                # checkpoint(net, model_store_folder, epoch, "encoder", args.period, args.frame_num,\
                #                 bestLoss, eval_epoch_loss, eval_r2, con_optimizer)
                

                if eval_epoch_loss < bestLoss:
                    checkpoint(net, model_store_folder, epoch, "best_net", args.period, args.frame_num,\
                                bestLoss, eval_epoch_loss, eval_auc, optimizer)
                    bestLoss = eval_epoch_loss

                # Write stats into csv file
                stats = dict(
                        epoch_reg      = epoch,
                        train_epoch_loss = train_epoch_loss,
                        train_r2 = train_auc,
                        eval_epoch_loss = eval_epoch_loss,
                        eval_r2 = eval_auc
                    )
                write_csv_stats(regressor_stats_csv_path, stats)


            load_model(model_store_folder, args.checkpoint, net, "best_net", csv_path = regressor_stats_csv_path)
            test_loss, test_auc = eval_2d_net(net, 0, criterion, testloader)
            print('test loss : {}, auc: {}'.format(test_loss, test_auc))


            stats = dict(
                        test_loss = test_loss,
                        test_r2 = test_auc
                    )
            write_csv_stats(regressor_stats_csv_path, stats)



        elif args.mode == '3d':
            net = construct_3d_enc(args.encoder_model, args.encoder_pretrained, \
                args.projection_size, 'representation')
            # inspect_model(net)

            regressor = construct_linear_regressor(net, args.projection_size)
            # inspect_model(regressor)

            if device.type == "cuda":
                net = torch.nn.DataParallel(net)
                regressor = torch.nn.DataParallel(regressor)
            net = net.to(device)
            regressor = regressor.to(device)

            # params = list(net.parameters()) + list(regressor.parameters())
            scaler = GradScaler()
            # criterion = torch.nn.MSELoss() # Standard L2 loss
            criterion = torch.nn.BCELoss()
            reg_optimizer = optim.SGD(regressor.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)
            con_optimizer = optim.SGD(net.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)
            # scheduler = torch.optim.lr_scheduler.StepLR(reg_optimizer, math.inf)
            epoch_start = 0
            load_model(model_store_folder, args.checkpoint, net, "best_simclr_300", optim=con_optimizer, csv_path = regressor_stats_csv_path)
            # epoch_start = load_model(model_store_folder, args.checkpoint, regressor, "best_regressor", optim=reg_optimizer, csv_path = regressor_stats_csv_path)

            print("\nStart training Regressor!\n")
            bestLoss = float("inf")
            for epoch in range(epoch_start, 45):
                train_epoch_loss, train_r2 = train_regressor(net, regressor, epoch, \
                    criterion, reg_optimizer, con_optimizer, trainloader, scaler)
                print('epoch {} average train loss : {}, r2: {}'.format(epoch, train_epoch_loss, train_r2))

                eval_epoch_loss, eval_r2 = eval_regressor(net, regressor, epoch, \
                    criterion, valloader)

                print('epoch {} average eval loss : {}, r2: {}'.format(epoch, eval_epoch_loss, eval_r2))
                
                # checkpoint(regressor, model_store_folder, epoch, "regressor", args.period, args.frame_num,\
                #                 bestLoss, eval_epoch_loss, eval_r2, reg_optimizer)
                # checkpoint(net, model_store_folder, epoch, "encoder", args.period, args.frame_num,\
                #                 bestLoss, eval_epoch_loss, eval_r2, con_optimizer)
                

                if eval_epoch_loss < bestLoss:
                    checkpoint(regressor, model_store_folder, epoch, "best_regressor", args.period, args.frame_num,\
                                bestLoss, eval_epoch_loss, eval_r2, reg_optimizer)
                    if con_optimizer != None:
                        checkpoint(net, model_store_folder, epoch, "best_net", args.period, args.frame_num,\
                                    bestLoss, eval_epoch_loss, eval_r2, con_optimizer)
                    bestLoss = eval_epoch_loss

            # Write stats into csv file
                stats = dict(
                        epoch_reg      = epoch,
                        train_epoch_loss = train_epoch_loss,
                        train_r2 = train_r2,
                        eval_epoch_loss = eval_epoch_loss,
                        eval_r2 = eval_r2
                    )
                write_csv_stats(regressor_stats_csv_path, stats)
            
            print("Regressor training completed!")

            load_model(model_store_folder, args.checkpoint, regressor, "best_regressor",csv_path = regressor_stats_csv_path)
            load_model(model_store_folder, args.checkpoint, net, "best_net",csv_path = regressor_stats_csv_path)

            test_loss, test_r2 = eval_regressor(net, regressor, epoch, \
                    criterion, testloader)

            stats = dict(
                        test_loss = test_loss,
                        test_r2 = test_r2
                    )
            write_csv_stats(regressor_stats_csv_path, stats)





        



