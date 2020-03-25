import os
import argparse


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data as data

from torchvideotransforms import video_transforms, volume_transforms


# from models.models import ResNet18, ConvLSTM, ClassifierModule, ClassifierModuleDense
# from utils import utils 
# from lib import radam
from loss import SimLoss
from utils.helper_functions import get_next_model_folder, inspect_model
from utils.helper_functions import get_image_patch_tensor_from_video_batch, write_csv_stats
from utils.helper_classes import AverageMeter, GaussianBlur
from data.echonet_dataset import get_train_and_test_echonet_datasets
from models.models_3d import construct_3d_enc 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

s=1
video_transform_list = [video_transforms.RandomRotation(15),
                        # video_transforms.RandomCrop((50, 50)),
                        # video_transforms.RandomResize((112, 112)),
                        # video_transforms.RandomHorizontalFlip(),
                        # video_transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s),
                        volume_transforms.ClipToTensor(3, 3)]

data_augment = video_transforms.Compose(video_transform_list)

# color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
# data_augment = transforms.Compose([transforms.ToPILImage(),
#                                    transforms.RandomResizedCrop(32),
#                                    transforms.RandomHorizontalFlip(),
#                                    transforms.RandomApply([color_jitter], p=0.8),
#                                    transforms.RandomGrayscale(p=0.2),
#                                    GaussianBlur(),
#                                    transforms.ToTensor()])

def dataloader(batch_size):
    trainloader, testloader = None, None
    if args.dataset.lower() == 'echonet':
        
        trainset, testset = get_train_and_test_echonet_datasets("EF", frames=8, period=4)

        print('TRAIN DATASET: ', len(trainset), trainset[0][0].shape)
        print('TEST DATASET: ', len(testset), testset[0][0].shape)
        
        trainloader = data.DataLoader(trainset, batch_size=batch_size, \
                                    shuffle = True, num_workers=2, drop_last=True)
        testloader  = data.DataLoader(testset, batch_size=batch_size, \
                                    shuffle = False, num_workers=2)

    return trainloader, testloader

def optimizer(net, args):
    assert args.optimizer.lower() in ["sgd", "adam", "radam"], "Invalid Optimizer"

    if args.optimizer.lower() == "sgd":
	       return optim.SGD(net.parameters(), lr=args.lr, momentum=args.beta1, nesterov=args.nesterov)
    elif args.optimizer.lower() == "adam":
	       return optim.Adam(net.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    elif args.optimizer.lower() == "radam":
            return radam.RAdam(net.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

def test(net, epoch, criterion, testloader, args):
    net.eval()
    with torch.no_grad():
        correct = 0
        for i, data in enumerate(testloader, 0):
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = net(inputs)
            pred = F.softmax(outputs, 1)
            _, pred = torch.max(pred, 1)
            correct += torch.sum(pred==labels)
        print("Test set accuracy: " + str(float(correct)/ float(testloader.__len__())))

def SimCLR(net, epoch, criterion, optimizer, trainloader, args):
    loss_meter = AverageMeter()
    running_loss = 0
    net.train()

    for (i, (b, _)) in enumerate(trainloader):

        optimizer.zero_grad()
        x_1 = torch.zeros_like(b).cuda()
        x_2 = torch.zeros_like(b).cuda()

        for idx, x in enumerate(b):
            x = get_image_patch_tensor_from_video_batch(x)

            x_1[idx] = data_augment(x)
            x_2[idx] = data_augment(x)


        out_1 = net(x_1)
        out_2 = net(x_2)

        loss = criterion(torch.cat([out_1, out_2], dim=0))
        loss.backward()
        loss_meter.update(loss.item())
        optimizer.step()
        print(loss.item())
        running_loss += loss.item()

    return loss_meter.average(), running_loss

def checkpoint(net, model_store_folder, epoch_num):
    print('Saving checkpoints...')
    
    suffix_latest = 'epoch_{}.pth'.format(epoch_num)
    dict_net = net.state_dict()
    torch.save(dict_net,
               '{}/{}'.format(model_store_folder, suffix_latest))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # model related arguments
    parser.add_argument('--encoder_model', default='r3d_18')
    parser.add_argument('--encoder_pretrained', default=True)
    # optimization related arguments
    parser.add_argument('--batch_size', default=4, type=int,
                        help='input batch size')
    parser.add_argument('--epoch', default=45, type=int,
                        help='epochs to train for')
    parser.add_argument('--dataset', default='echonet')
    parser.add_argument('--optimizer', default='sgd', help='optimizer')
    parser.add_argument('--lr', default=0.001, type=float, help='LR')
    parser.add_argument('--beta1', default=0.9, type=float,
                        help='momentum for sgd, beta1 for adam')
    parser.add_argument('--beta2', default=0.999, type=float)
    parser.add_argument('--nesterov', default=False)
    parser.add_argument('--tau', default=0.1, type=float)
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
                    'SimCLR_{}_{}'.format(args.encoder_model, pretrain_str), \
                    output_folder)
    os.mkdir(model_store_folder)
    stats_csv_path = os.path.join(model_store_folder, "pred_stats.csv")

    trainloader, testloader = dataloader(args.batch_size)

    net = construct_3d_enc(args.encoder_model, args.encoder_pretrained, 32, 'projection_head')
    if device.type == "cuda":
        net = torch.nn.DataParallel(net)
    net = net.to(device)
    # inspect_model(net)
    
    criterion = SimLoss(tau=args.tau)
    optimizer = optimizer(net, args)
    
    print("\nStart training!\n")
    for epoch in range(1, args.epoch+1):
        batch_loss, running_loss = SimCLR(net, epoch, criterion, optimizer, trainloader, args)
        #test(net, epoch, criterion, testloader, args)
        print('epoch {} average loss : {}'.format(epoch, batch_loss))
        if epoch%5==0:
            checkpoint(net, model_store_folder, epoch)

        # Write stats into csv file
        stats = dict(
                epoch      = epoch,
                batch_loss = batch_loss,
                running_loss = running_loss
            )
        write_csv_stats(stats_csv_path, stats)
    
    print("Training completed!")


