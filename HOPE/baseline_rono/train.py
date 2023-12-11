from __future__ import division, absolute_import
from pickle import FALSE
import os
import time
import torch
import argparse
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
from models.mnist_img_fc import Img_FC, HeadNet
from models.mnist_pt_fc import Pt_fc
from models.SVCNN import SingleViewNet,DGCNN

from tools.mnist_feature_loader import FeatureDataloader, get_3dmnist
from tools.modelnet_dataloader import get_modelnet

from tools.utils import calculate_accuracy

from losses.MAE import MeanAbsoluteError
from losses.rdc_loss import RDC_loss
from losses.cross_modal_loss import CrossModalLoss

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'



def training(args):
    if not os.path.exists(args.save):
        os.makedirs(args.save)
        
    if args.dataset == '3DMNIST':
        img_net = Img_FC()
        pt_net = Pt_fc(args)

    else:
        img_net = SingleViewNet(pre_trained=None)
        pt_net = DGCNN(args.num_classes)
    
    head_net = HeadNet(num_classes=args.num_classes)
    img_net.train(True)
    pt_net.train(True)
    head_net.train(True)
    img_net = img_net.to('cuda')
    pt_net = pt_net.to('cuda')
    head_net = head_net.to('cuda')

    crc_criterion = MeanAbsoluteError(num_classes=args.num_classes)
    rdc_criterion = RDC_loss(num_classes=args.num_classes,alpha=args.alpha, feat_dim=256, warmup=args.warm_up)
    mg_criterion = CrossModalLoss()


    optimizer_img = optim.Adam(img_net.parameters(), lr=args.lr_img, weight_decay=args.weight_decay)
    optimizer_pt = optim.Adam(pt_net.parameters(), lr=args.lr_pt, weight_decay=args.weight_decay)
    optimizer_head = optim.Adam(head_net.parameters(), lr=args.lr_head, weight_decay=args.weight_decay)
    optimizer_rdc = optim.Adam(rdc_criterion.parameters(), lr=args.lr_center)
    

 
    if args.dataset == '3DMNIST':
        
        labeled_mnist3d,unlabeled_mnist3d,test_mnist3d = get_3dmnist(n_labeled=args.n_labeled, num_class=10)
        data_loader_loader = torch.utils.data.DataLoader(labeled_mnist3d, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=False,pin_memory=True)
    
    elif args.dataset == 'ModelNet40':
        labeled_modelnet,unlabeled_modelnet,test_modelnet = get_modelnet(n_labeled=args.n_labeled, num_class=40)
        data_loader_loader = torch.utils.data.DataLoader(labeled_modelnet, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=False,pin_memory=True)
        
    else:
        labeled_modelnet,unlabeled_modelnet,test_modelnet = get_modelnet(n_labeled=args.n_labeled, num_class=10)
        data_loader_loader = torch.utils.data.DataLoader(labeled_modelnet, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=False,pin_memory=True)


    iteration = 0
    start_time = time.time()

    if args.dataset == '3DMNIST':
        for epoch in range(args.epochs):
            for data in data_loader_loader:
                # image, point cloud, noisy labels, original labels (True labels for val.).
                img_feat, pt_feat, target = data
                
                img_feat = Variable(img_feat).to(torch.float32).to('cuda')
                pt_feat = Variable(pt_feat).to(torch.float32).to('cuda')
                target = Variable(target).to(torch.long).to('cuda')
                
                optimizer_img.zero_grad()
                optimizer_pt.zero_grad()
                optimizer_head.zero_grad()
                optimizer_rdc.zero_grad()

                # get common representations
                _img_feat = img_net(img_feat)
                _pt_feat = pt_net(pt_feat)

                # get prediction
                _img_pred, _pt_pred, _vis_img_feat, _vis_pt_feat = head_net(_img_feat, _pt_feat)

                # compute loss    
                pt_crc_loss = crc_criterion(_pt_pred, target)
                img_crc_loss = crc_criterion(_img_pred, target)         
                crc_loss = pt_crc_loss + img_crc_loss

                # ori_label use to val. clutering
                rdc_loss, centers = rdc_criterion(torch.cat((_img_feat, _pt_feat), dim = 0), torch.cat((target, target), dim = 0),torch.cat((target, target), dim = 0),  epoch)
                mg_loss = mg_criterion(torch.cat((_img_feat, _pt_feat), dim = 0))
                loss = args.weight_ce * crc_loss + args.weight_center * rdc_loss + args.weight_mse * mg_loss

                loss.backward()

                optimizer_img.step()
                optimizer_pt.step()
                optimizer_head.step()

                optimizer_rdc.step()

                # val. of classifications
                img_acc = calculate_accuracy(_img_pred, target)
                pt_acc = calculate_accuracy(_pt_pred, target)


                if (iteration%args.lr_step) == 0:
                    lr_img = args.lr_img * (0.1 ** (iteration // args.lr_step))
                    lr_pt = args.lr_pt * (0.1 ** (iteration // args.lr_step))
                    lr_head = args.lr_head * (0.1 ** (iteration // args.lr_step))
                    
                    for param_group in optimizer_img.param_groups:
                        param_group['lr_img'] = lr_img
                    for param_group in optimizer_pt.param_groups:
                        param_group['lr_pt'] = lr_pt
                    for param_group in optimizer_head.param_groups:
                        param_group['lr_head'] = lr_head
                
                if (iteration%args.center_lr_step) == 0:
                    lr_center = args.lr_center * (0.1 ** (iteration // args.lr_step))
                    for param_group in optimizer_rdc.param_groups:
                        param_group['lr'] = lr_center

                if iteration % args.per_print == 0:
                    print("loss: %f  rcd_loss: %f  crc_loss: %f  mg_loss: %f" % (loss.item(), rdc_loss.item(), crc_loss, mg_loss))
                    print('[%d][%d]  img_acc: %f pt_acc %f time: %f  vid: %d' % (epoch, iteration, img_acc, pt_acc, time.time() - start_time, target.size(0)))
                    start_time = time.time()

                iteration = iteration + 1


                if((iteration+1) % args.per_save) ==0:
                    print('----------------- Save The Network ------------------------')
                    with open(args.save + str(args.n_labeled) +'-'+ str(iteration+1)+'-head_net.pkl', 'wb') as f:
                        torch.save(head_net, f)
                    with open(args.save + str(args.n_labeled) + '-'+ str(iteration+1)+'-img_net.pkl', 'wb') as f:
                        torch.save(img_net, f)
                    with open(args.save + str(args.n_labeled) + '-'+ str(iteration+1)+'-pt_net.pkl', 'wb') as f:
                        torch.save(pt_net, f)
                    np.save(args.save + str(args.n_labeled) + '-'+ str(iteration+1)+'-centers', centers.cpu().detach().numpy())
                    
    else:  ##training for ModelNet40/10
        for epoch in range(args.epochs):
            for data in data_loader_loader:
                # image, point cloud, noisy labels, original labels (True labels for val.).
                pt, img1, img2,  target, target_vec = data
                
                img1 = Variable(img1).to(torch.float32).to('cuda')
                img2 = Variable(img2).to(torch.float32).to('cuda')
                pt = Variable(pt.permute(0,2,1)).to(torch.float32).to('cuda')
                target = Variable(target.squeeze(1)).to(torch.long).to('cuda')
                #print('target: ', target.shape)
                
                optimizer_img.zero_grad()
                optimizer_pt.zero_grad()
                optimizer_head.zero_grad()
                optimizer_rdc.zero_grad()
                
                # get common representations
                _img_feat = img_net(img1,img2)
                _pt_feat = pt_net(pt)

                # get prediction
                _img_pred, _pt_pred, _vis_img_feat, _vis_pt_feat = head_net(_img_feat, _pt_feat)


                # compute loss    
                pt_crc_loss = crc_criterion(_pt_pred, target)
                img_crc_loss = crc_criterion(_img_pred, target)         
                crc_loss = pt_crc_loss + img_crc_loss

                # ori_label use to val. clutering
                rdc_loss, centers = rdc_criterion(torch.cat((_img_feat, _pt_feat), dim = 0), torch.cat((target, target), dim = 0),torch.cat((target, target), dim = 0),  epoch)
                mg_loss = mg_criterion(torch.cat((_img_feat, _pt_feat), dim = 0))
                #mg_loss = torch.zeros_like(mg_loss).cuda()
                loss = args.weight_ce * crc_loss + args.weight_center * rdc_loss + args.weight_mse * mg_loss

                loss.backward()

                optimizer_img.step()
                optimizer_pt.step()
                optimizer_head.step()
                optimizer_rdc.step()


                # val. of classifications
                img_acc = calculate_accuracy(_img_pred, target)
                pt_acc = calculate_accuracy(_pt_pred, target)


                if (iteration%args.lr_step) == 0:
                    lr_img = args.lr_img * (0.1 ** (iteration // args.lr_step))
                    lr_pt = args.lr_pt * (0.1 ** (iteration // args.lr_step))
                    lr_head = args.lr_head * (0.1 ** (iteration // args.lr_step))
                    
                    for param_group in optimizer_img.param_groups:
                        param_group['lr_img'] = lr_img
                    for param_group in optimizer_pt.param_groups:
                        param_group['lr_pt'] = lr_pt
                    for param_group in optimizer_head.param_groups:
                        param_group['lr_head'] = lr_head
                
                if (iteration%args.center_lr_step) == 0:
                    lr_center = args.lr_center * (0.1 ** (iteration // args.lr_step))
                    for param_group in optimizer_rdc.param_groups:
                        param_group['lr'] = lr_center

                if iteration % args.per_print == 0:
                    print("loss: %f  rcd_loss: %f  crc_loss: %f  mg_loss: %f" % (loss.item(), rdc_loss.item(), crc_loss, mg_loss))
                    print('[%d][%d]  img_acc: %f pt_acc %f time: %f  vid: %d' % (epoch, iteration, img_acc, pt_acc, time.time() - start_time, target.size(0)))
                    start_time = time.time()

                iteration = iteration + 1


                if((iteration+1) % args.per_save) ==0:
                    print('----------------- Save The Network ------------------------')
                    with open(args.save + str(args.n_labeled) +'-'+ str(iteration+1)+'-head_net.pkl', 'wb') as f:
                        torch.save(head_net, f)
                    with open(args.save + str(args.n_labeled) + '-'+ str(iteration+1)+'-img_net.pkl', 'wb') as f:
                        torch.save(img_net, f)
                    with open(args.save + str(args.n_labeled) + '-'+ str(iteration+1)+'-pt_net.pkl', 'wb') as f:
                        torch.save(pt_net, f)
                    np.save(args.save + str(args.n_labeled) + '-'+ str(iteration+1)+'-centers', centers.cpu().detach().numpy())

    
if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Cross Modal Retrieval for Point Cloud, Mesh, and Image Models')

    parser.add_argument('--dataset', type=str, default='3DMNIST', metavar='dataset',choices=['3DMNIST', 'ModelNet40', 'ModelNet10'],
                        help='ModelNet10 or ModelNet40')

    parser.add_argument('--n_labeled', type=int, default=200, metavar='n_labeled',
                        help='number of labeled data in the dataset')
        
    parser.add_argument('--num_classes', type=int, default=10, metavar='num_classes',
                        help='10 or 40')

    parser.add_argument('--save', type=str,  default='./checkpoints/RONO/3DMNIST/vis_codes/',
                        help='path to save the final model')

    parser.add_argument('--gpu_id', type=str,  default='2',
                        help='GPU used to train the network')
    
    parser.add_argument('--batch_size', type=int, default=50, metavar='batch_size',
                        help='Size of batch)')

    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of episode to train 100')
    
    parser.add_argument('--dropout', type=str, default=0.4, metavar='dropout',
                        help='dropout')
    #optimizer
    parser.add_argument('--lr_img', type=float, default=5e-5, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    
    parser.add_argument('--lr_pt', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    
    parser.add_argument('--lr_head', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')

    parser.add_argument('--lr_step', type=int,  default=50,
                        help='how many iterations to decrease the learning rate')
    
    parser.add_argument('--center_lr_step', type=int,  default=50,
                        help='how many iterations to decrease the learning rate')

    parser.add_argument('--lr_center', type=float, default=1e-4, metavar='LR',
                        help='learning rate for center loss (default: 0.5)  0.001')
                                         
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    
    parser.add_argument('--warm_up', type=float, default=25, metavar='M',
                        help='warm up epoch affect rdc loss')   #15 - 30

    parser.add_argument('--alpha', type=float, default=0.3, metavar='alpha',
                        help='alpha' )  

    parser.add_argument('--weight_center', type=float, default=1, metavar='weight_center',
                        help='weight center' )   # 0.1

    parser.add_argument('--weight_ce', type=float, default=1, metavar='weight_ce',
                        help='weight ce' ) #20 - 10   50

    parser.add_argument('--weight_mse', type=float, default=1, metavar='weight_mse',
                        help='weight mse' )

    parser.add_argument('--weight_decay', type=float, default=1e-5, metavar='weight_decay',
                        help='learning rate (default: 1e-3)')

    parser.add_argument('--per_save', type=int,  default=200,
                        help='how many iterations to save the model')

    parser.add_argument('--per_print', type=int,  default=100,
                        help='how many iterations to print the loss and accuracy')
                        


    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    torch.backends.cudnn.enabled = True
    training(args)