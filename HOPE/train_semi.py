from __future__ import division, absolute_import
import math
from pickle import FALSE
import os
import sys
import time
import torch
import argparse
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import scipy
from sklearn.preprocessing import normalize
from models.mnist_img_fc import Semi_img
from models.mnist_pt_fc import Semi_pt
import torch.nn as nn
from tools.mnist_feature_loader import FeatureDataloader, get_3dmnist
from models.SVCNN import Semi_img_mn, Semi_pt_mn
from losses.MAE import MeanAbsoluteError
from losses.scl import MGLoss,SupConLoss,modal_alignment_loss
from tools.modelnet_dataloader import get_modelnet
from losses.distributional_quantization_losses import quantization_swdc_loss
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def add_noise(model, noise_stddev):
    for param in model.parameters():
        noise = torch.randn(param.size()) * noise_stddev
        param.data.add_(noise)
        
def adaptive_threshold_generate(outputs, targets, threshold, epoch):
    beta = 0.97
    value = threshold[0]
    outputs_l = outputs[1:, :]
    targets_l = targets[1:]
    probs = torch.softmax(outputs_l, dim=1)
    max_probs, max_idx = torch.max(probs, dim=1)
    eq_idx = np.where(targets_l.eq(max_idx).cpu() == 1)[0]

    probs_new = max_probs[eq_idx]
    targets_new = targets_l[eq_idx]
    
    num_classes = outputs.shape[-1]
    
    for i in range(num_classes):
        idx = np.where(targets_new.cpu() == i)[0]
        if idx.shape[0] != 0:
            threshold[i] = probs_new[idx].mean().cpu() * beta / (1 + math.exp(-1 * epoch)) if probs_new[idx].mean().cpu() * beta / (1 + math.exp(-1 * epoch)) >= value else value
        else:
            threshold[i] = value
    return threshold

def mask_generate(num_classes, max_probs, max_idx, batch, threshold):
    mask_ori = torch.zeros(batch).cuda()
    for i in range(num_classes):
        idx = np.where(max_idx.cpu() == i)[0]
        m = max_probs[idx].ge(threshold[i]).float()
        for k in range(len(idx)):
            mask_ori[idx[k]]+=m[k]
    return mask_ori.cuda()

def to_one_hot(tensor,nClasses):
    one_hot = torch.nn.functional.one_hot(tensor, nClasses)
    return one_hot


def topk_ce_FPL(pred, inputs_2, temp_k=3, threshold=0.90, weight='concave',gamma=0.2):
    soft_plus = torch.nn.Softplus()
    prob_topk, pse_topk = torch.topk(inputs_2.float().softmax(dim=1).detach(), k=temp_k + 1, dim=1)
    class_num = pred.shape[1]

    margin = torch.zeros(pred.shape[0]).to(pred.device) + gamma * class_num
    sum_pred_pos = torch.zeros_like(pred) - 1  # b,c,w,h
    sum_pred_neg = torch.zeros_like(pred) - 1  # b,c,w,h
    flag_mask = torch.zeros_like(pred[:, 0]) - 1  # b,w,h
    weight_mask = torch.zeros_like(pred[:, 0]) + 1
    one_hot_dict = {}
    for i in range(pse_topk.shape[1]):
        one_hot_dict[i] = to_one_hot(pse_topk[:, i], class_num)

    cumulative_prob = prob_topk.clone()
    for i in range(1, cumulative_prob.shape[1]):
        cumulative_prob[:, i] = cumulative_prob[:, i] + cumulative_prob[:, i - 1]

    for i in range(1, cumulative_prob.shape[1]):
        k = i - 1  # i-1 form 0
        if i == cumulative_prob.shape[1] - 1:
            mask_k = (flag_mask == -1)
        else:
            mask_k = (cumulative_prob[:, i] >= threshold) * (flag_mask == -1)
        flag_mask[mask_k] = 1

        if weight == 'linear':
            weight_mask[mask_k] = ((cumulative_prob[:, i] / (i + 1) - prob_topk[:, i]) / (
                        cumulative_prob[:, i] / ((i + 1))))[mask_k]
        elif weight == 'convex':
            weight_mask[mask_k] = ((cumulative_prob[:, i] / (i + 1) - prob_topk[:, i]) / (
                        prob_topk[:, i] + cumulative_prob[:, i] / (i + 1)))[mask_k]
        elif weight == 'concave':
            weight_mask[mask_k] = (torch.log(
                1 + cumulative_prob[:, i] / (i + 1) * 50 - prob_topk[:, i] * 50) / torch.log(
                1 + 50 * cumulative_prob[:, i] / ((i + 1))))[mask_k]

        mask_k = mask_k[:, None].expand(pred.shape[0], class_num).contiguous()

        pse_mask_pos = 0
        for j in range(k + 1):
            pse_mask_pos += one_hot_dict[j]  # b,c,w,h
        pse_mask_neg = 1 - pse_mask_pos
        sum_pred_pos[mask_k] = (-(pred * pse_mask_pos) - pse_mask_neg * 1e7)[mask_k]
        sum_pred_neg[mask_k] = ((pred * pse_mask_neg) - pse_mask_pos * 1e7)[mask_k]

    loss_topk = soft_plus(torch.logsumexp(sum_pred_pos, dim=1) + torch.logsumexp(sum_pred_neg, dim=1)+ torch.exp(margin))
    weight_mask[weight_mask > 1] = 1
    return loss_topk, weight_mask

def training(args):
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    if args.dataset =='3DMNIST':
        semi_img_net = Semi_img(args.num_classes)
        semi_pt_net = Semi_pt(args)
    else:
        semi_img_net = Semi_img_mn(args.num_classes)
        semi_pt_net = Semi_pt_mn(args.num_classes)
        
    noise_stddev = 0.0001  
    add_noise(semi_img_net, noise_stddev)
    add_noise(semi_pt_net, noise_stddev)
    
    semi_img_net.train(True)
    semi_pt_net.train(True)
    semi_img_net = semi_img_net.to('cuda')
    semi_pt_net = semi_pt_net.to('cuda')
    

    crc_criterion = MeanAbsoluteError(num_classes=args.num_classes)
    align_criterion = MGLoss(temperature=0.07)
    global_align_criterion = SupConLoss()
    unsup_criterion = nn.CrossEntropyLoss(reduction='mean')

    optimizer_img = optim.Adam(semi_img_net.parameters(), lr=args.lr_img, weight_decay=args.weight_decay)
    optimizer_pt = optim.Adam(semi_pt_net.parameters(), lr=args.lr_pt, weight_decay=args.weight_decay)

    if args.dataset == '3DMNIST':
        labeled_trainset,unlabeled_trainset,test_set = get_3dmnist(n_labeled=args.n_labeled, num_class=10)
        train_loader = torch.utils.data.DataLoader(labeled_trainset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=False,pin_memory=True)
        unsupervised_train_loader = torch.utils.data.DataLoader(unlabeled_trainset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True,pin_memory=True)

    else:
        labeled_trainset,unlabeled_trainset,test_set = get_modelnet(n_labeled=args.n_labeled, num_class=args.num_classes)
        train_loader = torch.utils.data.DataLoader(labeled_trainset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=False,pin_memory=True)
        unsupervised_train_loader = torch.utils.data.DataLoader(unlabeled_trainset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True,pin_memory=True)

    iteration = 0
    start_time = time.time()

    num_train_imgs = len(labeled_trainset)
    num_unsup_imgs = len(unlabeled_trainset)
    max_samples = max(num_train_imgs, num_unsup_imgs)     # Define the iterations in an epoch
    niters_per_epoch = int(math.ceil(max_samples * 1.0 // args.batch_size))
    
    #threshold = [ 0.8 for i in range(args.num_classes)]
    
    best = 0

    for epoch in range(args.epochs):
        if epoch == 0:
            threshold  = [ 0.7 for i in range(args.num_classes)]
        else:
            outputs_new = torch.ones(1, args.num_classes).cuda()
            targets_new = torch.ones(1).long().cuda()
            if args.dataset == '3DMNIST':
                val_loader = torch.utils.data.DataLoader(labeled_trainset, batch_size=16, shuffle=False, num_workers=10)
                for data in val_loader:
                    img_feat, pt_feat, ori_label = data
                    with torch.no_grad():
                        img_feat = Variable(img_feat).to(torch.float32).to('cuda')
                        pt_feat = Variable(pt_feat).to(torch.float32).to('cuda')
                        ori_label = Variable(ori_label).to(torch.long).to('cuda')
                        ##########################################
                        semi_img_net = semi_img_net.to('cuda')
                        _img_feat, sup_img_pred1 = semi_img_net(img_feat, step=1)
                        semi_pt_net = semi_pt_net.to('cuda')
                        _pt_feat, sup_pt_pred1 = semi_pt_net(pt_feat, step=1)
                        outputs = (sup_img_pred1 + sup_pt_pred1)/2
                        outputs_new = torch.cat((outputs_new, outputs), dim=0)
                        targets_new = torch.cat((targets_new, ori_label.squeeze(-1)), dim=0)
                
            else:
                val_loader = torch.utils.data.DataLoader(labeled_trainset, batch_size=16, shuffle=False, num_workers=10)
                for data in val_loader:
                    pt, img1, img2,  target, target_vec = data
                    with torch.no_grad():
                        img1 = Variable(img1).to(torch.float32).to('cuda')
                        img2 = Variable(img2).to(torch.float32).to('cuda')
                        pt = Variable(pt.permute(0,2,1)).to(torch.float32).to('cuda')
                        target = Variable(target.squeeze(1)).to(torch.long).to('cuda')
                        ##########################################
                        semi_img_net = semi_img_net.to('cuda')
                        _img_feat, sup_img_pred1 = semi_img_net(img1,img2, step=1)
                        semi_pt_net = semi_pt_net.to('cuda')
                        _pt_feat, sup_pt_pred1 = semi_pt_net(pt, step=1)
                        outputs = (sup_img_pred1 + sup_pt_pred1)/2
                        outputs_new = torch.cat((outputs_new, outputs), dim=0)
                        targets_new = torch.cat((targets_new, target.squeeze(-1)), dim=0)
            
            threshold = adaptive_threshold_generate(outputs_new, targets_new, threshold, epoch)
            
        print('\nEpoch: [%d | %d] ' % (epoch, args.epochs))
        print('\nThreshold:',threshold)


        dataloader = iter(train_loader)
        unsupervised_dataloader = iter(unsupervised_train_loader)


        ''' supervised part '''
        for idx in range(niters_per_epoch):
            optimizer_img.zero_grad()
            optimizer_pt.zero_grad()

            start_time = time.time()

            
            try:
                minibatch = dataloader.__next__()
                unsup_minibatch = unsupervised_dataloader.__next__()

            except:
                dataloader = iter(train_loader)
                unsupervised_dataloader = iter(unsupervised_train_loader)
          
                minibatch = dataloader.__next__()
                unsup_minibatch = unsupervised_dataloader.__next__()

            if args.dataset == '3DMNIST': 
                imgs = minibatch[0].to(torch.float32)
                pts = minibatch[1].to(torch.float32)
                gts = minibatch[2].to(torch.long)
                unsup_imgs = unsup_minibatch[0].to(torch.float32)
                unsup_pts = unsup_minibatch[1].to(torch.float32)
                imgs = imgs.cuda(non_blocking=True)
                pts = pts.cuda(non_blocking=True)
                gts = gts.cuda(non_blocking=True)
                unsup_imgs = unsup_imgs.cuda(non_blocking=True)
                unsup_pts = unsup_pts.cuda(non_blocking=True)
            else:
                imgs1 = minibatch[1].to(torch.float32)
                imgs2 = minibatch[2].to(torch.float32)
                pts = minibatch[0].permute(0,2,1).to(torch.float32)
                gts = minibatch[3].squeeze(1).to(torch.long)
                unsup_imgs1 = unsup_minibatch[1].to(torch.float32)
                unsup_imgs2 = unsup_minibatch[2].to(torch.float32)
                unsup_pts = unsup_minibatch[0].permute(0,2,1).to(torch.float32)
                imgs1 = imgs1.cuda(non_blocking=True)
                imgs2 = imgs2.cuda(non_blocking=True)
                pts = pts.cuda(non_blocking=True)
                gts = gts.cuda(non_blocking=True)
                unsup_imgs1 = unsup_imgs1.cuda(non_blocking=True)
                unsup_imgs2 = unsup_imgs2.cuda(non_blocking=True)
                unsup_pts = unsup_pts.cuda(non_blocking=True)

            ##semi-supervised learning
            with torch.no_grad():
                if args.dataset == '3DMNIST': 
                    img_feat1, img_pred1 = semi_img_net(unsup_imgs, step=1)
                    logits_img_tea_1 = img_pred1.detach()
                    img_feat2, img_pred2 = semi_img_net(unsup_imgs, step=2)
                    logits_img_tea_2 = img_pred2.detach()
                else:
                    img_feat1, img_pred1 = semi_img_net(unsup_imgs1,unsup_imgs2, step=1)
                    logits_img_tea_1 = img_pred1.detach()
                    img_feat2, img_pred2 = semi_img_net(unsup_imgs1,unsup_imgs2, step=2)
                    logits_img_tea_2 = img_pred2.detach()   
                    
                pt_feat1, pt_pred1 = semi_pt_net(unsup_pts, step=1)
                logits_pt_tea_1 = pt_pred1.detach()
                pt_feat2, pt_pred2 = semi_pt_net(unsup_pts, step=2)
                logits_pt_tea_2 = pt_pred2.detach()

            ##semi_img_cps_loss#####
            _, ps_label_img_1 = torch.max(logits_img_tea_1, dim=1)
            ps_label_img_1 = ps_label_img_1.long()
            _, ps_label_img_2 = torch.max(logits_img_tea_2, dim=1)
            ps_label_img_2 = ps_label_img_2.long()

            if args.dataset == '3DMNIST': 
                semi_feature_img_1, logits_img_stu_1 = semi_img_net(unsup_imgs, step=1)
                semi_feature_img_2, logits_img_stu_2 = semi_img_net(unsup_imgs, step=2)
            else:
                semi_feature_img_1, logits_img_stu_1 = semi_img_net(unsup_imgs1,unsup_imgs2, step=1)
                semi_feature_img_2, logits_img_stu_2 = semi_img_net(unsup_imgs1,unsup_imgs2, step=2)

            loss_img_topk_12, weight_img_mask_12 = topk_ce_FPL(logits_img_stu_1, logits_img_tea_2, threshold = 0.9, weight='concave')
            loss_img_topk_21, weight_img_mask_21 = topk_ce_FPL(logits_img_stu_2, logits_img_tea_1, threshold = 0.9, weight='concave')
            
            
            cps_loss_img_base = unsup_criterion(logits_img_stu_1, ps_label_img_2) + unsup_criterion(logits_img_stu_2, ps_label_img_1)
            #cps_img_scalar = cps_loss_img_base.item()/((loss_img_topk_12.mean()+loss_img_topk_21.mean()).item() +1e-7)# converge to >0.93 is usually good
            cps_loss_img = (loss_img_topk_12*weight_img_mask_12).mean()+(loss_img_topk_21*weight_img_mask_21).mean()
            
            cps_img_weight_balance = cps_loss_img_base.item()/(cps_loss_img.item() +1e-7)# keep loss balance
            cps_loss_img = cps_loss_img * cps_img_weight_balance
            cps_img_weight = 1.0
            cps_loss_img = cps_loss_img * cps_img_weight
            
            ##semi_pt_cps_loss#####
            _, ps_label_pt_1 = torch.max(logits_pt_tea_1, dim=1)
            ps_label_pt_1 = ps_label_pt_1.long()
            _, ps_label_pt_2 = torch.max(logits_pt_tea_2, dim=1)
            ps_label_pt_2 = ps_label_pt_2.long()

            semi_feature_pt_1, logits_pt_stu_1 = semi_pt_net(unsup_pts, step=1)
            semi_feature_pt_2, logits_pt_stu_2 = semi_pt_net(unsup_pts, step=2)


            loss_pt_topk_12, weight_pt_mask_12 = topk_ce_FPL(logits_pt_stu_1, logits_pt_tea_2, threshold = 0.9, weight='concave')
            loss_pt_topk_21, weight_pt_mask_21 = topk_ce_FPL(logits_pt_stu_2, logits_pt_tea_1, threshold = 0.9, weight='concave')
            cps_loss_pt_base = unsup_criterion(logits_pt_stu_1, ps_label_pt_2) + unsup_criterion(logits_pt_stu_2, ps_label_pt_1)
            #cps_pt_scalar = cps_loss_pt_base.item()/((loss_pt_topk_12.mean()+loss_pt_topk_21.mean()).item()+1e-7) # converge to >0.93 is usually good
            
            cps_loss_pt = (loss_pt_topk_12*weight_pt_mask_12).mean()+(loss_pt_topk_21*weight_pt_mask_21).mean()
            
            cps_pt_weight_balance = cps_loss_pt_base.item()/(cps_loss_pt.item()+1e-7) # keep loss balance
            cps_loss_pt = cps_loss_pt * cps_pt_weight_balance
            cps_pt_weight = 1.0
            cps_loss_pt = cps_loss_pt * cps_pt_weight
            ##sum two type cps loss##
            cps_loss = cps_loss_img + cps_loss_pt
            #cps_loss = torch.zeros_like(cps_loss).cuda()
            
            ##supervised learning
            if args.dataset == '3DMNIST':
                sup_img_feat1, sup_img_pred1 = semi_img_net(imgs, step=1)
                sup_img_feat2, sup_img_pred2 = semi_img_net(imgs, step=2)
            else:
                sup_img_feat1, sup_img_pred1 = semi_img_net(imgs1,imgs2, step=1)
                sup_img_feat2, sup_img_pred2 = semi_img_net(imgs1,imgs2, step=2)       
                         
            sup_pt_feat1, sup_pt_pred1 = semi_pt_net(pts, step=1)
            sup_pt_feat2, sup_pt_pred2 = semi_pt_net(pts, step=2)

            # compute  crc loss    
            img_crc_loss1 = crc_criterion(sup_img_pred1, gts)
            img_crc_loss2 = crc_criterion(sup_img_pred2, gts)
            img_crc_loss = img_crc_loss1 + img_crc_loss2
            
            pt_crc_loss1 = crc_criterion(sup_pt_pred1, gts)  
            pt_crc_loss2 = crc_criterion(sup_pt_pred2, gts)       
            pt_crc_loss = pt_crc_loss1 + pt_crc_loss2
            
            crc_loss = pt_crc_loss + img_crc_loss
            #crc_loss = torch.zeros_like(crc_loss).cuda()
            
            ##compute  mg loss 
            # mg_loss_local_1 = mg_criterion(torch.cat((sup_img_feat1, sup_pt_feat1), dim = 0))
            # mg_loss_local_2 = mg_criterion(torch.cat((sup_img_feat2, sup_pt_feat2), dim = 0))
            # local_mg_loss = mg_loss_local_1 + mg_loss_local_2 
            
            # # #supervised_mg_loss

            sup_mg_loss_local_1 = modal_alignment_loss(sup_img_feat1, sup_pt_feat1,gts,gts)
            sup_mg_loss_local_2 = modal_alignment_loss(sup_img_feat2, sup_pt_feat2,gts,gts)
            
            local_mg_loss = sup_mg_loss_local_1 + sup_mg_loss_local_2

            # # #semi-supervised_mg_loss

            ## mask
            p_img1 = torch.softmax(logits_img_stu_1, dim=1) 
            max_probs_img1, max_idx_img1 = torch.max(p_img1, dim=1)
            max_idx_img1 = max_idx_img1.detach()
            mask_img1 = mask_generate(args.num_classes,max_probs_img1, max_idx_img1, args.batch_size, threshold)
            p_img2 = torch.softmax(logits_img_stu_2, dim=1) 
            max_probs_img2, max_idx_img2 = torch.max(p_img2, dim=1)
            max_idx_img2 = max_idx_img2.detach()
            mask_img2 = mask_generate(args.num_classes,max_probs_img2, max_idx_img2, args.batch_size, threshold)
            p_pt1 = torch.softmax(logits_pt_stu_1, dim=1)
            max_probs_pt1, max_idx_pt1 = torch.max(p_pt1, dim=1)
            max_idx_pt1 = max_idx_pt1.detach()
            mask_pt1 = mask_generate(args.num_classes,max_probs_pt1, max_idx_pt1, args.batch_size, threshold)
            p_pt2 = torch.softmax(logits_pt_stu_2, dim=1)
            max_probs_pt2, max_idx_pt2 = torch.max(p_pt2, dim=1)
            max_idx_pt2 = max_idx_pt2.detach()
            mask_pt2 = mask_generate(args.num_classes,max_probs_pt2, max_idx_pt2, args.batch_size, threshold)
            mask1 = torch.cat((mask_img1.unsqueeze(0),mask_pt1.unsqueeze(0)),dim=0)
            mask2 = torch.cat((mask_img2.unsqueeze(0),mask_pt2.unsqueeze(0)),dim=0)
            

            semi_mg_loss_local_1 = (align_criterion(torch.cat((semi_feature_img_1.unsqueeze(1), semi_feature_pt_1.unsqueeze(1)),dim = 1),ps_label_img_1,ps_label_pt_1)*mask1).mean()
            semi_mg_loss_local_2 = (align_criterion(torch.cat((semi_feature_img_2.unsqueeze(1), semi_feature_pt_2.unsqueeze(1)),dim = 1),ps_label_img_2,ps_label_pt_2)*mask2).mean()
            local_semi_mg_loss = semi_mg_loss_local_1 + semi_mg_loss_local_2 

            
            img1_weight = torch.cat((semi_img_net.head1[0].weight,semi_img_net.head2[0].weight),dim=0)
            img2_weight = torch.cat((semi_img_net.head1[2].weight,semi_img_net.head2[2].weight),dim=0)
            pt1_weight = torch.cat((semi_pt_net.head1[0].weight,semi_pt_net.head2[0].weight),dim=0)
            pt2_weight = torch.cat((semi_pt_net.head1[2].weight,semi_pt_net.head2[2].weight),dim=0)

            global_mg_loss_1 = global_align_criterion(torch.cat((img1_weight.unsqueeze(1), pt1_weight.unsqueeze(1)),dim=1)).mean()

            global_mg_loss_2 = global_align_criterion(torch.cat((img2_weight.unsqueeze(1), pt2_weight.unsqueeze(1)),dim=1)).mean()
            global_mg_loss = global_mg_loss_1 + global_mg_loss_2
            
            mg_loss = local_mg_loss + 0.01*(local_semi_mg_loss  + global_mg_loss)
            #mg_loss = torch.zeros_like(mg_loss).cuda()
            
            # compute quantization loss
            if args.quantization_type == 'swdc':
                quantization_loss1 = quantization_swdc_loss(sup_img_feat1.view(sup_img_feat1.size(0), -1),sup_pt_feat1.view(sup_pt_feat1.size(0), -1))
                quantization_loss2 = quantization_swdc_loss(sup_img_feat2.view(sup_img_feat2.size(0), -1),sup_pt_feat2.view(sup_pt_feat2.size(0), -1))
               
            else:
                quantization_loss1 = torch.tensor(0.0)
                quantization_loss2 = torch.tensor(0.0)


            quantization_loss = quantization_loss1+quantization_loss2
            
    
            crc_loss = args.weight_crc * crc_loss
            quan_loss = args.quantization_weight * quantization_loss
            cps_loss = args.cps_weight *cps_loss
            align_loss= args.weight_align * mg_loss

            #total loss
            loss = crc_loss + align_loss + cps_loss + quan_loss
            
            
            loss.backward()

            optimizer_img.step()
            optimizer_pt.step()



            if (iteration%args.lr_step) == 0:
                lr_img = args.lr_img * (0.1 ** (iteration // args.lr_step))
                lr_pt = args.lr_pt * (0.1 ** (iteration // args.lr_step))     
                
                for param_group in optimizer_img.param_groups:
                    param_group['lr_img'] = lr_img
                for param_group in optimizer_pt.param_groups:
                    param_group['lr_pt'] = lr_pt

            


            if iteration % args.per_print == 0:
                print('[%d][%d] time: %f vid: %d' % (epoch, iteration, time.time() - start_time, gts.size(0)))
                print("loss: %f  crc_loss: %f align_loss: %f cps_loss: %f quan_loss: %f semi_mg_loss: %f  global_loss: %f " % (loss.item(),crc_loss.item(), align_loss.item(), cps_loss.item(),quan_loss.item(),local_semi_mg_loss.item(), global_mg_loss))
                

                start_time = time.time()

            iteration = iteration + 1


            if  (iteration%niters_per_epoch) == 0:
                print('----------------- Save The Network ------------------------')
                with open(args.save + str(args.n_labeled) + '-semi_img_net.pkl', 'wb') as f:
                    torch.save(semi_img_net, f)
                with open(args.save + str(args.n_labeled) + '-semi_pt_net.pkl', 'wb') as f:
                    torch.save(semi_pt_net, f)
            

                extract(args)
                res = eval_func(1)
                if res['average'] > best:
                    best = res['average']
                    ans = res
                print(ans)
    return ans
def extract(args):
    if args.dataset == '3DMNIST':
        if args.quantization_type== 'swd':
            model_folder = '3DMNIST_swd/test_result'
        elif args.quantization_type== 'swdc':
            model_folder = '3DMNIST_swdc/test_result'
        if args.quantization_type== 'None':
            model_folder = '3DMNIST/test_result'    
                
    elif args.dataset == 'ModelNet10':
        if args.quantization_type== 'swd':
            model_folder = 'ModelNet10_swd/test_result'
        elif args.quantization_type== 'swdc':
            model_folder = 'ModelNet10_swdc/test_result'
        if args.quantization_type== 'None':
            model_folder = 'ModelNet10/test_result'    
        
    elif args.dataset == 'ModelNet40':
        if args.quantization_type== 'swd':
            model_folder = 'ModelNet40_swd/test_result'
        elif args.quantization_type== 'swdc':
            model_folder = 'ModelNet40_swdc/test_result'
        if args.quantization_type== 'None':
            model_folder = 'ModelNet40/test_result'    
            
    
    semi_img_net = torch.load('%s/%d-semi_img_net.pkl' % (args.save, args.n_labeled),
                         map_location=lambda storage, loc: storage)
    semi_img_net = semi_img_net.eval()
    semi_pt_net = torch.load('%s/%d-semi_pt_net.pkl' % (args.save, args.n_labeled),
                       map_location=lambda storage, loc: storage)
    semi_pt_net = semi_pt_net.eval()

    torch.cuda.empty_cache()
    #################################
    if args.dataset == '3DMNIST':
        test_set = FeatureDataloader(num_classes=10, partition='test')
    elif args.dataset == 'ModelNet40':
        labeled_modelnet,unlabeled_modelnet,test_set = get_modelnet(n_labeled=args.n_labeled, num_class=40)
    else:
        labeled_modelnet,unlabeled_modelnet,test_set = get_modelnet(n_labeled=args.n_labeled, num_class=10)
        
    data_loader_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=10)


    print('length of the dataset: ', len(test_set))
    #################################
    if args.dataset =='3DMNIST':
        img_feat_list = np.zeros((len(test_set), 256))
        pt_feat_list = np.zeros((len(test_set), 256))
        vis_img_feat_list = np.zeros((len(test_set), 2))
        vis_pt_feat_list = np.zeros((len(test_set), 2))
        tag_list = np.zeros((len(test_set)))
        label = np.zeros((len(test_set)))
        #################################
        iteration = 0
        for data in data_loader_loader:
            img_feat, pt_feat, ori_label = data
                
            img_feat = Variable(img_feat).to(torch.float32).to('cuda')
            pt_feat = Variable(pt_feat).to(torch.float32).to('cuda')
            ori_label = Variable(ori_label).to(torch.long).to('cuda')
            ##########################################
            semi_img_net = semi_img_net.to('cuda')
            _img_feat, sup_img_pred1 = semi_img_net(img_feat, step=1)
            semi_pt_net = semi_pt_net.to('cuda')
            _pt_feat, sup_pt_pred1 = semi_pt_net(pt_feat, step=1)

            img_feat_list[iteration, :] = _img_feat.data.cpu().numpy()
            pt_feat_list[iteration, :] = _pt_feat.data.cpu().numpy()
            label[iteration] = ori_label.data.cpu().numpy()
            iteration = iteration + 1

    else:
        img_feat_list = np.zeros((len(test_set), 256))
        pt_feat_list = np.zeros((len(test_set), 256))
        label = np.zeros((len(test_set)))
        #################################
        iteration = 0
        for data in data_loader_loader:
            pt, img1, img2,  target, target_vec = data
                
            img1 = Variable(img1).to(torch.float32).to('cuda')
            img2 = Variable(img2).to(torch.float32).to('cuda')
            pt = Variable(pt.permute(0,2,1)).to(torch.float32).to('cuda')
            target = Variable(target.squeeze(1)).to(torch.long).to('cuda')
            ##########################################
            semi_img_net = semi_img_net.to('cuda')
            _img_feat, sup_img_pred1 = semi_img_net(img1,img2, step=1)
            semi_pt_net = semi_pt_net.to('cuda')
            _pt_feat, sup_pt_pred1 = semi_pt_net(pt, step=1)
            
            img_feat_list[iteration, :] = _img_feat.data.cpu().numpy()
            pt_feat_list[iteration, :] = _pt_feat.data.cpu().numpy()
            label[iteration] = target.data.cpu().numpy()
            iteration = iteration + 1
            
    np.save(args.save + '/img_feat', img_feat_list)
    np.save(args.save + '/pt_feat', pt_feat_list)
    np.save(args.save + '/label', label)

def fx_calc_map_label(view_1, view_2, label_test):
    dist = scipy.spatial.distance.cdist(view_1, view_2, 'cosine')  # rows view_1 , columns view 2
    ord = dist.argsort()
    numcases = dist.shape[0]
    res = []
    for i in range(numcases):
        order = ord[i]
        p = 0.0
        r = 0.0
        for j in range(numcases):
            if label_test[i] == label_test[order[j]]:
                r += 1
                p += (r / (j + 1))
        if r > 0:
            res += [p / r]
        else:
            res += [0]
    return np.mean(res)

def eval_func(img_pairs):
    print('number of img views: ', img_pairs)
    img_feat = np.load(args.save + '/img_feat.npy')
    pt_feat = np.load(args.save + '/pt_feat.npy')
    label = np.load(args.save + '/label.npy')
    ########################################
    img_test = normalize(img_feat, norm='l1', axis=1)
    pt_test = normalize(pt_feat, norm='l1', axis=1)
    ########################################
    par_list = [
        (img_test, pt_test, 'Image2Pt'),
        (pt_test, img_test, 'Pt2Image')]
    ########################################
    
    name1 = ['Image2Pt', 'Pt2Image']
    res ={}
    avg_acc=0
    for index in range(2):
        view_1, view_2, name = par_list[index]
        print(name + '---------------------------')
        acc = fx_calc_map_label(view_1, view_2, label)
        
        acc_round = round(acc * 100, 2)
        print(str(acc_round))
        avg_acc+=acc
        res[name1[index]] = str(acc_round)
    avg_acc = round(avg_acc*100/2,2)
    res['average'] = avg_acc
    print('average---------------------------')
    print(str(avg_acc))         

    return res
if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Cross Modal Retrieval for Point Cloud, Mesh, and Image Models')
    
    parser.add_argument('--weight_crc', type=float, default=1, metavar='weight_crc', help='weight crc' ) #20 - 10   50

    parser.add_argument('--weight_align', type=float, default=1, metavar='weight_align', help='weight align' )
    
    parser.add_argument('--cps_weight', type=float, default=1, help='cps loss ratio')
    
    parser.add_argument('--quantization_type', type=str, default='swdc', choices=['swdc','None'])
    
    parser.add_argument('--gpu_id', type=str,  default='1', help='GPU used to train the network')
    
    parser.add_argument('--save', type=str,  default='./checkpoints/3DMNIST_swdc/vis_codes/', help='path to save the final model')

    parser.add_argument('--dataset', type=str, default='3DMNIST', metavar='dataset',choices=['ModelNet10', 'ModelNet40','3DMNIST'], help='ModelNet10 or ModelNet40')

    parser.add_argument('--num_classes', type=int, default=10, metavar='num_classes',help='10 or 40')

    parser.add_argument('--n_labeled', type=int, default=400, metavar='n_labeled', help='number of labeled data in the dataset')

    parser.add_argument('--batch_size', type=int, default=50, metavar='batch_size',  help='Size of batch)')

    parser.add_argument('--epochs', type=int, default=50, metavar='N', help='number of episode to train 100')
    
    parser.add_argument('--quantization_weight', type=float, default=0.1, help='quantization loss ratio')
    
    parser.add_argument('--dropout', type=str, default=0.4, metavar='dropout', help='dropout')
    #optimizer
    parser.add_argument('--lr_img', type=float, default=5e-5, metavar='LR',  help='learning rate (default: 0.001, 0.1 if using sgd)')
    
    parser.add_argument('--lr_pt', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    
    parser.add_argument('--lr_step', type=int,  default=50,
                        help='how many iterations to decrease the learning rate')
    
    parser.add_argument('--weight_decay', type=float, default=1e-5, metavar='weight_decay',
                        help='learning rate (default: 1e-3)')

    parser.add_argument('--per_print', type=int,  default=100,
                        help='how many iterations to print the loss and accuracy')
                        
    args = parser.parse_args()
    print('weight_crc:', args.weight_crc)
    print('weight_align:', args.weight_align)
    print('cps_weight:', args.cps_weight)
    print('quantization_type:', args.quantization_type)
    print('save:', args.save)
    print('dataset:', args.dataset)
    print('num_classes:', args.num_classes)
    print('n_labeled:', args.n_labeled)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    torch.backends.cudnn.enabled = True
    ans = training(args)
    print(ans)
    # torch.backends.cudnn.enabled = False
    # if not os.path.exists(args.save):
    #     os.mkdir(args.save)

    # extract(args)
    # res = eval_func(1)
