
import numpy as np
import os
import torch
from torch.autograd import Variable
import argparse
import numpy as np
from sklearn.preprocessing import normalize
import scipy
from tools.test_mnist_feature_loader import FeatureDataloader
from tools.modelnet_dataloader import get_modelnet


def extract(args):
    if args.dataset == '3DMNIST':
        model_folder = 'RONO/3DMNIST/vis_codes'
        
    elif args.dataset == 'ModelNet10':
        model_folder = 'RONO/ModelNet10/vis_codes'
        
    elif args.dataset == 'ModelNet40':
        model_folder = 'RONO/ModelNet40/test_result'

    img_net = torch.load('./checkpoints/%s/%d-%d-img_net.pkl' % (model_folder, args.n_labeled,args.iterations),
                        map_location=lambda storage, loc: storage)
    img_net = img_net.eval()
    pt_net = torch.load('./checkpoints/%s/%d-%d-pt_net.pkl' % (model_folder, args.n_labeled,args.iterations),
                    map_location=lambda storage, loc: storage)
    pt_net = pt_net.eval()
    head_net = torch.load('./checkpoints/%s/%d-%d-head_net.pkl' % (model_folder, args.n_labeled,args.iterations),
                    map_location=lambda storage, loc: storage)
    head_net = head_net.eval()
        
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
        label = np.zeros((len(test_set)))
        #################################
        iteration = 0
        for data in data_loader_loader:
            img_feat, pt_feat, ori_label = data
                
            img_feat = Variable(img_feat).to(torch.float32).to('cuda')
            pt_feat = Variable(pt_feat).to(torch.float32).to('cuda')
            ori_label = Variable(ori_label).to(torch.long).to('cuda')
            ##########################################
            img_net = img_net.to('cuda')
            _img_feat = img_net(img_feat)
            pt_net = pt_net.to('cuda')
            _pt_feat = pt_net(pt_feat)

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
            img_net = img_net.to('cuda')
            _img_feat = img_net(img1,img2)
            pt_net = pt_net.to('cuda')
            _pt_feat = pt_net(pt)
            
            img_feat_list[iteration, :] = _img_feat.data.cpu().numpy()
            pt_feat_list[iteration, :] = _pt_feat.data.cpu().numpy()
            label[iteration] = target.data.cpu().numpy()
            iteration = iteration + 1

    np.save('%s/%d-img_feat' % (args.save, args.n_labeled), img_feat_list)
    np.save('%s/%d-pt_feat' % (args.save, args.n_labeled), pt_feat_list)
    np.save('%s/%d-label' % (args.save, args.n_labeled), label)
    
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
    img_feat = np.load('%s/%d-img_feat.npy' % (args.save, args.n_labeled))
    pt_feat = np.load('%s/%d-pt_feat.npy' % (args.save, args.n_labeled))
    label = np.load('%s/%d-label.npy' % (args.save, args.n_labeled))
    ########################################
    img_test = normalize(img_feat, norm='l1', axis=1)
    pt_test = normalize(pt_feat, norm='l1', axis=1)
    ########################################
    par_list = [
        (img_test, pt_test, 'Image2Pt'),
        (pt_test, img_test, 'Pt2Image')]
    ########################################

    avg_acc=0
    for index in range(2):
        view_1, view_2, name = par_list[index]
        print(name + '---------------------------')
        acc = fx_calc_map_label(view_1, view_2, label)
        
        acc_round = round(acc * 100, 2)
        print(str(acc_round))
        avg_acc+=acc
    avg_acc = round(avg_acc*100/2,2)
    print('average---------------------------')
    print(str(avg_acc))  
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cross Modal Retrieval for Point Cloud, Mesh, and Image Models')

    parser.add_argument('--dataset', type=str, default='3DMNIST', metavar='dataset',choices=['3DMNIST', 'ModelNet40', 'ModelNet10'],
                        help='ModelNet10 or ModelNet40')
    
    parser.add_argument('--dataset_dir', type=str, default='./datasets/',
                        metavar='dataset_dir', help='dataset_dir')

    parser.add_argument('--n_labeled', type=int, default = 800, metavar='n_labeled',
                        help='number of labeled data in the dataset')
    # you can modify this root to run provided checkpoints.
    # parser.add_argument('--model_folder', type=str, default='3DMNIST/40', help='path to load model')

    parser.add_argument('--iterations', type=int, default = 800, help='iteration to load the model')

    parser.add_argument('--gpu_id', type=str, default='2', help='GPU used to train the network')

    parser.add_argument('--save', type=str, default='extracted_features/3DMNIST/vis_codes/rono', help='save features')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    torch.backends.cudnn.enabled = False

    if not os.path.exists(args.save):
        os.mkdir(args.save)

    extract(args)
    eval_func(1)