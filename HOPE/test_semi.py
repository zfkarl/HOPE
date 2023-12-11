
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
        
    semi_img_net = torch.load('./checkpoints/%s/%d-semi_img_net.pkl' % (model_folder, args.n_labeled),
                         map_location=lambda storage, loc: storage)
    semi_img_net = semi_img_net.eval()
    semi_pt_net = torch.load('./checkpoints/%s/%d-semi_pt_net.pkl' % (model_folder, args.n_labeled),
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

    parser.add_argument('--dataset', type=str, default='3DMNIST', metavar='dataset',choices=['3D_MNIST', 'ModelNet40', 'ModelNet10'],
                        help='ModelNet10 or ModelNet40')
    
    parser.add_argument('--quantization_type', type=str, default='swd', choices=['ot','swd','swdc','None'])
    
    parser.add_argument('--n_labeled', type=int, default = 200, metavar='n_labeled',
                        help='number of labeled data in the dataset')

    parser.add_argument('--iterations', type=int, default = 4801, help='iteration to load the model')

    parser.add_argument('--gpu_id', type=str, default='3', help='GPU used to train the network')

    parser.add_argument('--save', type=str, default='extracted_features/3D_MNIST/test_result', help='save features')
    
    parser.add_argument('--dataset_dir', type=str, default='./datasets/',
                        metavar='dataset_dir', help='dataset_dir')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    torch.backends.cudnn.enabled = False

    if not os.path.exists(args.save):
        os.mkdir(args.save)

    extract(args)
    eval_func(1)