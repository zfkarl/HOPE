from torch.utils.data import Dataset
from torchvision import transforms, datasets
import numpy as np
import h5py
from torchvision import transforms
import torch

class FeatureDataloader(Dataset):
    def __init__(self, num_classes=10, partition='train'):
        self.num_classes = num_classes
        if partition == 'train':
            self.img_feat = np.load('datasets/3D_MNIST/train_img_feat.npy')
            self.pt_feat = np.load('datasets/3D_MNIST/train_pt_feat.npy')  
            #self.label = np.load('datasets/3D_MNIST/train_label_60.npy')
            self.ori_label = np.load('datasets/3D_MNIST/train_ori_label.npy')
        if partition == 'test':
            self.img_feat = np.load('./datasets/3D_MNIST/test_img_feat.npy')
            self.pt_feat = np.load('./datasets/3D_MNIST/test_pt_feat.npy')        
            self.ori_label = np.load('./datasets/3D_MNIST/test_ori_label.npy')
            
    def __getitem__(self, item):
        img_feat =  self.img_feat[item]
        pt_feat =  self.pt_feat[item]
        #label = self.label[item]
        label = self.ori_label[item]

        return img_feat, pt_feat, label
    
    def __len__(self):
        return self.ori_label.shape[0]

# class FeatureDataloader_ori(Dataset):
#     def __init__(self, num_classes=10, partition='train'):

#         self.num_classes = num_classes
#         if partition == 'train':
#             self.img_feat = np.load('datasets/3D_MNIST/mnist3d/ori_version/train_imgs_ori.npy')
#             self.pt_feat = np.load('datasets/3D_MNIST/mnist3d/ori_version/train_points_ori.npy')  
#             self.ori_label = np.load('datasets/3D_MNIST/mnist3d/ori_version/train_labels_ori.npy')
#         if partition == 'test':
#             self.img_feat = np.load('./datasets/3D_MNIST/mnist3d/ori_version/test_imgs_ori.npy')
#             self.pt_feat = np.load('./datasets/3D_MNIST/mnist3d/ori_version/test_points_ori.npy')        
#             self.ori_label = np.load('./datasets/3D_MNIST/mnist3d/ori_version/test_labels_ori.npy')
        
            
#     def __getitem__(self, item):
#         img_feat =  self.img_feat[item]
#         pt_feat =  self.pt_feat[item]
#         #label = self.label[item]
#         label = self.ori_label[item]

#         return img_feat, pt_feat, label
    
#     def __len__(self):
#         return self.ori_label.shape[0]

class mnist3d_labeled(FeatureDataloader):

    def __init__(self,indexs=None,partition='train'):
        super(mnist3d_labeled, self).__init__(num_classes=10, partition=partition)

        if indexs is not None:
            self.img_feat = self.img_feat[indexs]
            self.pt_feat = self.pt_feat[indexs]
            self.ori_label = np.array(self.ori_label)[indexs]

    def __getitem__(self, index):
        img_feat =  self.img_feat[index]
        pt_feat =  self.pt_feat[index]
        label = self.ori_label[index]

        return img_feat, pt_feat, label

class mnist3d_unlabeled(mnist3d_labeled):

    def __init__(self, indexs,partition='train'):
        super(mnist3d_unlabeled, self).__init__(indexs, partition=partition)
        self.ori_label = np.array([-1 for i in range(len(self.ori_label))])
        
    
##split train set into semi-supervised setting
def train_unlabeled_split(labels, num_class,n_labeled_per_class):
    SEED = 42
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []

    for i in range(num_class):
        idxs = np.where(labels == i)[0]
        np.random.seed(SEED)
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:n_labeled_per_class])
        train_unlabeled_idxs.extend(idxs[n_labeled_per_class:])
    np.random.seed(SEED)
    np.random.shuffle(train_labeled_idxs)
    np.random.seed(SEED)
    np.random.shuffle(train_unlabeled_idxs)

    return train_labeled_idxs, train_unlabeled_idxs
    
    
def get_3dmnist(n_labeled, num_class):

    base_dataset = FeatureDataloader(10,'train')
    train_labeled_idxs, train_unlabeled_idxs = train_unlabeled_split(base_dataset.ori_label, num_class, int(n_labeled/num_class))

    train_labeled_dataset = mnist3d_labeled(train_labeled_idxs,'train')
    train_unlabeled_dataset = mnist3d_unlabeled(train_unlabeled_idxs,'train')
    test_dataset = mnist3d_labeled(indexs=None,partition='test')

    print (f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_idxs)} #test: {len(test_dataset)}")
    return train_labeled_dataset, train_unlabeled_dataset, test_dataset

if __name__ == '__main__':
    mnist3d_train = FeatureDataloader(10,'train')
    mnist3d_test = FeatureDataloader(10,'test')
    print('train_dataset length:',len(mnist3d_train))
    print('mnist3d img_feat:',mnist3d_train.img_feat.shape)
    print('mnist3d pt_feat:',mnist3d_train.pt_feat.shape)
    print('mnist3d ori_label:',mnist3d_train.ori_label.shape)
    print('test_dataset length:',len(mnist3d_test))
    print('mnist3d img_feat:',mnist3d_test.img_feat.shape)
    print('mnist3d pt_feat:',mnist3d_test.pt_feat.shape)
    print('mnist3d ori_label:',mnist3d_test.ori_label.shape)
    #import torch
    #from torch.autograd import Variable
    #data_loader_loader = torch.utils.data.DataLoader(mnist3d, batch_size=128, shuffle=True, num_workers=10, drop_last=False,pin_memory=True)
    
    # labeled_mnist3d,unlabeled_mnist3d,test_mnist3d = get_3dmnist(400,10)
    # print('labeled_mnist3d:',labeled_mnist3d.ori_label[0:10])
    # print('unlabeled_mnist3d:',unlabeled_mnist3d.ori_label[0:10])
    # print('test_mnist3d:',test_mnist3d.ori_label[0:10])
    # import torch
    # from torch.autograd import Variable
    # data_loader_loader = torch.utils.data.DataLoader(mnist3d, batch_size=128, shuffle=True, num_workers=10, drop_last=False)
    # for data in data_loader_loader:
    #     img_feat, pt_feat, ori_label = data
    #     img_feat = Variable(img_feat).to(torch.float32).to('cuda')
    #     pt_feat = Variable(pt_feat).to(torch.float32).to('cuda')
    #     ori_label = Variable(ori_label).to(torch.long).to('cuda')
    #     print('mnist3d img_feat:',img_feat.shape) ##[bsz,30,30]  length of training set: 5000
    #     print('mnist3d pt_feat:',pt_feat.shape)   ##[bsz,2048]
    #     print('mnist3d ori_label:',ori_label.shape) ##[bsz]