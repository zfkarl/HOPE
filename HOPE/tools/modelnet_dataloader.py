import os
import sys
import glob
import h5py
import json
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms, datasets
import torch
import numpy as np

# from tools.visualize import showpoints

def load__modelnet40_data(partition,dataset_dir):
    all_data = []
    all_label = []
    img_lst = []
    for h5_name in glob.glob(os.path.join(dataset_dir, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        split = h5_name[-4]
        jason_name = dataset_dir+'modelnet40_ply_hdf5_2048/ply_data_' + partition +'_' + split + '_id2file.json'

        with open(jason_name) as json_file:
            images = json.load(json_file)        
        img_lst = img_lst + images
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0) # DEBUG:
    #print(len(all_data), len(all_label), len(img_lst))
    img_lst = np.array(img_lst)
    #print('img_lst:',img_lst.shape)
    return all_data, all_label, img_lst


def load_modelnet10_data(partition,dataset_dir):
    all_data = []
    all_label = []
    img_lst = []
    for h5_name in glob.glob(os.path.join(dataset_dir, 'modelnet10_hdf5_2048', '%s*.h5'%partition)):
        split = h5_name[-4]
        jason_name = dataset_dir+'modelnet10_hdf5_2048/'+partition + split + '_id2file.json'
        with open(jason_name) as json_file:
            images = json.load(json_file)
        img_lst = img_lst + images
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0) # DEBUG:
    #print(len(all_data), len(all_label), len(img_lst))
    img_lst = np.array(img_lst)
    return all_data, all_label, img_lst



def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

def rotate_pointcloud(pointcloud):
    theta = np.pi*2 * np.random.rand()
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    pointcloud[:,[0,2]] = pointcloud[:,[0,2]].dot(rotation_matrix) # random rotation (x,z)
    return pointcloud

def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud

def random_scale(pointcloud, scale_low=0.8, scale_high=1.25):
    N, C = pointcloud.shape
    scale = np.random.uniform(scale_low, scale_high)
    pointcloud = pointcloud*scale
    return pointcloud

class ModelNet(Dataset):
    def __init__(self, dataset, num_points, num_classes, dataset_dir, partition='train'):
        self.dataset = dataset
        self.dataset_dir = dataset_dir

        if self.dataset == 'ModelNet40':
            self.data, self.label, self.img_lst = load__modelnet40_data(partition,self.dataset_dir)
        else:
            self.data, self.label, self.img_lst = load_modelnet10_data(partition,self.dataset_dir)
        self.num_points = num_points
        self.partition = partition
        self.num_classes=num_classes

        self.img_train_transform = transforms.Compose([
            transforms.RandomCrop(224),
            transforms.Resize(112),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.img_test_transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.Resize(112),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def get_modelnet40_data(self, item):

        # Get Image Data first
        names = self.img_lst[item]
        names = names.split('/')
        
        #randomly select one image from the 12 images for each object
        img_idx = random.randint(0, 179)
        img_names =self.dataset_dir+'ModelNet40-Images-180/%s/%s/%s.%d.png' % (names[0], names[1][:-4], names[1][:-4], img_idx)
        img = Image.open(img_names).convert('RGB')

        img_idx2 = random.randint(0, 179)
        while img_idx == img_idx2:
            img_idx2 = random.randint(0, 179)

        img_name2 = self.dataset_dir+'ModelNet40-Images-180/%s/%s/%s.%d.png' % (names[0], names[1][:-4], names[1][:-4], img_idx2)
        img2 = Image.open(img_name2).convert('RGB')

        label = self.label[item]

        pointcloud = self.data[item]
        choice = np.random.choice(len(pointcloud), self.num_points, replace=True)
        pointcloud = pointcloud[choice, :]

        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            pointcloud = rotate_pointcloud(pointcloud)
            pointcloud = jitter_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)

            img = self.img_train_transform(img)
            img2 = self.img_train_transform(img2)
        else:
            img = self.img_test_transform(img)
            img2 = self.img_test_transform(img2)

        return pointcloud, label, img, img2

    def get_modelnet10_data(self, item):

        # Get Image Data first
        names = self.img_lst[item]
        names = names.split('/')
        
        #randomly select one image from the 12 images for each object
        img_idx = random.randint(0, 179)
        img_names =self.dataset_dir+'ModelNet10-Images-180/%s/%s/%s.%d.png' % (names[0], names[1][:-4], names[1][:-4], img_idx)
        img = Image.open(img_names).convert('RGB')

        img_idx2 = random.randint(0, 179)
        while img_idx == img_idx2:
            img_idx2 = random.randint(0, 179)

        img_name2 = self.dataset_dir+'ModelNet10-Images-180/%s/%s/%s.%d.png' % (names[0], names[1][:-4], names[1][:-4], img_idx2)
        img2 = Image.open(img_name2).convert('RGB')

        label = self.label[item]

        pointcloud = self.data[item]
        choice = np.random.choice(len(pointcloud), self.num_points, replace=True)
        pointcloud = pointcloud[choice, :]

        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            pointcloud = rotate_pointcloud(pointcloud)
            pointcloud = jitter_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)

            img = self.img_train_transform(img)
            img2 = self.img_train_transform(img2)
        else:
            img = self.img_test_transform(img)
            img2 = self.img_test_transform(img2)

        return pointcloud, label, img, img2
    

    def __getitem__(self, item):
        

        if self.dataset == 'ModelNet40':
            pt, target, img, img_v = self.get_modelnet40_data(item)
        else:
            pt, target, img, img_v = self.get_modelnet10_data(item)
            
        pt = torch.from_numpy(pt)
        target_vec = np.zeros((1, self.num_classes))
        target_vec[0, target] = 1

        return pt, img, img_v,  target, target_vec

    def __len__(self):
        return self.data.shape[0]
    
    
    
    
    
class modelnet_labeled(ModelNet):

    def __init__(self,indexs=None,dataset='ModelNet40',partition='train'):
        self.dataset = dataset
        if self.dataset =='ModelNet40':
            num_classes=40
            dataset_dir='/home/zf/dataset/modelnet/modelnet40/'
        elif self.dataset == 'ModelNet10':
            num_classes=10
            dataset_dir='/home/zf/dataset/modelnet/modelnet10/'
        else:
            raise ValueError("Dataset must be either ModelNet10 or ModelNet40!")
        
        super(modelnet_labeled, self).__init__(dataset=dataset, num_points=1024, num_classes=num_classes, dataset_dir=dataset_dir, partition=partition)

        if indexs is not None:
            self.img_lst = self.img_lst[indexs]
            self.data = self.data[indexs]
            self.label = self.label[indexs]

    def __getitem__(self, index):
        if self.dataset == 'ModelNet40':
            pt, target, img, img_v = self.get_modelnet40_data(index)
        else:
            pt, target, img, img_v = self.get_modelnet10_data(index)
            
        pt = torch.from_numpy(pt)
        target_vec = np.zeros((1, self.num_classes))
        target_vec[0, target] = 1

        return pt, img, img_v,  target, target_vec


class modelnet_unlabeled(modelnet_labeled):

    def __init__(self, indexs,dataset,partition='train'):
        super(modelnet_unlabeled, self).__init__(indexs, dataset, partition=partition)
        
        self.label = np.array([-1 for i in range(len(self.label))])
        
    
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
    
    
def get_modelnet(n_labeled, num_class):
    if num_class == 40:
        dataset = "ModelNet40"
        base_dataset = ModelNet(dataset = 'ModelNet40', num_points = 1024, num_classes=40, dataset_dir='/home/zf/dataset/modelnet/modelnet40/',  partition='train')
    elif num_class == 10:
        dataset = "ModelNet10"
        base_dataset = ModelNet(dataset = 'ModelNet10', num_points = 1024, num_classes=10, dataset_dir='/home/zf/dataset/modelnet/modelnet10/',  partition='train')
        
        
    train_labeled_idxs, train_unlabeled_idxs = train_unlabeled_split(base_dataset.label, num_class, int(n_labeled/num_class))

    train_labeled_dataset = modelnet_labeled(train_labeled_idxs,dataset,'train')
    
    train_unlabeled_dataset = modelnet_labeled(train_unlabeled_idxs,dataset,'train')
    
    test_dataset = modelnet_labeled(indexs=None,dataset=dataset,partition='test')

    print (f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_idxs)} #test: {len(test_dataset)}")
    return train_labeled_dataset, train_unlabeled_dataset, test_dataset
   
if __name__ == '__main__':
    train_set = ModelNet(dataset = 'ModelNet40', num_points = 1024, num_classes=40, dataset_dir='/home/zf/dataset/modelnet/modelnet40/',  partition='test')
    data_loader_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True,num_workers=12)
    
    # for data in data_loader_loader:
    #     pt, img, img_v, target, target_vec = data
    #     print('pt:', pt.shape)
    #     print('img:', img.shape)
    #     print('img_v:', img_v.shape)
    #     print('target:', target.shape)
    #     print('target_vec:', target_vec.shape)
    
    
    labeled_modelnet,unlabeled_modelnet,test_modelnet = get_modelnet(400,10)
    print(len(labeled_modelnet))
    print(len(unlabeled_modelnet))
    print(len(test_modelnet))
    # print('labeled_modelnet:',labeled_modelnet.label[0:10])
    # print('unlabeled_modelnet:', unlabeled_modelnet.label[0:10])
    # print('test_modelnet:', test_modelnet.label[0:10])