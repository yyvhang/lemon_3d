import json
import numpy as np
import random
import torch
import pdb
import os
import cv2
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tools.utils.mesh_sampler import get_sample

mesh_sampler = get_sample(device=None)

def img_normalize(img):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ])
    img = transform(img)
    return img

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

class _3DIR(Dataset):
    def __init__(self, img_file, obj_file, human_file, behave_file, mode):
        super(_3DIR).__init__()

        self.mode = mode
        self.img_files = self.read_file(img_file)
        self.data_dict = json.load(open(human_file, 'r'))
        # self.behave_dict = json.load(open(behave_file, 'r'))

        if mode == 'train':
            number_dict = {'Earphone':0, 'Baseballbat':0, 'Tennisracket':0, 'Bag':0, 'Motorcycle':0, 'Guitar':0, 
                        'Backpack':0, 'Chair':0, 'Knife':0, 'Bicycle':0, 'Umbrella':0, 'Keyboard':0,
                        'Scissors':0, 'Bottle':0, 'Bowl':0, 'Surfboard':0,  'Mug':0, 'Suitcase':0, 'Vase':0, 
                        'Skateboard':0, 'Bed':0}

            self.obj_files, self.number_dict = self.read_file(obj_file, number_dict=number_dict)
            self.obj_list = list(number_dict.keys())
            self.pts_split = {}
            start_index = 0
            for obj_ in self.obj_list:
                temp_split = [start_index, start_index + self.number_dict[obj_]]
                self.pts_split[obj_] = temp_split
                start_index += self.number_dict[obj_]
        else:
            self.obj_files = self.read_file(obj_file)

        self.affordance_list = ['grasp', 'lift', 'open', 'lay', 'sit', 'support', 'wrapgrasp', 'pour', 
                        'move', 'pull', 'listen', 'press', 'cut', 'stab', 'ride', 'play', 'carry']
        
        self.img_size = (224, 224)
        self.Hm_curvaure_folder = 'Data/Curvature/Human'
        self.Obj_curvaure_folder = 'Data/Curvature/Object'
        # self.Behave_curvaure_folder = 'Data/Curvature/Behave'
        self.Spatial_folder = 'Data/obj_center'

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        
        data_info = {}
        img_path = self.img_files[index]
        data_type = img_path.split('/')[1]
        object_ = img_path.split('/')[-1].split('_')[0]

        if(data_type != 'Behave'):
            mask_path = img_path.split('/')[0] + '/mask/' + img_path.split('/')[2] + '/' + img_path.split('/')[-1].split('.')[0] + '.png'
        else:
            mask_path = img_path.replace('Images','mask').replace('.jpg','.png')
        Img = self.mask_img(img_path, mask_path)

        if(data_type != 'Behave'):
            key_dict = os.path.join(*img_path.split('/')[1:])
            smplh_param = self.data_dict[key_dict]['smplh_param']

            contact_path = img_path.split('.')[0]
            contact_path = contact_path.split('/')
            contact_path[1] = 'smplh_contact_pkl'
            contact_path[-1] = contact_path[-1] + '.pkl'
            contact_label = os.path.join(*contact_path)

            sphere_center = os.path.join(self.Spatial_folder, img_path.split('/')[2], img_path.split('/')[-1].replace('.jpg','.pkl'))

            hm_curvature_path = os.path.join(self.Hm_curvaure_folder, object_, img_path.split('/')[-1].replace('.jpg','.pkl'))
            hm_curvature = np.load(hm_curvature_path, allow_pickle=True)
            hm_curvature = torch.from_numpy(hm_curvature).to(torch.float32)
            hm_curvature = mesh_sampler.downsample(hm_curvature)
        else:
            key_dict = img_path.split('/')[-1].split('_')[:-1]
            key_ = '_'.join(key_dict)
            smplh_param = self.behave_dict[key_]
            
            contact_folder = img_path.replace('Images','Contact').split('/')[:4]
            contact_label = '/'.join(contact_folder) + '/' + key_ + '_contact.pkl'

            sphere_folder = img_path.replace('Images','obj_center').split('/')[:4]
            sphere_center = '/'.join(sphere_folder) + '/' + key_ + '.pkl'

            seq_id = img_path.split('/')[3]
            hm_curvature_path = os.path.join(self.Behave_curvaure_folder, seq_id, key_ + '.pkl')
            hm_curvature = np.load(hm_curvature_path, allow_pickle=True)
            hm_curvature = torch.from_numpy(hm_curvature).to(torch.float32)
            hm_curvature = mesh_sampler.downsample(hm_curvature)

        smplh_param['shape'] = torch.tensor(smplh_param['shape'])
        smplh_param['transl'] = torch.tensor(smplh_param['transl'])
        smplh_param['body_pose'] = torch.tensor(smplh_param['body_pose']).reshape(21, 3, 3)
        smplh_param['left_hand_pose'] = torch.tensor(smplh_param['left_hand_pose']).reshape(15, 3, 3)
        smplh_param['right_hand_pose'] = torch.tensor(smplh_param['right_hand_pose']).reshape(15, 3, 3)
        smplh_param['global_orient'] = torch.tensor(smplh_param['global_orient']).reshape(1, 3, 3)
        data_info['human'] = smplh_param


        contact_label = np.load(contact_label, allow_pickle=True)
        contact_label = torch.from_numpy(contact_label).unsqueeze(dim=1).to(torch.float32)
        contact_label_mid = mesh_sampler.downsample(contact_label)
        contact_label_corase = mesh_sampler.downsample(contact_label, n1=0, n2=2)

        sphere_center_pts = np.load(sphere_center, allow_pickle=True)
        sphere_center_pts = torch.from_numpy(sphere_center_pts).to(torch.float32).squeeze()

        Img = Img.resize(self.img_size)
        Img_data = img_normalize(Img)
        data_info['img'] = Img_data
        data_info['img_path'] = img_path
        contact_dict = {'contact_fine': contact_label, 'contact_mid': contact_label_mid, 'contact_coarse': contact_label_corase}

        #contact_label
        data_info['contact'] = contact_dict

        #hm_curvature
        data_info['hm_curvature'] = hm_curvature

        #sphere_center
        data_info['sphere_center'] = sphere_center_pts

        #mask
        mvm_mask = np.ones((1723,1))
        mvm_percent = 0.2
        if self.mode == 'train':
            num_vertices = 1723
            pb = np.random.random_sample()
            masked_num = int(pb * mvm_percent * num_vertices) # at most x% of the vertices could be masked
            indices = np.random.choice(np.arange(num_vertices),replace=False,size=masked_num)
            mvm_mask[indices,:] = 0.0
        mvm_mask = torch.from_numpy(mvm_mask).float()
        data_info['mvm_mask'] = mvm_mask

        #obj_pts& affordance label

        if self.mode == 'train':
            Pts = []
            affordance_ = []
            Pts_path = []
            affordance_logits = []
            obj_range = self.pts_split[object_]
            obj_curvatures = []
            point_sample_idx = random.sample(range(obj_range[0], obj_range[1]), 1)
            for idx in point_sample_idx:
                point_path = self.obj_files[idx]
                obj_curvature_path = os.path.join(self.Obj_curvaure_folder, object_, point_path.split('/')[-1].replace('.txt','.pkl'))
                obj_curvature = np.load(obj_curvature_path, allow_pickle=True)
                obj_curvature = torch.from_numpy(obj_curvature).to(torch.float32).unsqueeze(dim=-1)
                obj_curvatures.append(obj_curvature)

                Points, affordance_label = self.extract_point_file(point_path)
                Points = pc_normalize(Points)
                Points = Points.transpose()
                affordance_label, affordance_index = self.get_affordance_label(img_path, affordance_label)
                Pts.append(Points)
                affordance_.append(affordance_label)
                Pts_path.append(point_path)
                affordance_logits.append(affordance_index)
        else:
            Pts_path = self.obj_files[index]
            obj_curvature_path = os.path.join(self.Obj_curvaure_folder, object_, Pts_path.split('/')[-1].replace('.txt','.pkl'))
            obj_curvature = np.load(obj_curvature_path, allow_pickle=True)
            obj_curvatures = torch.from_numpy(obj_curvature).to(torch.float32).unsqueeze(dim=-1)

            Pts, affordance_label = self.extract_point_file(Pts_path)
            Pts = pc_normalize(Pts)
            Pts = Pts.transpose()
            affordance_, affordance_logits = self.get_affordance_label(img_path, affordance_label)

        data_info['Pts'] = Pts
        data_info['aff_gt'] = affordance_
        data_info['Pts_path'] = Pts_path
        data_info['logits'] = affordance_logits
        data_info['obj_curvature'] = obj_curvatures
        return data_info

    def mask_img(self, img_path, mask_path):
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('RGB')
        img, mask = np.asarray(img), np.asarray(mask)
        back_ground = np.array([0, 0, 0])
        mask_bi = np.all(mask == back_ground, axis=2)
        mask_img = np.ones_like(mask)
        mask_img[mask_bi] = back_ground
        masked_img = img * mask_img
        masked_img = Image.fromarray(masked_img)
        return masked_img

    def read_file(self, path, number_dict=None):
        file_list = []
        with open(path,'r') as f:
            files = f.readlines()
            for file in files:
                file = file.strip('\n')
                if number_dict != None:
                    object_ = file.split('/')[2]
                    number_dict[object_] += 1
                file_list.append(file)

            f.close()
        if number_dict != None:
            return file_list, number_dict
        else:
            return file_list
            
    def extract_point_file(self, path):
        with open(path,'r') as f:
            coordinates = []
            lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            line = line.strip(' ')
            data = line.split(' ')
            coordinate = [float(x) for x in data]
            coordinates.append(coordinate)
        data_array = np.array(coordinates)
        points_coordinates = data_array[:, 0:3]
        affordance_label = data_array[: , 3:]

        return points_coordinates, affordance_label

    def get_affordance_label(self, str, label):

        obj = str.split('/')[-1].split('_')[0]
        affordance = str.split('/')[-1].split('_')[1]
        index = self.affordance_list.index(affordance)
        label = label[:, index]
        
        return label, index


if __name__=='__main__':
    pass
