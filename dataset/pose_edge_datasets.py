import os
import numpy as np
from ipdb import set_trace
import random
import json
import torch
import cv2
import json
from torch.utils import data
import torchvision.transforms as transforms

from dataset.target_generation import generate_edge, generate_pose, flip_joints
from utils.transforms import get_affine_transform
import datetime

class LIPDataSet(data.Dataset):
    def __init__(self, root, pose_anno_file, dataset, crop_size=[473, 473], scale_factor=0.25, dataset_list='_id.txt', pose_net_stride=4, sigma=7,
                 rotation_factor=30, ignore_label=255, transform=None):
        """
        :rtype:
        """
        self.root = root
        self.aspect_ratio = crop_size[1] * 1.0 / crop_size[0]
        self.crop_size = np.asarray(crop_size)
        self.ignore_label = ignore_label
        self.scale_factor = scale_factor
        self.rotation_factor = rotation_factor
        self.flip_prob = 0.5
        self.pose_net_stride = pose_net_stride
        self.sigma = sigma
        self.flip_pairs = [[0, 5], [1, 4], [2, 3], [11, 14], [12, 13], [10, 15]]
        self.transform = transform
        self.dataset = dataset
        self.dataset_list = dataset_list

        list_path = os.path.join(self.root, self.dataset + self.dataset_list)

        self.im_list = [i_id.strip() for i_id in open(list_path)]
        self.number_samples = len(self.im_list)

        # get pose data
        train_list = []
        self.pose_info = {}
        with open(pose_anno_file, 'r') as data_file:
            data_this = json.load(data_file)
            data_this = data_this['root']
            train_list = train_list + data_this

            for img in data_this:
                img_name = img['im_name'].split('.')[0]
                pose_anno = img['joint_self']
                self.pose_info[img_name] = pose_anno

        self.pose_anno_list = train_list
        self.N_train = len(self.pose_anno_list)


    def __len__(self):
        return self.number_samples

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w * 1.0, h * 1.0], dtype=np.float32)

        return center, scale

    def __getitem__(self, index):
        # Load training image
        # im_name = self.im_list[index]
        if self.dataset == 'train' or self.dataset == 'val':
            #  train_item = self.pose_anno_list[index]
            #  im_name = train_item['im_name'].split('.')[0]
            im_name = self.im_list[index]
        else:
            im_name = self.im_list[index]

        im_path = os.path.join(self.root, self.dataset + '_images', im_name + '.jpg')
        parsing_anno_path = os.path.join(self.root, self.dataset + '_segmentations', im_name + '.png')

        im = cv2.imread(im_path, cv2.IMREAD_COLOR)
        h, w, _ = im.shape
        parsing_anno = np.zeros((h, w), dtype=np.long)

        # get pose anno
        if self.dataset == 'train' or self.dataset == 'val':
            joints_all_info = np.array(self.pose_info[im_name])
            joints_loc = np.zeros((joints_all_info.shape[0], 2))
            joints_loc[:, :] = joints_all_info[:, 0:2] # 1st and 2nd column

            # TODO:reorder joints?

            # get visibility of joints
            coord_sum = np.sum(joints_loc, axis=1)
            visibility = coord_sum != 0

        # Get center and scale
        center, s = self._box2cs([0, 0, w - 1, h - 1])
        r = 0

        if self.dataset != 'test':
            parsing_anno = cv2.imread(parsing_anno_path, cv2.IMREAD_GRAYSCALE)

            if self.dataset == 'train' or self.dataset == 'trainval':

                sf = self.scale_factor
                rf = self.rotation_factor
                s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
                r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) \
                    if random.random() <= 0.6 else 0

                if random.random() <= self.flip_prob:
                    im = im[:, ::-1, :]
                    parsing_anno = parsing_anno[:, ::-1]

                    center[0] = im.shape[1] - center[0] - 1
                    right_idx = [15, 17, 19]
                    left_idx = [14, 16, 18]
                    for i in range(0, 3):
                        right_pos = np.where(parsing_anno == right_idx[i])
                        left_pos = np.where(parsing_anno == left_idx[i])
                        parsing_anno[right_pos[0], right_pos[1]] = left_idx[i]
                        parsing_anno[left_pos[0], left_pos[1]] = right_idx[i]

                    # flip the joints
                    joints_loc = flip_joints(joints_loc, w)

                    # swap the visibility of left and right joints
                    #r_joint = [1,2,3,11,12,13]
                    #l_joint = [4,5,6,14,15,16]
                    r_joint = [0,1,2,10,11,12]
                    l_joint = [3,4,5,13,14,15]
                    for i in range(0,6):
                        temp_visibility = visibility[r_joint[i]]
                        visibility[r_joint[i]] = visibility[l_joint[i]]
                        visibility[l_joint[i]] = temp_visibility

        trans = get_affine_transform(center, s, r, self.crop_size)
        #  gt_trans = get_affine_transform(center, s, r, np.asarray([96, 96]))

        input = cv2.warpAffine(
            im,
            trans,
            (int(self.crop_size[1]), int(self.crop_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))
        #  cv2.imwrite('/home/zzw/segment/train_images/' + im_name + '.jpg', input)

        if self.transform:
            input = self.transform(input)

        meta = {
            'name': im_name,
            'center': center,
            'height': h,
            'width': w,
            'scale': s,
            'rotation': r
        }

        if self.dataset == 'test':
            return input, meta
        else:

            label_parsing = cv2.warpAffine(
                parsing_anno,
                trans,
                (int(self.crop_size[1]), int(self.crop_size[0])),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255))

            #  cv2.imwrite('/home/zzw/segment/train_segmentations/' + meta['name'] + '.png', label_parsing)

            '''
            label_edge = generate_edge(label_parsing)
            label_parsing = torch.from_numpy(label_parsing)
            label_edge = torch.from_numpy(label_edge)
            '''
            grid_x = int(self.crop_size[1] / self.pose_net_stride)
            grid_y = int(self.crop_size[0] / self.pose_net_stride)
            #  grid_x = int(self.crop_size[1])
            #  grid_y = int(self.crop_size[0])

            for i in range(joints_all_info.shape[0]):
                if visibility[i] > 0:
                    joints_loc[i, 0:2] = self.affine_trans(joints_loc[i, 0:2], trans)


            #  print("joints loc:", joints_loc)
            label_pose = generate_pose(joints_loc, visibility, trans, grid_x, grid_y, self.pose_net_stride, self.sigma)
            label_edge = generate_edge(label_parsing)
            #  save_edge = cv2.resize(label_edge, (96,96))
            #  cv2.imwrite('/home/zzw/segment/CE2P_0_4_1/lip_edge_gt/' + im_name + '.png', save_edge)
#
            return input, label_parsing, label_pose, label_edge, meta

    def affine_trans(self, pt, t):
        new_pt = np.array([pt[0], pt[1], 1.]).T
        new_pt = np.dot(t, new_pt)
        return new_pt[:2]


if __name__ == '__main__':
    a = datetime.datetime.now()

    data_dir = '/data/zzw/segment/data/lip/images_labels'
    dataset = 'train'
    crop_size = [384,384]
    batch_size = 12

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    loader = data.DataLoader(LIPDataSet(data_dir, dataset, crop_size, transform),
            batch_size, shuffle=True, num_workers=56, pin_memory=True)

    for i, batch in enumerate(loader):
        print(i)

    print(datetime.datetime.now() - a)
