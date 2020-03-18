from ipdb import set_trace
import os
import sys
import numpy as np
import random
import cv2

def flip_joints(joints, im_w, r_joint = [0,1,2,10,11,12], l_joint = [3,4,5,13,14,15]):
    flipped_joints = joints.copy()
    flipped_joints[:, 0] = im_w - 1 - flipped_joints[:, 0]
    flipped_joints = swap_left_and_right(flipped_joints, r_joint, l_joint)
    return flipped_joints

def swap_left_and_right(joints, r_joint = [0,1,2,10,11,12], l_joint = [3,4,5,13,14,15]):

    swapped_joints = joints.copy()
    for i in range(len(r_joint)):
        temp_joint = np.zeros((1, 2))
        temp_joint[0, :] = swapped_joints[r_joint[i], :]
        swapped_joints[r_joint[i], :] = swapped_joints[l_joint[i], :]
        swapped_joints[l_joint[i], :] = temp_joint[0, :]

    return swapped_joints


#  def generate_pose(joints, visibility, trans, grid_x, grid_y, stride, sigma):
    #  # affine joints
    #  joint_num = joints.shape[0] # 17 for lip
    #  #print("joint_num:", joint_num)
    #  one_array = np.ones((1, joint_num))
    #  joints = joints.transpose()
    #  joints_ = np.vstack((joints, one_array))
    #  affine_joints = np.dot(trans, joints_)
    #  joints = affine_joints.transpose()
    #  #print("affine_joints shape:", joints.shape)
    #
    #  # get gaussian map
    #  gaussian_maps = np.zeros((joint_num, grid_y, grid_x))
    #  for ji in range(0, joint_num-1):
    #      if visibility[ji]:
    #          gaussian_map = gen_single_gaussian_map(joints[ji, :], stride, grid_x, grid_y, sigma)
    #          gaussian_maps[ji, :, :] = gaussian_map[:, :]
    #
    #  # Get background heatmap
    #  max_heatmap = gaussian_maps.max(0)
    #
    #  gaussian_maps[joint_num - 1, :, :] = 1 - max_heatmap
    #
    #  return gaussian_maps


def generate_pose(joints, visibility, trans, grid_x, grid_y, stride, sigma):
    #print("before trans", joints)
    joint_num = joints.shape[0] # 16 for lip
    #  one_array = np.ones((1, joint_num))
    #  joints = joints.transpose()
    #  joints_ = np.vstack((joints, one_array))
    #  affine_joints = np.dot(trans, joints_)
    #  joints = affine_joints.transpose()

    tmp_size = sigma * 3
    # get gaussian ma,p
    gaussian_maps = np.zeros((joint_num, grid_y, grid_x))
    target_weight = np.ones((joint_num, 1), dtype=np.float32)
    target_weight[:, 0] = visibility[:]

    for joint_id in range(0, joint_num):
        mu_x = int(joints[joint_id][0] / stride + 0.5)
        mu_y = int(joints[joint_id][1] / stride + 0.5)
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= grid_x or ul[1] >= grid_y or br[0] < 0 or br[1] < 0:
            target_weight[joint_id] = 0
            continue

        # Generate gaussian
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], grid_x) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], grid_y) - ul[1]
        img_x = max(0, ul[0]), min(br[0], grid_x)
        img_y = max(0, ul[1]), min(br[1], grid_y)

        v = target_weight[joint_id]
        if v > 0.5:
            gaussian_maps[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    #  return gaussian_maps, target_weight
    return gaussian_maps


def gen_single_gaussian_map(center, stride, grid_x, grid_y, sigma):
    #print "Target generation -- Single gaussian maps"

    gaussian_map = np.zeros((grid_y, grid_x))
    start = stride / 2.0 - 0.5

    max_dist = np.ceil(np.sqrt(4.6052 * sigma * sigma * 2.0))
    start_x = int(max(0, np.floor((center[0] - max_dist - start) / stride)))
    end_x = int(min(grid_x, np.ceil((center[0] + max_dist - start) / stride)))
    start_y = int(max(0, np.floor((center[1] - max_dist - start) / stride)))
    end_y = int(min(grid_y, np.ceil((center[1] + max_dist - start) / stride)))

    #  for g_y in range(start_y, end_y):
        #  for g_x in range(start_x, end_x):
        #      x = start + g_x * stride
        #      y = start + g_y * stride
        #      d2 = (x - center[0]) * (x - center[0]) + (y - center[1]) * (y - center[1])
        #      exponent = d2 / 2.0 / sigma / sigma
        #      if exponent > 4.6052:
        #          continue
        #      gaussian_map[g_y, g_x] += np.exp(-exponent)
        #      if gaussian_map[g_y, g_x] > 1:
                #  gaussian_map[g_y, g_x] = 1

    return gaussian_map


def generate_edge(label, edge_width=3):
    h, w = label.shape
    edge = np.zeros(label.shape)

    # right
    edge_right = edge[1:h, :]
    edge_right[(label[1:h, :] != label[:h - 1, :]) & (label[1:h, :] != 255)
               & (label[:h - 1, :] != 255)] = 1

    # up
    edge_up = edge[:, :w - 1]
    edge_up[(label[:, :w - 1] != label[:, 1:w])
            & (label[:, :w - 1] != 255)
            & (label[:, 1:w] != 255)] = 1

    # upright
    edge_upright = edge[:h - 1, :w - 1]
    edge_upright[(label[:h - 1, :w - 1] != label[1:h, 1:w])
                 & (label[:h - 1, :w - 1] != 255)
                 & (label[1:h, 1:w] != 255)] = 1

    # bottomright
    edge_bottomright = edge[:h - 1, 1:w]
    edge_bottomright[(label[:h - 1, 1:w] != label[1:h, :w - 1])
                     & (label[:h - 1, 1:w] != 255)
                     & (label[1:h, :w - 1] != 255)] = 1

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (edge_width, edge_width))
    edge = cv2.dilate(edge, kernel)
    return edge

def _box2cs(box, aspect_ratio, pixel_std):
    x, y, w, h = box[:4]
    return _xywh2cs(x, y, w, h, aspect_ratio, pixel_std)


def _xywh2cs(x, y, w, h, aspect_ratio, pixel_std):
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std], dtype=np.float32)
    # if center[0] != -1:
    #    scale = scale * 1.25

    return center, scale





if __name__ == '__main__':
    joints = np.array([[2.0, 1.0],
                       [3.0, 4.0],
                       [1.0, 2.0],
                       [1.0, 1.0]])
    visibility = np.array([[1], [1], [1]])
    #  visibility = np.ones((1, 3))
    trans = np.array([1])
    grid_x = 6
    grid_y = 6
    stride = 1
    sigma = 1
    pose1 = generate_pose(joints, visibility, trans, grid_x, grid_y, stride, sigma)
    print(pose1[0][0])
