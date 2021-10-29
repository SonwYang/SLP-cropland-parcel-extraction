from __future__ import division
import numpy as np
from PIL import Image
import cv2
import scipy.io as sio
import os
import random
import torch
from torch.utils.data import Dataset
from scipy import ndimage
from skimage.morphology import skeletonize
import networkx as nx


def im_normalize(im):
    """
    Normalize image
    """
    imn = (im - im.min()) / (im.max() - im.min())
    return imn


def construct_name(p, prefix):
    """
    Construct the name of the model
    p: dictionary of parameters
    prefix: the prefix
    name: the name of the model - manually add ".pth" to follow the convention
    """
    name = prefix
    for key in p.keys():
        if (type(p[key]) != tuple) and (type(p[key]) != list):
            name = name + '_' + str(key) + '-' + str(p[key])
        else:
            name = name + '_' + str(key) + '-' + str(p[key][0])
    return name


def make_gaussian(size, sigma=10, center=None):
    """ Make a square gaussian kernel.
    size: is the dimensions of the output gaussian
    sigma: is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size[1], 1, float)
    y = np.arange(0, size[0], 1, float)
    y = y[:, np.newaxis]

    if center is None:
        x0 = y0 = size[0] // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)


def make_gt(img, labels, outputRes=None, sigma=10):
    """ Make the ground-truth for each landmark.
    img: the original color image
    labels: the json labels with the Gaussian centers {'x': x, 'y': y}
    sigma: sigma of the Gaussian.
    """

    if outputRes is not None:
        h, w = outputRes
    else:
        h, w = img.shape
    # print (h, w, len(labels))
    #gt = np.zeros((h, w, len(labels)), np.float32)
    gt = np.zeros((h, w, 1), np.float32)

    for land in range(0, labels.shape[0]):
        gt[:,:,0] = gt[:,:,0] + (make_gaussian((h, w), sigma, (labels[land, 0], labels[land, 1])))
    return gt


def txt2mat(idx):
    annotation_file = '/scratch_net/boxy/carlesv/RetinalFeatures/%02d_manual1_gt.txt' % (idx)
    file = open(annotation_file, 'r')
    lines = file.readlines()
    file.close()
    x = np.nan * np.zeros((len(lines), 2))
    for i in range(0,len(lines)):
        x[i, 0] = int(lines[i].split(',')[1])
        x[i, 1] = int(lines[i].split(',')[0])
    return x


def overlay_mask(img, mask, transparency=0.5):
    """
    Overlay a h x w x 3 mask to the image
    img: h x w x 3 image
    mask: h x w x 3 mask
    transparency: between 0 and 1
    """
    im_over = np.ndarray(img.shape)
    im_over[:, :, 0] = (1 - mask[:, :, 0]) * img[:, :, 0] + mask[:, :, 0] * (
    255 * transparency + (1 - transparency) * img[:, :, 0])
    im_over[:, :, 1] = (1 - mask[:, :, 1]) * img[:, :, 1] + mask[:, :, 1] * (
    255 * transparency + (1 - transparency) * img[:, :, 1])
    im_over[:, :, 2] = (1 - mask[:, :, 2]) * img[:, :, 2] + mask[:, :, 2] * (
    255 * transparency + (1 - transparency) * img[:, :, 2])
    return im_over

def generate_graph_patch(pred):

    G=nx.DiGraph()

    for row_idx in range(0,pred.shape[0]):
        for col_idx in range(0,pred.shape[1]):
            node_idx = row_idx*pred.shape[1] + col_idx

            if row_idx > 0 and col_idx > 0:
                node_topleft_idx = (row_idx-1)*pred.shape[1] + col_idx-1
                cost = 1 - pred[row_idx-1,col_idx-1]
                G.add_edge(node_idx,node_topleft_idx,weight=cost)

            if row_idx > 0:
                node_top_idx = (row_idx-1)*pred.shape[1] + col_idx
                cost = 1 - pred[row_idx-1,col_idx]
                G.add_edge(node_idx,node_top_idx,weight=cost)

            if row_idx > 0 and col_idx < pred.shape[1]-1:
                node_topright_idx = (row_idx-1)*pred.shape[1] + col_idx+1
                cost = 1 - pred[row_idx-1,col_idx+1]
                G.add_edge(node_idx,node_topright_idx,weight=cost)

            if col_idx > 0:
                node_left_idx = row_idx*pred.shape[1] + col_idx-1
                cost = 1 - pred[row_idx,col_idx-1]
                G.add_edge(node_idx,node_left_idx,weight=cost)

            if col_idx < pred.shape[1]-1:
                node_right_idx = row_idx*pred.shape[1] + col_idx+1
                cost = 1 - pred[row_idx,col_idx+1]
                G.add_edge(node_idx,node_right_idx,weight=cost)

            if row_idx < pred.shape[0]-1 and col_idx > 0:
                node_bottomleft_idx = (row_idx+1)*pred.shape[1] + col_idx-1
                cost = 1 - pred[row_idx+1,col_idx-1]
                G.add_edge(node_idx,node_bottomleft_idx,weight=cost)

            if row_idx < pred.shape[0]-1:
                node_bottom_idx = (row_idx+1)*pred.shape[1] + col_idx
                cost = 1 - pred[row_idx+1,col_idx]
                G.add_edge(node_idx,node_bottom_idx,weight=cost)

            if row_idx < pred.shape[0]-1 and col_idx < pred.shape[1]-1:
                node_bottomright_idx = (row_idx+1)*pred.shape[1] + col_idx+1
                cost = 1 - pred[row_idx+1,col_idx+1]
                G.add_edge(node_idx,node_bottomright_idx,weight=cost)

    return G


def find_output_connected_points(root_dir, save_vertices_indxs, train, img_idx, patch_size, img_filenames):

    if train:
        img_filename = img_filenames[img_idx]
        img = cv2.imread(os.path.join(root_dir, 'training', '../images', img_filename))
        mask_filename = img_filename[0:len(img_filename)]
        mask_gt = cv2.imread(os.path.join(root_dir, 'training', '1st_manual', mask_filename), 0)
    else:
        img_filename = img_filenames[img_idx]
        img = cv2.imread(os.path.join(root_dir, 'val', '../images', img_filename))
        mask_filename = img_filename[0:len(img_filename)]
        mask_gt = cv2.imread(os.path.join(root_dir, 'val', '1st_manual', mask_filename), 0)

    img = np.array(img, dtype=np.float32)
    h, w = img.shape[:2]

    if len(img.shape) < 3:
        temp = np.zeros((h, w, 3))
        for i in range(3):
            temp[:, :, i] = img

        img = temp

    void_pixels = np.prod((img == np.array([255, 255, 255])), axis=2).astype(np.float32)
    void_pixels_eroded = ndimage.binary_erosion(void_pixels, structure=np.ones((5,5))).astype(void_pixels.dtype)
    void_pixels = ndimage.binary_dilation(void_pixels_eroded, structure=np.ones((5,5))).astype(void_pixels_eroded.dtype)
    valid_pixels = 1-void_pixels

    mask_gt = np.array(mask_gt)
    # mask_gt_skeleton = skeletonize(mask_gt>0)
    mask_gt_skeleton = mask_gt > 0

    mask_gt_skeleton_valid = mask_gt_skeleton*valid_pixels
    valid_indxs = np.argwhere(mask_gt_skeleton_valid==1)

    margin = int(np.round(patch_size/10.0))

    #Select a random point from the ground truth
    #print(img_idx)
    #print(img_filename)
    #print(valid_indxs)
    if len(valid_indxs) > 0:
        num_atempts = 0
        selected_point = random.randint(0,len(valid_indxs)-1)
        center = (valid_indxs[selected_point,1], valid_indxs[selected_point,0])
        while (center[0] < patch_size/2 or center[1] < patch_size/2 or center[0] >  w - patch_size/2 or center[1] >  h - patch_size/2) and num_atempts < 20:
            selected_point = random.randint(0,len(valid_indxs)-1)
            center = (valid_indxs[selected_point,1], valid_indxs[selected_point,0])
            num_atempts += 1

        if num_atempts < 20:

            #Add selected vertex to file to reproduce the experiments
            if save_vertices_indxs:
                f = open(os.path.join(root_dir, 'points_selected.txt'), 'a')
                f.write(str(img_idx) + " " + str(valid_indxs[selected_point,0]) + " " + str(valid_indxs[selected_point,1]) + "\n")
                f.close()

            x_tmp = int(center[0]-patch_size/2)
            y_tmp = int(center[1]-patch_size/2)
            img_crop = img[y_tmp:y_tmp+patch_size,x_tmp:x_tmp+patch_size,:]

            #Find intersection points between skeleton and inner bbox from patch
            mask_gt_crop = mask_gt_skeleton_valid[y_tmp:y_tmp+patch_size,x_tmp:x_tmp+patch_size]

            bbox_mask = np.zeros((patch_size,patch_size))
            bbox_mask[margin,margin:patch_size-margin] = 1
            bbox_mask[margin:patch_size-margin,margin] = 1
            bbox_mask[margin:patch_size-margin,patch_size-margin] = 1
            bbox_mask[patch_size-margin,margin:patch_size-margin] = 1

            intersection_bbox_with_gt = mask_gt_crop*bbox_mask

            #Discard intersection points not connected to the patch center
            G = generate_graph_patch(mask_gt_crop)
            idx_end = (patch_size/2)*patch_size + patch_size/2
            intersection_idxs = np.argwhere(intersection_bbox_with_gt==1)
            connected_intersection_idxs = []
            for ii in range(0,len(intersection_idxs)):
                idx_start = intersection_idxs[ii,0]*patch_size + intersection_idxs[ii,1]
                length_pred, path_pred = nx.bidirectional_dijkstra(G, idx_start, idx_end, weight='weight')
                if length_pred == 0:
                    connected_intersection_idxs.append(intersection_idxs[ii])

            output_points = np.asarray(connected_intersection_idxs)
            for ii in range(0,len(output_points)):
                tmp_value = output_points[ii,0]
                output_points[ii,0] = output_points[ii,1]
                output_points[ii,1] = tmp_value

            return img_crop, output_points, mask_gt_crop

        else:

            output_points = []
            img_crop = []
            mask_gt_crop = []

            return img_crop, output_points, mask_gt_crop

    else:
        output_points = []
        img_crop = []
        mask_gt_crop = []
        return img_crop, output_points, mask_gt_crop


def find_output_connected_points_selected_point(root_dir, selected_vertex, train, img_idx, patch_size, center, img_filenames):

    if train:
        img_filename = img_filenames[img_idx]
        img = Image.open(os.path.join(root_dir, 'training', '../images', img_filename))
        mask_filename = img_filename[0:len(img_filename)-1]
        mask_gt = Image.open(os.path.join(root_dir, 'training', '1st_manual', mask_filename))
    else:
        img_filename = img_filenames[img_idx]
        img = Image.open(os.path.join(root_dir, 'val', '../images', img_filename))
        mask_filename = img_filename[0:len(img_filename)-1]
        mask_gt = Image.open(os.path.join(root_dir, 'val', '1st_manual', mask_filename))

    img = np.array(img, dtype=np.float32)
    h, w = img.shape[:2]

    void_pixels = np.prod((img == np.array([255, 255, 255])), axis=2).astype(np.float32)
    void_pixels_eroded = ndimage.binary_erosion(void_pixels, structure=np.ones((5,5))).astype(void_pixels.dtype)
    void_pixels = ndimage.binary_dilation(void_pixels_eroded, structure=np.ones((5,5))).astype(void_pixels_eroded.dtype)
    valid_pixels = 1-void_pixels

    mask_gt = np.array(mask_gt)
    mask_gt_skeleton = skeletonize(mask_gt>0)

    mask_gt_skeleton_valid = mask_gt_skeleton*valid_pixels
    valid_indxs = np.argwhere(mask_gt_skeleton_valid==1)

    margin = int(np.round(patch_size/10.0))

    x_tmp = int(center[0]-patch_size/2)
    y_tmp = int(center[1]-patch_size/2)
    img_crop = img[y_tmp:y_tmp+patch_size,x_tmp:x_tmp+patch_size,:]

    #Find intersection points between skeleton and inner bbox from patch
    mask_gt_crop = mask_gt_skeleton_valid[y_tmp:y_tmp+patch_size,x_tmp:x_tmp+patch_size]

    bbox_mask = np.zeros((patch_size,patch_size))
    bbox_mask[margin,margin:patch_size-margin] = 1
    bbox_mask[margin:patch_size-margin,margin] = 1
    bbox_mask[margin:patch_size-margin,patch_size-margin] = 1
    bbox_mask[patch_size-margin,margin:patch_size-margin] = 1

    intersection_bbox_with_gt = mask_gt_crop*bbox_mask

    #Discard intersection points not connected to the patch center
    G = generate_graph_patch(mask_gt_crop)
    idx_end = (patch_size/2)*patch_size + patch_size/2
    intersection_idxs = np.argwhere(intersection_bbox_with_gt==1)
    connected_intersection_idxs = []
    for ii in range(0,len(intersection_idxs)):
        idx_start = intersection_idxs[ii,0]*patch_size + intersection_idxs[ii,1]
        length_pred, path_pred = nx.bidirectional_dijkstra(G, idx_start, idx_end, weight='weight')
        if length_pred == 0:
            connected_intersection_idxs.append(intersection_idxs[ii])

    output_points = np.asarray(connected_intersection_idxs)
    for ii in range(0,len(output_points)):
        tmp_value = output_points[ii,0]
        output_points[ii,0] = output_points[ii,1]
        output_points[ii,1] = tmp_value

    return img_crop, output_points


class ToolDataset(Dataset):
    """Tool Dataset constructed using the PyTorch built-in functionalities"""

    def __init__(self, train=True,
                 inputRes=(64, 64),
                 outputRes=(64, 64),
                 sigma=5,
                 db_root_dir='D:/MyWorkSpace/dl_dataset/road_extraction/masa/',
                 transform=None,
                 save_vertices_indxs=False):
        """Loads image to label pairs for tool pose estimation
        db_elements: the names of the video files
        db_root_dir: dataset directory with subfolders "frames" and "Annotations"
        """
        self.train = train
        self.db_root_dir = db_root_dir
        self.inputRes = inputRes
        self.outputRes = outputRes
        self.sigma = sigma
        self.transform = transform
        self.save_vertices_indxs = save_vertices_indxs

        if self.train:
            #self.img_list = range(1,1109)

            path = db_root_dir + 'training/images/'
            self.train_img_filenames = os.listdir(path)
            self.img_list = range(1, len(self.train_img_filenames))

        else:

            path = db_root_dir + 'val/images/'
            self.test_img_filenames = os.listdir(path)
            self.img_list = range(0, len(self.test_img_filenames))

        print('Done initializing Dataset')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img, gt, valid_img, mask_gt_crop = self.make_img_gt_pair(idx)

        sample = {'image': img, 'gt': gt, 'valid_img': valid_img, 'mask_gt_crop': mask_gt_crop}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def make_img_gt_pair(self, idx):
        """
        Make the image-ground-truth pair
        """
        if self.train:

            img_idx = self.img_list[idx]
            patch_size = self.inputRes[0]

            img_crop, output_points, mask_gt_crop = find_output_connected_points(self.db_root_dir, False, self.train, img_idx, patch_size, self.train_img_filenames)

            if len(output_points) > 0:

                gt = make_gt(img_crop, output_points, (patch_size,patch_size), self.sigma)
                return img_crop, gt, 1, mask_gt_crop

            else:

                img_crop = np.zeros((patch_size, patch_size, 3))
                gt = np.zeros((patch_size, patch_size))
                return img_crop, gt, 0, mask_gt_crop

        else:

            img_idx = self.img_list[idx]
            patch_size = self.inputRes[0]

            img_crop, output_points, mask_gt_crop = find_output_connected_points(self.db_root_dir, self.save_vertices_indxs, self.train, img_idx, patch_size, self.test_img_filenames)

            if len(output_points) > 0:

                gt = make_gt(img_crop, output_points, (patch_size,patch_size), self.sigma)
                return img_crop, gt, 1, mask_gt_crop

            else:

                img_crop = np.zeros((patch_size,patch_size,3))
                mask_gt_crop = np.zeros((patch_size,patch_size,3))
                gt = np.zeros((patch_size,patch_size))
                return img_crop, gt, 0, mask_gt_crop



    def store_gt_asmatfile(self):
        gt_tool = np.zeros((2, 3, len(self.img_list)), dtype=np.float32)
        for i in range(0, len(self.img_list)):
            temp = txt2mat(self.img_list[i])
            if temp.shape[0] == 0:
                gt_tool[:, :, i] = np.nan
            else:
                gt_tool[:, :, i] = np.transpose(temp)

        a = {'gt_tool': gt_tool}
        sio.savemat('gt_tool', a)

    def get_img_size(self):
        if self.train:
            img = Image.open(os.path.join(self.db_root_dir, 'training', '../images', '%02d_training.tif' % (self.img_list[0])))
        else:
            img = Image.open(os.path.join(self.db_root_dir, 'val', '../images', '%02d_test.tif' % (self.img_list[0])))

        return list(reversed(img.size))


class ScaleNRotate(object):
    """Scale (zoom-in, zoom-out) and Rotate the image and the ground truth.
    Args:
        maxRot (float): maximum rotation angle to be added
        maxScale (float): maximum scale to be added
    """
    def __init__(self, rots=(-30, 30), scales=(.75, 1.25)):
        self.rots = rots
        self.scales = scales

    def __call__(self, sample):

        rot = (self.rots[1] - self.rots[0]) * random.random() - \
              (self.rots[1] - self.rots[0])/2

        sc = (self.scales[1] - self.scales[0]) * random.random() - \
             (self.scales[1] - self.scales[0]) / 2 + 1

        img, gt = sample['image'], sample['gt']
        h, w = img.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, rot, sc)
        img_ = cv2.warpAffine(img, M, (w, h))

        h_gt, w_gt = gt.shape[:2]
        center_gt = (w_gt / 2, h_gt / 2)
        M = cv2.getRotationMatrix2D(center_gt, rot, sc)
        gt_ = cv2.warpAffine(gt, M, (w_gt, h_gt))

        return {'image': img_, 'gt': gt_}

class RandomHorizontalFlip(object):
    """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

    def __call__(self, sample):

        image, gt, valid_img = sample['image'], sample['gt'], sample['valid_img']

        if random.random() < 0.5:
            image = cv2.flip(image, flipCode=1)
            gt = cv2.flip(gt, flipCode=1)

        sample['image'], sample['gt'], sample['valid_img'] = image, gt, valid_img

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, gt, valid_img, mask_gt_crop = sample['image'], sample['gt'], sample['valid_img'], sample['mask_gt_crop']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        if len(gt.shape) == 2:
            gt_tmp = gt
            h, w = gt_tmp.shape
            gt = np.zeros((h, w, 1), np.float32)
            gt[:,:,0] = gt_tmp
        if len(mask_gt_crop.shape) == 2:
            mask_gt_crop_tmp = mask_gt_crop
            h, w = mask_gt_crop_tmp.shape
            mask_gt_crop = np.zeros((h, w, 1), np.float32)
            mask_gt_crop[:,:,0] = mask_gt_crop_tmp
        if len(image.shape) == 2:
            image_tmp = image
            h, w = image_tmp.shape
            image = np.zeros((h, w, 3), np.float32)
            image[:,:,0] = image_tmp
            image[:,:,1] = image_tmp
            image[:,:,2] = image_tmp
        image = image.transpose((2, 0, 1))
        gt = gt.transpose((2, 0, 1))
        mask_gt_crop = mask_gt_crop.transpose((2, 0, 1))

        return {'image': torch.from_numpy(image),
                'gt': torch.from_numpy(gt),
                'valid_img': valid_img,
                'mask_gt_crop': torch.from_numpy(mask_gt_crop)}


class normalize(object):

    def __init__(self, mean=[171.0773/255, 98.4333/255, 58.8811/255], std=[1.0, 1.0, 1.0]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):

        image, gt, valid_img = sample['image'], sample['gt'], sample['valid_img']
        for t, m, s in zip(image, self.mean, self.std):
            t.sub_(m).div_(s)

        sample['image'], sample['gt'], sample['valid_img'] = image, gt, valid_img

        return sample


if __name__ == 'main':
    a = ToolDataset(train=True)

    for i, sample in enumerate(a):
        print(i)
        gt = sample['gt']
        if gt.shape[2] == 0:
            break

    b = a[153]
    #plt.imshow(b['image'])
    #plt.imshow(b['gt'])

