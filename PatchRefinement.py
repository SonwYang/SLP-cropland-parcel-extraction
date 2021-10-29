# Includes
import os
import numpy as np
import utils.toolbox as tb
import torch
from torch.autograd import Variable
import Nets as nt
import scipy.misc
from PIL import Image
from photutils import find_peaks
from astropy.stats import sigma_clipped_stats
import json
import cv2
from skimage.morphology import skeletonize
import glob
import matplotlib.pyplot as plt
import math
from skimage import morphology
import tqdm
import gdalTools
import shutil


def compute_Euclidean_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    distance =np.sqrt(np.sum(np.square(x1-x2)+np.square(y1-y2)))
    return distance


def GetPredInDirectionBreakPoint(centerpoint,
                                 breakPointList,
                                 predPoint,
                                 angThreshold=40):
    InBreakPoint = False
    for breakpoint in breakPointList:
        v1 = [centerpoint[0], centerpoint[1], breakpoint[0], breakpoint[1]]
        v2 = [centerpoint[0], centerpoint[1], predPoint[0], predPoint[1]]
        ang = compute_angle(v1, v2)
        if ang < angThreshold:
            InBreakPoint = True
            predPoint = breakpoint
            break

    return predPoint, InBreakPoint


def checkCross(p1, p2, skeleton):
    minX = min(p1[0], p2[0]) - 1
    maxX = max(p1[0], p2[0]) + 2
    minY = min(p1[1], p2[1]) - 1
    maxY = max(p1[1], p2[1]) + 2
    subSkeleton = skeleton[minX: maxX, minY: maxY]
    w, h = subSkeleton.shape[:2]
    tempImg = np.zeros((w, h, 3))
    sP1 = [p1[0] - minX, p1[1] - minY]
    sP2 = [p2[0] - minX, p2[1] - minY]
    cv2.line(tempImg, (sP1[1], sP1[0]), (sP2[1], sP2[0]), (255, 255, 255), 1, 8, 0)
    cv2.line(tempImg, (sP1[1]+1, sP1[0]), (sP2[1], sP2[0]), (255, 255, 255), 1, 8, 0)
    cv2.line(tempImg, (sP1[1]-1, sP1[0]), (sP2[1], sP2[0]), (255, 255, 255), 1, 8, 0)
    # cv2.circle(tempImg, (sP1[1], sP1[0]), 1, (0, 0, 255), -1)
    # cv2.circle(tempImg, (sP2[1], sP2[0]), 1, (0, 0, 255), -1)

    # print(f'p1:{p1}, p2:{p2}')
    # print(f'the shape of tempImg:{tempImg.shape}')
    # print(f'sP1:{sP1}, sP2:{sP2}')

    tempImg2 = tempImg[:, :, 0]
    tempImg2 = np.where(tempImg2 > 0, 1, 0)
    tempImg2[sP1[0], sP1[1]] = 0
    tempImg2[sP2[0], sP2[1]] = 0
    newSkeleton = subSkeleton + tempImg2
    unis = np.unique(newSkeleton)

    # cmap = 'nipy_spectral'
    # plt.subplot(121)
    # plt.imshow(subSkeleton)
    # plt.title('subSkeleton')
    # plt.subplot(122)
    # plt.imshow(newSkeleton, cmap=plt.get_cmap(cmap))
    # plt.colorbar()
    # plt.title('newSkeleton')
    # plt.show()

    if len(unis) > 2:
        return True
    else:
        return False



def obtainPointsInLines(point, skeleton):
    x, y = point

    subImg = skeleton[x - 2: x + 3, y - 2: y + 3]
    xs, ys = np.where(subImg)
    if len(xs) > 0:
        return (x - 2 + xs[0], y - 2 + ys[0]), True
    else:
        return point, False

def compute_angle(v1, v2):
    dx1 = v1[2] - v1[0]
    dy1 = v1[3] - v1[1]
    dx2 = v2[2] - v2[0]
    dy2 = v2[3] - v2[1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180/math.pi)
    # print(angle1)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180/math.pi)
    # print(angle2)
    if angle1*angle2 >= 0:
        included_angle = abs(angle1-angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle
    return included_angle


def detect_breakpoints(img):
    r, c = img.shape
    new_image = np.zeros((r, c))
    detection_operator = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    break_points = []
    for i in range(r-2):
        for j in range(c-2):
            if img[i+1, j+1] == 0:
                new_image[i + 1, j + 1] = 0
            else:
                if np.sum(img[i:i + 3, j:j + 3] * detection_operator) == 2:
                    new_image[i+1, j+1] = 1
                    break_points.append([i+1, j+1])
                else:
                    new_image[i+1, j+1] = 0
    return np.uint8(new_image), break_points


def detect_breakpoints_and_direction(img):
    r, c = img.shape
    new_image = np.zeros((r, c))
    detection_operator = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    break_points = []
    for i in range(r-2):
        for j in range(c-2):
            if img[i+1, j+1] == 0:
                new_image[i + 1, j + 1] = 0
            else:
                if np.sum(img[i:i + 3, j:j + 3] * detection_operator) == 2:
                   xs, ys = np.where(img[i:i + 3, j:j + 3])
                   break_points.append([i+xs[0], j+ys[0], i+xs[1], j+ys[1]])
                else:
                   new_image[i+1, j+1] = 0
    return np.uint8(new_image), break_points


def detect_breakpoints_and_direction2(img):
    r, c = img.shape
    detection_operator = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    points = {}
    points['breakpoints'] = []
    points['directions'] = []
    for i in range(r-2):
        for j in range(c-2):
            if img[i+1, j+1] == 0:
                pass
            else:
                if np.sum(img[i:i + 3, j:j + 3] * detection_operator) == 2:
                   xs, ys = np.where(img[i:i + 3, j:j + 3])
                   temp = [[xs[0], ys[0]], [xs[1], ys[1]]]
                   index = temp.index([1, 1])
                   temp.pop(index)
                   # temp = np.array(temp, dtype=np.int8)
                   points['breakpoints'].append([i + 1, j + 1])
                   points['directions'].append([i + temp[0][0], j + temp[0][1]])
                else:
                   pass
    return points


def detectCenterDirection(img, centerPoint):
    detection_operator = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    x, y = centerPoint
    dx, dy = x, y
    whilestopSignal = False
    count = 0

    linePoints = []
    linePoints.append((x, y))
    while True:
       forStopSiganl = False

       for i in [-1, 0, 1]:
           for j in [-1, 0, 1]:
               start_center_x, start_center_y = x + i, y + j
               # print(f'***********{linePoints}')
               if (start_center_x, start_center_y) in linePoints:
                   # print(f'***********{linePoints}')
                   continue

               subImg = img[start_center_x - 1: start_center_x + 2, start_center_y - 1: start_center_y + 2]
               if np.sum(subImg * detection_operator) > 3:
                   dx, dy = start_center_x, start_center_y
                   linePoints.append((start_center_x, start_center_y))
                   whilestopSignal = True
                   forStopSiganl = True
                   break

               if np.sum(subImg * detection_operator) == 3 and subImg[1, 1] == 1:

                   x, y = start_center_x, start_center_y
                   dx, dy = x, y
                   linePoints.append((start_center_x, start_center_y))
                   forStopSiganl = True
                   break

           if forStopSiganl == True:
               break

       if whilestopSignal == True:
            break

       count += 1
       if count > 8:
            whilestopSignal = True

    return [centerPoint[0], centerPoint[1], dx, dy]



class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def direction_estimation(img, dx, dy):

    gray = cv2.bilateralFilter(img, 3, 3 * 2, 3 / 2)
    edge = cv2.Canny(gray,100,200,3)

    fft2 = np.fft.fft2(edge)
    shift2center = np.fft.fftshift(fft2)
    log_shift2center = np.log(1 + np.abs(shift2center))

    sumfft = np.zeros(180)
    row, col = gray.shape
    R = row if row < col else col
    R //= 2
    y0 = row // 2
    x0 = col // 2
    for theta in range(180):
        sumfft[theta] = 0
        for r in range(R):
            x = int(x0 + r * np.cos(theta * np.pi / 180.0))
            y = int(y0 + r * np.sin(theta * np.pi / 180.0))
            sumfft[theta] += log_shift2center[y, x]

    angle = np.argmax(sumfft)

    p0, p1 = np.zeros(2), np.zeros(2)
    p2 = np.zeros(2)
    p0[0] = int(y0 + 10 * np.sin(angle * np.pi / 180))
    p0[1] = int(x0 + 10 * np.cos(angle * np.pi / 180))
    p1[0] = int(y0 - 10 * np.sin(angle * np.pi / 180))
    p1[1] = int(x0 - 10 * np.cos(angle * np.pi / 180))
    p2[0] = dy
    p2[1] = dx

    p0, p1 = p0.astype(np.int8), p1.astype(np.int8)
    p2 = p2.astype(np.int8)

    v0 = [x0, y0, p0[1], p0[0]]
    v1 = [x0, y0, p1[1], p1[0]]
    v2 = [x0, y0, p2[1], p2[0]]
    ang0 = compute_angle(v2, v0)
    ang1 = compute_angle(v2, v1)
    min_ang = min(ang0, ang1)
    # print(f"min_ang:{min_ang}")
    if ang0 == min_ang:
        vector = v0
    else:
        vector = v1

    return vector

## python PatchRefinement.py --lineDN D:\MyWorkSpace\myProject\SLP-CroplandExtraction\results\line_dn.tif --img D:\MyWorkSpace\myProject\SLP-CroplandExtraction\images\test.tif --weights D:\MyWorkSpace\myProject\SLP-CroplandExtraction\ckpts\patchRefinement.pth

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lineDN', type=str, default=r'D:\MyWorkSpace\paper\plough\data\SOED2_cq\line_dn.tif', help='the path of image')
    parser.add_argument('--img',  type=str, default=r'D:\2021\3\EESNet_test\test\test3.tif', help='the out path of shapefile')
    parser.add_argument('--weights',  type=str, default=r'D:\MyWorkSpace\myProject\SLP-CroplandExtraction\ckpts\patchRefinement.pth', help='the out path of shapefile')
    args = parser.parse_args()

    # Setting of parameters
    epoch = 50
    output_dir = os.path.split(args.lineDN)[0]
    mkdir(output_dir)
    lineDNPath = args.lineDN
    imgPath = args.img
    # Parameters in p are used for the name of the model

    p = {}
    p['useRandom'] = 1  # Shuffle Images
    p['useAug'] = 0  # Use Random rotations in [-30, 30] and scaling in [.75, 1.25]
    p['inputRes'] = (48, 48)  # Input Resolution
    p['outputRes'] = (48, 48)  # Output Resolution (same as input)
    p['g_size'] = 64  # Higher means narrower Gaussian

    p['trainBatch'] = 1  # Number of Images in each mini-batch
    p['numHG'] = 2  # Number of Stacked Hourglasses
    p['Block'] = 'ConvBlock'  # Select: 'ConvBlock', 'BasicBlock', 'BottleNeck'
    p['GTmasks'] = 0 # Use GT Vessel Segmentations as input instead of Retinal Images

    # Setting other parameters
    numHGScales = 4  # How many times to downsample inside each HourGlass
    gpu_id = 0  # Select which GPU, -1 if CPU
    modelName = tb.construct_name(p, "HourGlass")
    patch_radius = int(p['outputRes'][0] / 2)

    # Define the Network and load the pre-trained weights as a CPU tensor
    net = nt.Net_SHG(p['numHG'], numHGScales, p['Block'], 128, 1)
    net.load_state_dict(torch.load(args.weights,
                                   map_location=lambda storage, loc: storage))
    # No need to back-propagate
    for par in net.parameters():
        par.requires_grad = False

    # Transfer to GPU if needed
    if gpu_id >= 0:
        torch.cuda.set_device(device=gpu_id)
        net.cuda()

    num_patches_per_image = 50
    num_images = 2

    max_values = []
    min_values = []


    im_proj, im_geotrans, im_width, im_height, image = gdalTools.read_img(lineDNPath)
    _, _, _, _, image2 = gdalTools.read_img(imgPath)
    image2 = image2.transpose((1, 2, 0))

    width, height = image.shape[:2]

    ret2, th2 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th2 = np.where(th2 > 0, 1, 0)
    skeleton = skeletonize(th2)
    skeleton = np.where(skeleton > 0, 1, 0).astype(np.uint8)
    w, h = skeleton.shape[:2]
    image_line = np.zeros((w, h, 3))

    # for i in range(3):
    #     image_line[:, :, i] = skeleton * 255

    binary = skeleton.astype(bool)
    binary = morphology.remove_small_objects(binary, min_size=8, connectivity=8)
    binary = binary.astype(np.uint8)
    skeleton = binary

    _, breakpoints = detect_breakpoints(skeleton)
    connectBreakPoints = []

    for kk, breakpoint in tqdm.tqdm(enumerate(breakpoints)):
        # cv2.circle(image_line, (breakpoint[1], breakpoint[0]), 1,
        #            (255, 0, 255), -1)

        x, y = breakpoint
        start_x, start_y = x - patch_radius, y - patch_radius
        if start_x < 0:
            continue
        if start_y < 0:
            continue
        if start_x + patch_radius * 2 > width:
            continue
        if start_y + patch_radius * 2 > height:
            continue

        center_x, center_y = patch_radius, patch_radius
        sub_skeleton = skeleton[start_x:start_x + patch_radius * 2, start_y:start_y + patch_radius * 2]
        sub_line_dn = image[start_x:start_x + patch_radius * 2, start_y:start_y + patch_radius * 2]
        sub_break_points = detect_breakpoints_and_direction2(sub_skeleton)

        img = image2[start_x:start_x + patch_radius * 2, start_y:start_y + patch_radius * 2]
        img_save = img.copy()
        # plt.subplot(121)
        # plt.imshow(img)
        # plt.subplot(122)
        # plt.imshow(sub_skeleton)
        # plt.show()

        img = np.array(img, dtype=np.float32)

        # if len(img.shape) == 2:
        #     image_tmp = img
        #     h, w = image_tmp.shape
        #     img = np.zeros((h, w, 3))
        #     img[:,:,0] = image_tmp
        #     img[:,:,1] = image_tmp
        #     img[:,:,2] = image_tmp
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).float()
        img = img.unsqueeze(0)

        inputs = img / 255 - 0.5

        # Forward pass of the mini-batch
        inputs = Variable(inputs)
        if gpu_id >= 0:
            inputs = inputs.cuda()

        output = net.forward(inputs)
        pred = np.squeeze(np.transpose(output[len(output)-1].cpu().data.numpy()[0, :, :, :], (1, 2, 0)))

        # cv2.imwrite(output_dir + 'img_%02d_patch_%02d.png' %(ii+1, jj+1), pred)
        # np.save(output_dir + 'img_%02d_patch_%02d.npy' %(ii+1, jj+1), pred)

        mean, median, std = sigma_clipped_stats(pred, sigma=3.0)
        threshold = median + (10.0 * std)
        sources = find_peaks(pred, threshold, box_size=3)

        data = {}
        data['peaks'] = []
        try:
            indxs = np.argsort(sources['peak_value'])
        except:
            continue

        img2 = np.zeros((sub_skeleton.shape[0], sub_skeleton.shape[1], 3))
        img2[:, :, 0] = sub_skeleton * 255
        img2[:, :, 1] = sub_skeleton * 255
        img2[:, :, 2] = sub_skeleton * 255

        center_index = sub_break_points['breakpoints'].index([center_x, center_y])
        center_dx, center_dy = sub_break_points['directions'][center_index]
        center_vector = detectCenterDirection(sub_skeleton, (center_x, center_y))
        sub_break_points['breakpoints'].pop(center_index)
        sub_break_points['directions'].pop(center_index)
        cv2.line(img2, (center_vector[1], center_vector[0]), (center_vector[3], center_vector[2]), (0, 255, 0), 1, 8, 0)

        # for i in range(len(sub_break_points['breakpoints'])):
        #     try:
        #         cv2.circle(img2, (sub_break_points['breakpoints'][i][1], sub_break_points['breakpoints'][i][0]), 1,
        #                    (255, 0, 0), -1)
        #         cv2.circle(img2, (sub_break_points['directions'][i][1], sub_break_points['directions'][i][0]), 1,
        #                    (255, 255, 0), -1)
        #     except:
        #         continue

        connect_candidates = []
        max_ang = 0
        pred_points_save = []
        for ii in range(0, len(indxs)):
            if ii == 4:
                break

            idx = indxs[len(indxs) - 1 - ii]
            if sources['peak_value'][idx] < 15:
                continue

            pred_y, pred_x = sources['x_peak'][idx], sources['y_peak'][idx]
            pred_points_save.append([pred_x, pred_y])
            # pred_vec = [center_x, center_y, pred_x, pred_y]
            # ## according to the angle, delete some useless points
            # ang = compute_angle(center_vector, pred_vec)
            # if ang < 30:
            #     continue
            # if ang > max_ang:
            #     max_ang = ang

            cv2.circle(img2, (sources['x_peak'][idx], sources['y_peak'][idx]), 1, (0, 0, 255), -1)
            cv2.circle(img_save, (sources['x_peak'][idx], sources['y_peak'][idx]), 2, (0, 0, 255), -1)

            ## Gets a prediction of the direction of the breakpoint
            (pred_x, pred_y), InBreakPoint = GetPredInDirectionBreakPoint((center_x, center_y),
                                                                          sub_break_points['breakpoints'],
                                                                          (pred_x, pred_y))

            cross = False
            if InBreakPoint:
                ## check cross, if two line is crossing, then abandan it
                cross = checkCross((center_x, center_y), (pred_x, pred_y), sub_skeleton)
                if not cross:
                    connect_candidates.append([pred_x, pred_y])
                    break
                else:
                    continue
            else:
                continue
                # connect_candidates.append([pred_x, pred_y])
                # cross = checkCross((center_x, center_y), (pred_x, pred_y), sub_skeleton)
                # if not cross:
                #     connect_candidates.append([pred_x, pred_y])

        # if os.path.exists('temp'):
        #     shutil.rmtree('temp')


        gdalTools.mkdir('temp3')
        outImgName = os.path.join('temp3', str(kk).zfill(6) + '.jpg')
        cv2.imwrite(outImgName, img_save)

        # if pointLine is not None:
            #     print(pointLine)
            #     pred_x, pred_y = pointLine
            #     cv2.circle(img2, (pred_y, pred_x), 1, (255, 0, 0), -1)
            #     connect_candidates.append([pred_x, pred_y])

                # if ang >= max_ang:
                #     connect_candidates.append([pred_x, pred_y])

        # print(f"connect_candidates:{connect_candidates}")
        for p in connect_candidates:
            cv2.line(img2, (center_y, center_x), (p[1], p[0]), (0, 255, 255), 1, 8, 0)
            # print(f"connect points:{p}")
            cv2.line(image_line, (start_y+center_y, start_x+center_x), (start_y+p[1], start_x+p[0]), (255, 255, 255), 1, 8, 0)
            connectBreakPoints.append([start_x+p[0], start_y+p[1]])
            connectBreakPoints.append([start_x+center_x, start_y+center_y])

        for p in sub_break_points['breakpoints']:
            connectBreakPoints.append([start_x+p[0], start_y+p[1]])


        # plt.subplot(221)
        # plt.imshow(img2)
        # plt.subplot(222)
        # plt.imshow(sub_line_dn)
        # plt.subplot(223)
        # plt.imshow(sub_skeleton)
        # plt.subplot(224)
        # plt.imshow(pred)
        # plt.show()

    image_line = image_line[:, :, 0]
    image_line = np.where(image_line > 0, 1, 0)

    skeleton = skeleton.astype(np.uint8) + image_line.astype(np.uint8)
    skeleton = np.where(skeleton > 0, 255, 0).astype(np.uint8)
    subRoot = os.path.split(lineDNPath)[0]
    gdalTools.write_img(os.path.join(subRoot, 'skeleton2.tif'), im_proj, im_geotrans, skeleton)






