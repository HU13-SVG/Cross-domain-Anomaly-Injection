import random
import time

import numpy as np
from joblib import Parallel, delayed
import os
import glob
import cv2
from PIL import Image
from sympy.codegen.ast import continue_
from tqdm import tqdm
from multiprocessing import Process
from datetime import datetime  # 获取当前时间

from CAI_utils_functions import *


def get_ano_poisson(image_path, src_path, mask_path, L, M, S):
    src_paths = sorted(glob.glob(src_path + "/*.jpg"))  # All abnormal images

    image_ori = cv2.imread(image_path)  # Normal image
    image = cv2.resize(image_ori,(256, 256))

    num = int(image_path.split('/')[-1].split('.')[0])

    dst_mask_ori = Image.open(mask_path + '/' + image_path.split('/')[-1])
    dst_mask = dst_mask_ori.resize((256, 256))

    center_assemble = find_point(dst_mask)

    number_ano = generate_random_odd(len(src_paths) - 1)
    while src_paths[number_ano][-5] == '4':  # Prevent data leakage
        number_ano = generate_random_odd(len(src_paths) - 1)

    anomaly_source_path = src_paths[number_ano]

    anomaly_source_img= cv2.imread(anomaly_source_path)
    anomaly_source_msk = cv2.imread(anomaly_source_path.replace("jpg", "png"))

    # Based on the mask, segment the image containing abnormal textures and remove superfluous regions.
    left_top_pixel, right_bottom_pixel = find_white_pixels(anomaly_source_msk)
    src = crop_image(anomaly_source_img, left_top_pixel, right_bottom_pixel)
    src_mask = crop_image(anomaly_source_msk, left_top_pixel, right_bottom_pixel)

    # Get the mask boundary rectangle of the normal image here
    left_top_pixel_norm, right_bottom_pixel_norm = find_white_pixels(
        np.array(dst_mask))  # min() arg is an empty sequence
    dst_mask_cut = crop_image(np.array(dst_mask), left_top_pixel_norm, right_bottom_pixel_norm)
    src_dst_ratio = max(src.shape) / max(dst_mask_cut.shape)

    L_number = 4
    M_number = 3
    S_number = 3
    if L == L_number and M == M_number and S == S_number:
        print("L,M,S are full, but it's not over?")
    if src_dst_ratio >= 0.7:
        if L < L_number:
            label = 'L'
            src, src_mask,label_error = gen_Large(src, src_mask, src_dst_ratio)
            if label_error == 1:
                return 0, 0, 0, [0, 0], 0, 0, 0, 0, L, M, S
            L += 1
        elif L >= L_number and M < M_number:
            label = 'M'
            src, src_mask,label_error = gen_Medium(src, src_mask, src_dst_ratio)
            if label_error == 1:
                return 0, 0, 0, [0, 0], 0, 0, 0, 0, L, M, S
            M += 1
        elif L >= L_number and M >= M_number and S < S_number:
            label = 'S'
            src, src_mask,label_error = gen_Small(src, src_mask, src_dst_ratio)
            if label_error == 1:
                return 0, 0, 0, [0, 0], 0, 0, 0, 0, L, M, S
            S += 1
        else:
            return 0, 0, 0, [0, 0], 0, 0, 0, 0, L, M, S
    elif 0.3 <= src_dst_ratio < 0.7:  # If the ratio is less than 0.7, regenerate
        if M < M_number:
            label = 'M'
            src, src_mask,label_error = gen_Medium(src, src_mask, src_dst_ratio)
            if label_error == 1:
                return 0, 0, 0, [0, 0], 0, 0, 0, 0, L, M, S
            M += 1
        elif M >= M_number and S < S_number:

            label = 'S'
            src, src_mask,label_error = gen_Small(src, src_mask, src_dst_ratio)
            if label_error == 1:
                return 0, 0, 0, [0, 0], 0, 0, 0, 0, L, M, S
            S += 1
        else:
            return 0, 0, 0, [0, 0], 0, 0, 0, 0, L, M, S
    elif 0.1 <= src_dst_ratio < 0.3:
        if S < S_number:
            label = 'S'
            src, src_mask,label_error = gen_Small(src, src_mask, src_dst_ratio)
            if label_error == 1:
                return 0, 0, 0, [0, 0], 0, 0, 0, 0, L, M, S
            S += 1
        else:
            return 0, 0, 0, [0, 0], 0, 0, 0, 0, L, M, S
    else:  # Too small, regenerate
        return 0, 0, 0, [0, 0], 0, 0, 0, 0, L, M, S

    while min(src.shape[0], src.shape[1]) <= 8:
        src = cv2.copyMakeBorder(src, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        src_mask = cv2.copyMakeBorder(src_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    height_src, width_src = np.array(src).shape[:2]
    height_dst, width_dst = image.shape[:2]
    # Here randomize the center point, this center point must be legal, will not exceed the abnormal image and the source image
    x_min = width_src / 2
    y_min = height_src / 2
    x_max = width_dst - width_src / 2
    y_max = height_dst - height_src / 2
    center_asse_final = []
    for j in range(len(center_assemble)):
        # center_assemble[j][0] is s the y coordinate
        if x_min <= center_assemble[j][1] <= x_max and y_min <= center_assemble[j][0] <= y_max:
            center_asse_final.append(center_assemble[j])  # This step is not time-consuming, but it is a point where you can improve and speed up
    if len(center_asse_final) == 0:
        center1 = [0, 0]
        if label == 'L':
            L -= 1
        elif label == 'M':
            M -= 1
        elif label == 'S':
            S -= 1
    else:
        center1 = center_asse_final[int(random.randint(0, len(center_asse_final) - 1))]
    center = [center1[1], center1[0]]  # My understanding provides that the latter is wide, while opencv considers the former to be wide
    # print(center,anomaly_source_path)
    return src, image, src_mask, center, dst_mask, num, anomaly_source_path, label, L, M, S


def transform_image(image_path):
    random_seed = int(image_path.split('/')[-1].split('.')[0])
    random.seed(random_seed)
    L, M, S = 0, 0, 0
    for i_count in range(0, 10):
        obj_name = image_path.split('/')[-4]
        txt_save_path = 'path to store the generated image in normal mode, end with "/"  '
        save_path = txt_save_path + obj_name
        save_path1 = 'path to store the generated image in paste mode, end with "/"' + obj_name
        save_path2 = 'path to store the generated image in mixed mode, end with "/"' + obj_name
        mask_path = 'mask_path：path of mvtec mask generated by Rembg, remove, end with "/"' + obj_name
        src_path = 'src_path：path of our dataset, do not end with "/"'

        msk_normal = np.array([1, 1])
        msk_mixed = np.array([1, 1])
        while msk_normal.sum() < 10 or msk_mixed.sum() < 10:
            center = [0, 0]
            while center == [0, 0]:
                src, image, src_mask, center, dst_mask, num, anomaly_source_path, label, L, M, S = get_ano_poisson(
                    image_path, src_path,
                    mask_path, L, M, S)
            try:
                dilate_mask_normal = np.array(dilate_image(src_mask))
                augmented_image_flaw_normal = cv2.seamlessClone(src, image, dilate_mask_normal, center,
                                                                cv2.NORMAL_CLONE)
                # cv2.MIXED_CLONE  cv2.MONOCHROME_TRANSFER  cv2.NORMAL_CLONE
                augmented_image_normal = np.array(merge_images(augmented_image_flaw_normal, image, dst_mask))
                msk_normal = difference_image(Image.fromarray(augmented_image_normal), Image.fromarray(image))
            except:
                msk_normal = np.array([1, 1])

            try:
                dilate_mask_mixed = np.array(dilate_image(src_mask))
                augmented_image_flaw_mixed = cv2.seamlessClone(src, image, dilate_mask_mixed, center, cv2.MIXED_CLONE)
                augmented_image_mixed = np.array(merge_images(augmented_image_flaw_mixed, image, dst_mask))
                msk_mixed = difference_image(Image.fromarray(augmented_image_mixed), Image.fromarray(image))
            except:
                msk_mixed = np.array([1, 1])

            if msk_normal.sum() < 10 or msk_mixed.sum() < 10:
                if label == 'L':
                    L -= 1
                elif label == 'M':
                    M -= 1
                elif label == 'S':
                    S -= 1

        msk_normal[msk_normal == 1] = 255
        msk_mixed[msk_mixed == 1] = 255

        folder_image_path = save_path + '/synthesis/image/'
        folder_mask_path = save_path + '/synthesis/mask/'
        os.makedirs(folder_image_path, exist_ok=True)
        os.makedirs(folder_mask_path, exist_ok=True)
        with open(txt_save_path + obj_name + ".txt", "a") as file:
            file.write(str(10 * num + i_count) + ':' + 'Large, medium, small：' + str(label) + anomaly_source_path + '\n')

        # print(10*num+i_count,'finished')
        cv2.imwrite(folder_image_path + str(10 * num + i_count) + '.jpg', augmented_image_normal)
        cv2.imwrite(folder_mask_path + str( 10 * num + i_count) + '.jpg', np.array(msk_normal))
        print('{} finished'.format(10 * num + i_count))

        # mixed
        folder_image_path2 = save_path2 + '/synthesis/image/'
        folder_mask_path2 = save_path2 + '/synthesis/mask/'
        os.makedirs(folder_image_path2, exist_ok=True)
        os.makedirs(folder_mask_path2, exist_ok=True)
        cv2.imwrite(folder_image_path2 + str(10 * num + i_count) + '.jpg', augmented_image_mixed)
        cv2.imwrite(folder_mask_path2 + str(10 * num + i_count) + '.jpg', np.array(msk_mixed))

        # paste
        folder_image_path1 = save_path1 + '/synthesis/image/'
        folder_mask_path1 = save_path1 + '/synthesis/mask/'
        os.makedirs(folder_image_path1, exist_ok=True)
        os.makedirs(folder_mask_path1, exist_ok=True)
        augmented_image_flaw1, msk1 = paste_image_with_mask(src, src_mask, image, center)
        augmented_image_paste = np.array(merge_images(augmented_image_flaw1, image, dst_mask))
        msk_paste = difference_image(Image.fromarray(augmented_image_flaw1), Image.fromarray(image))
        msk_paste[msk_paste == 1] = 255
        cv2.imwrite(folder_image_path1 + str(10 * num + i_count) + '.jpg', augmented_image_paste)
        cv2.imwrite(folder_mask_path1 + str(10 * num + i_count) + '.jpg', np.array(msk_paste))

    return 0


def split_list(lst, n):
    if n <= 0:
        raise ValueError("n must be a positive integer")

    avg = len(lst) // n
    remain = len(lst) % n

    result = []
    start = 0

    for i in range(n):
        end = start + avg + (1 if i < remain else 0)
        result.append(lst[start:end])
        start = end

    return result


if __name__ == '__main__':

    now = datetime.now() # Print the current time, formatted as year month day hour minute second
    print("Current time:", now.strftime('%Y%m%d%H%M%S'))
    obj_list = ['capsule','bottle', 'carpet', 'leather', 'pill',
                'transistor', 'tile', 'cable', 'zipper', 'toothbrush',
                'metal_nut', 'hazelnut', 'screw', 'grid', 'wood']

    for obj in tqdm(obj_list):
        print('Current class:', obj)
        image_path = 'image_path：path to MvTec,end with "/" ' + obj + "/train/good/"
        dst_paths = sorted(glob.glob(image_path + "/*.png"))  # MvTec
        # dst_paths = sorted(glob.glob(image_path + "/*.JPG"))  # visa
        dst_paths_split = split_list(dst_paths, 40)  # 40 is the number of processes
        processes = []
        for param in dst_paths_split:
            # dst_paths = sorted(glob.glob(image_path + "/*.JPG"))  #  visa
            for k in range(len(param)):
                p = Process(target=transform_image, args=(param[k]))
                processes.append(p)
                p.start()  # Start process
        for p in processes:
            p.join()
    now = datetime.now()
    print("Current time:", now.strftime('%Y%m%d %H%M%S'))
