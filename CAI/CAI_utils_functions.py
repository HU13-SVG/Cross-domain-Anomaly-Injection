import numpy as np
import os
import random
from PIL import Image, ImageDraw,ImageChops
from scipy.sparse import lil_matrix, linalg, csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse import lil_matrix
import cv2


seed_value = 42
np.random.seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)

def find_white_pixels(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    white_pixels = np.where(gray_image > 128)
    left_top_pixel = (np.min(white_pixels[1]), np.min(white_pixels[0]))
    right_bottom_pixel = (np.max(white_pixels[1]), np.max(white_pixels[0]))
    return left_top_pixel, right_bottom_pixel


def crop_image(image, start_pixel, end_pixel):
    cropped_img = Image.fromarray(image).crop((start_pixel[0], start_pixel[1], end_pixel[0], end_pixel[1]))
    return np.array(cropped_img)

def dilate_image(img):
    kernel = np.ones((8, 8), np.uint8)
    dst_mask = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    dst_mask = Image.fromarray(cv2.dilate(dst_mask, kernel, iterations=1))
    return dst_mask

def merge_images(image_a, image_b, mask):
    image_a = Image.fromarray(image_a)
    image_b = Image.fromarray(image_b)
    mask_width, mask_height = mask.size
    new_image = Image.new('RGB', image_a.size, (0, 0, 0))

    draw = ImageDraw.Draw(new_image)
    for y in range(mask_height):
        for x in range(mask_width):
            mask_pixel = mask.getpixel((x, y))
            if mask_pixel == (255, 255, 255):
                draw.point((x, y), fill=image_a.getpixel((x, y)))
            else:
                draw.point((x, y), fill=image_b.getpixel((x, y)))
    return new_image

def find_point(dst_mask):
    dst_mask_array = np.array(dst_mask)
    center_assemble = []
    for x, y in np.ndindex(dst_mask.width, dst_mask.height):
        r, g, b = dst_mask_array[y, x]
        if int(r) + int(g) + int(b) > 0.1:
            center_assemble.append([y, x])
    return center_assemble

def difference_image(img1, img2):
    if img1.mode != 'L' or img2.mode != 'L':
        img1 = img1.convert('L')
        img2 = img2.convert('L')
    width, height = img1.size
    diff_img = Image.new('L', (width, height), 255)
    diff = ImageChops.difference(img1, img2).convert('L')
    diff_array = np.array(diff)
    threshold = 7
    diff_array[diff_array > threshold] = 255
    diff_array[diff_array <= threshold] = 0
    diff_img = Image.fromarray(diff_array)
    dst_mask = cv2.medianBlur(np.array(diff_img), 7)
    return dst_mask


def generate_random_odd(b):
    num = random.randint(0, b)
    return num if num % 2 == 0 else num - 1

def gen_Large(src,src_mask,src_dst_radio):
    random_number = np.random.uniform(0.7, 1)
    rounded_number = round(random_number, 1)
    new_size = (int(src.shape[1] / src_dst_radio * rounded_number), int(src.shape[0] / src_dst_radio * rounded_number))
    label = 0
    if 0 in new_size:
        label = 1
        return 0, 0, label
    src = np.array(Image.fromarray(src).resize(new_size, resample=Image.BICUBIC))
    src_mask = np.array(Image.fromarray(src_mask).resize(new_size, resample=Image.BICUBIC))
    return src, src_mask, label

def gen_Medium(src,src_mask,src_dst_radio):
    random_number = np.random.uniform(0.3, 0.7)
    rounded_number = round(random_number, 1)
    new_size = (int(src.shape[1] / src_dst_radio * rounded_number), int(src.shape[0] / src_dst_radio * rounded_number))
    label = 0
    if 0 in new_size:
        label = 1
        return 0, 0, label
    src = np.array(Image.fromarray(src).resize(new_size, resample=Image.BICUBIC))
    src_mask = np.array(Image.fromarray(src_mask).resize(new_size, resample=Image.BICUBIC))
    return src, src_mask, label

def gen_Small(src,src_mask,src_dst_radio):
    random_number = np.random.uniform(0.1, 0.3)
    rounded_number = round(random_number, 1)
    new_size = (int(src.shape[1] / src_dst_radio * rounded_number), int(src.shape[0] / src_dst_radio * rounded_number))
    label = 0
    if 0 in new_size:
        label = 1
        return 0, 0, label
    src = np.array(Image.fromarray(src).resize(new_size, resample=Image.BICUBIC))
    src_mask = np.array(Image.fromarray(src_mask).resize(new_size, resample=Image.BICUBIC))
    return src, src_mask, label


def paste_image_with_mask(img_a, mask, img_b, center):
    img_a = Image.fromarray(img_a)
    mask = Image.fromarray(mask)
    img_b = Image.fromarray(img_b)
    mask_data = mask.load()
    width, height = img_a.size
    mask_width, mask_height = mask.size
    left = center[0] - mask_width // 2
    top = center[1] - mask_height // 2

    if left < 0:
        left = 0
    if top < 0:
        top = 0
    if left + mask_width > img_b.width:
        left = img_b.width - mask_width
    if top + mask_height > img_b.height:
        top = img_b.height - mask_height
    result_mask = Image.new("L", img_b.size, 0)
    result_mask_data = result_mask.load()
    for x in range(mask_width):
        for y in range(mask_height):
            if sum(mask_data[x, y]) > 0:
                img_b.putpixel((left + x, top + y), img_a.getpixel((x, y)))
                result_mask_data[left + x, top + y] = 255
    return np.array(img_b), result_mask




