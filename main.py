import random

from PIL import Image
from glob import glob
import numpy as np
from scipy.spatial import distance_matrix
import cv2
import time
import sys

def progressbar(it, full, start=0, prefix="",size=60 , out=sys.stdout):  # Python3.6+
    remaining = time.time() - start

    mins, sec = divmod(remaining, 60)
    time_str = f"{int(mins):02}:{sec:05.2f}"
    x = int(size*it/full)

    print(f"\r{prefix}[{u'█' * x}{('.' * (size - x))}] {it}/{full} Est wait {time_str}", end='', file=out, flush=True)

def split_image(image_path, rows, columns):
    # Open the image
    img = Image.open(image_path)
    img = img.resize((5000, 5000))
    img_array = np.array(img)

    # Get dimensions of the original image
    height, width, channels = img_array.shape

    # Calculate width and height of each grid cell
    cell_height = height // rows
    cell_width = width // columns
    res = []
    # Loop through each cell in the grid
    for i in range(rows):
        res.append([])
        for j in range(columns):
            # Calculate coordinates of the cell
            top = i * cell_height
            bottom = (i + 1) * cell_height
            left = j * cell_width
            right = (j + 1) * cell_width

            # Crop the cell from the original image
            cell_img = img_array[top:bottom, left:right, :]

            res[i].append(cell_img)
    return res

def folder_to_images(folder_name):
    files = glob(folder_name + "/*")
    res = []
    for file in files:
        img = Image.open(file)
        res.append(np.array(img))
        img.close()

    return [r for r in res if len(r.shape) == 3]


def compute_similarity(matrix1, matrix2):
    # return distance_matrix(matrix1, matrix2)
    return np.sum((matrix1-matrix2)**2)
    return np.mean((matrix1-matrix2)**2)


def average_color(image):
    return np.average(np.average(image, axis=0), axis=0)

def blend_images(image, image2, percentage):
    avg_color = average_color(image2)
    percentage = max(0, min(100, percentage))
    percentage = percentage / 100.0
    mask = np.zeros_like(image)
    mask[:] = avg_color
    blended_image = cv2.addWeighted(image, percentage, mask, 1 - percentage, 0)
    return blended_image


def mosaic(image, folder):
    tiles = [[image, average_color(image)] for image in folder_to_images(folder)]
    split = [[[image, average_color(image)] for image in x] for x in split_image(image, 100, 100)]
    
    res = [[None for _ in range(100)] for _ in range(100)]
    # void = []

    tiles = [[t, 6] for t in tiles]
    indexes = [[x, y] for x in range(100) for y in range(100)]
    random.shuffle(indexes)
    iter = 0
    start = time.time()
    for i, j in indexes:
        fimg = tiles[0][0]
        min = 99999999999999999999
        index = 0
        for ind, img in enumerate(tiles):
            # tmp = abs(img[0][1] - split[i][j][1])
            tmp = np.sum((img[0][1] - split[i][j][1])**2)
            if min > tmp:
                fimg = img
                index = ind
                min = tmp
        res[i][j] = blend_images(fimg[0][0], split[i][j][0], 50)
        fimg[1] -= 1
        if fimg[1] == 0:
            tiles.pop(index)
        if iter % 100 == 0:
            progressbar(iter / 100 + 1, 100, start, "assing")
        iter += 1
    concatenated_image = np.hstack(np.hstack(res))
    Image.fromarray(concatenated_image).show()

def mosaicos(image, folder):
    print(image)
    tiles = folder_to_images(folder)
    split = split_image(image, 100, 100)
    res = [[None for _ in range(100)] for _ in range(100)]
    # void = []

    tiles = [[t, 6] for t in tiles]
    indexes = [[x, y] for x in range(100) for y in range(100)]
    random.shuffle(indexes)
    iter = 0
    start = time.time()
    for i, j in indexes:
        fimg = tiles[0][0]
        min = 99999999999999999999
        index = 0
        for ind, img in enumerate(tiles):
            tmp = compute_similarity(split[i][j], img[0])
            if min > tmp:
                fimg = img
                index = ind
                min = tmp
        res[i][j] = blend_images(fimg[0], split[i][j], 50)
        fimg[1] -= 1
        if fimg[1] == 0:
            tiles.pop(index)
        if iter % 100 == 0:
            progressbar(iter / 100 + 1, 100, start, "assing")
        iter += 1
    concatenated_image = np.hstack(np.hstack(res))
    Image.fromarray(concatenated_image).show()

mosaic("amazonne.jpeg", "crop_panties")