import imgaug.augmenters as iaa
import imgaug.parameters as iap
import imgaug as ia
import cv2
import numpy as np
import os
import math, random
import time
import pandas as pd
import yaml

def generatePolygon(ctrX, ctrY):

    irregularity = clip(angular_variance, 0, 1) * 2 * math.pi / numVerts
    spikiness = clip(radii_variance, 0, 1) * aveRadius

    # generate n angle steps
    angleSteps = []
    lower = (2*math.pi / numVerts) - irregularity
    upper = (2*math.pi / numVerts) + irregularity
    sum = 0
    for i in range(numVerts) :
        tmp = random.uniform(lower, upper)
        angleSteps.append(tmp)
        sum = sum + tmp

    # normalize the steps so that point 0 and point n+1 are the same
    k = sum / (2*math.pi)
    for i in range(numVerts) :
        angleSteps[i] = angleSteps[i] / k

    # now generate the points
    points = []
    angle = math.pi
    for i in range(numVerts):
        r_i = clip( random.gauss(aveRadius, spikiness), 0, 2*aveRadius )
        x = ctrX + r_i*math.cos(angle)
        y = ctrY + r_i*math.sin(angle)

        points.append( (int(x),int(y)) )
        if check_collinear(points, numVerts) == True:
            return []
        if check_intersect(points, numVerts) == True:
            return []
        angle = angle + angleSteps[i]

    return points


def check_collinear(points, numVerts):
    collin_bool = False
    n = len(points)
    if n > 2:
        if collinear(points[n-3], points[n-2], points[n-1]):
            collin_bool = True
    if n == numVerts:
        if collinear(points[n-2], points[n-1], points[0]):
            collin_bool = True
        if collinear(points[n-1], points[0], points[1]):
            collin_bool = True
        
    return collin_bool


def check_intersect(points, numVerts):
    inter_bool = False
    n = len(points)
    for i in range(n-3):
        if intersect(points[i], points[i+1], points[-2], points[-1]):
            inter_bool = True
    if n == numVerts:
        for i in range(1, n-2):
            if intersect(points[i], points[i+1], points[-1], points[0]):
                inter_bool = True
    return inter_bool


def clip(x, min, max):
    if( min > max ) :  return x    
    elif( x < min ) :  return min
    elif( x > max ) :  return max
    else :             return x


def intersect(A, B, C, D):
    def ccw(A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def collinear(a, b, c):
    crossproduct = (c[1] - a[1]) * (b[0] - a[0]) - (c[0] - a[0]) * (b[1] - a[1])
    if abs(crossproduct) > (aveRadius**2 * collin_coef)/numVerts:
        return False
    else:
        return True


def fit_to_frame(pts, input_size):
    mins = np.min(pts, axis=0)
    maxs = np.max(pts, axis=0)
    for i in range(2):
        if mins[i]<1:
            for pt in pts:
                pt[i] += (abs(mins[i])+20)
        if maxs[i] > input_size:
            for pt in pts:
                pt[i] -= ((maxs[i] - input_size) + 20)
    return pts


def main():
    start = time.time()
    labels_pts = np.empty((img_num, numVerts*2))
    i = 0
    while i < img_num:
        x = np.random.randint(20, input_size-20, 1)[0]
        y = np.random.randint(20, input_size-20, 1)[0]
        pts = generatePolygon(x, y)
        if len(pts) > 0:
            pts = fit_to_frame(np.array(pts), input_size)
            thickness = np.random.randint(4,12)
            pts = pts.reshape((-1, 1, 2))
            img = np.zeros((input_size, input_size, 3), dtype = np.uint8)
            cv2.polylines(img, [pts], True, (181, 243, 255), thickness)
            cv2.fillPoly(img, [pts], color = (95, 113, 228))
            if distort: 
                img = aug(image=img)
            filename = dir + f"/images/img.{i}.jpg"
            cv2.imwrite(filename, img)
            labels_pts[i] = np.ravel(pts)
            i +=1
            if i % 1000 == 0:
                print(f"Passed {i} images.")
    col = [f"p{int(i-i/2)}x" if i % 2 ==0 else f"p{int(i-i/2)}y" for i in range(2, numVerts*2+2)]
    df = pd.DataFrame(labels_pts, columns = col)
    df['ID'] = [i for i in range(img_num)]
    df.to_csv(dir + "/labels_pts.csv")    
    print(f"Generated {i} images. Time spent = {round(time.time() - start,1)} seconds")

if __name__ == "__main__":

    augs = [
        iaa.GaussianBlur(sigma=(0.0, 4.0)),
        iaa.MotionBlur(k=15),
        iaa.MultiplyElementwise((0.5, 1.5), per_channel=0.5), # Noise inside figure
        iaa.Multiply((0.5, 1.5)), #brightness
        iaa.Dropout(p=(0, 0.2)), #random pxls to 0
        iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.25)), #Big rectangular dropout
        iaa.BlendAlphaSimplexNoise(iaa.EdgeDetect(1.0), sigmoid_thresh=iap.Normal(10.0, 5.0)), #Blurred blobs
        iaa.BlendAlphaHorizontalLinearGradient(iaa.AveragePooling(11), start_at=(0.0, 1.0), end_at=(0.0, 1.0)), #Massive blend
        iaa.PiecewiseAffine(scale=(0.01, 0.02)), #Little curve
        iaa.AveragePooling([2, 8]), ]
    aug = iaa.SomeOf((0, None), augs)

    with open("poly_gen_config.yaml", 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    collin_coef = config["collin_coef"]
    aveRadius = config["aveRadius"]
    angular_variance = config["angular_variance"]
    radii_variance = config["radii_variance"]
    numVerts = config["numVerts"]
    img_num = config["img_num"]
    dir = config["dir"]
    distort = config["distort"]
    input_size = config["input_size"]

    main()