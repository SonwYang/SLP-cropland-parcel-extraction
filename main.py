import os
import glob
from osgeo import gdal, gdalconst
import subprocess
import gdalTools
import cv2 as cv
import numpy as np
from tqdm import tqdm
from config_eval import ConfigEval
import sys
from Redundancy_predict import predict
from Redundancy_predict_segmentation import predict_seg
from osgeo import ogr,osr
import os
import gdalTools
from shutil import rmtree, copyfile
from skimage import morphology
import argparse

def polygonize(imagePath, raster_path, forest_shp_path):
    pwd = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.realpath(os.path.join(pwd, 'polygonize/')))
    polygonize_exe = os.path.realpath(os.path.join(pwd, 'polygonize/polygonize0529.exe'))
    # polygonize_exe = os.path.realpath(os.path.join(pwd, 'polygonize/polygonize1126.exe'))
    polygonize_path = os.path.realpath(os.path.join(pwd, 'polygonize/polygonize.config'))

    rmHole = "64"
    simpoly = "2"

    scale = "3"
    with open(polygonize_path,'w') as f_config:
        f_config.write("--image=" + imagePath+'\n')
        f_config.write("--edgebuf="+raster_path+'\n')
        f_config.write("--line="+forest_shp_path+'\n')
        f_config.write("--rmHole=" + rmHole + '\n')
        f_config.write("--simpoly=" + simpoly + '\n')
        f_config.write("--scale=" + scale)
    f_config.close()
    subprocess.call(polygonize_exe)


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

## python main.py --img D:\MyWorkSpace\myProject\SLP-CroplandExtraction\images\test.tif --weights ./ckpts/SOED3.pth --shp D:\MyWorkSpace\myProject\SLP-CroplandExtraction\results\result.shp
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, default=r'D:\2021\3\EESNet_test\test\test.tif', help='the path of image')
    parser.add_argument('--weights', type=str, default='D:/MyWorkSpace/paper/plough2/cropland_prediction/model_results/SOED3.pth', help='the path of weights')
    parser.add_argument('--shp',  type=str, default=r'D:\MyWorkSpace\paper\plough\data\SOED2_cq\results.shp', help='the out path of shapefile')
    args = parser.parse_args()

    import time
    t1 = time.time()
    print('starting parsing arguments')
    cfg = ConfigEval()

    cfg.data_path = args.img
    cfg.model_path = args.weights
    outPath = args.shp
    outRoot = os.path.split(outPath)[0]
    outName = os.path.split(outPath)[-1].split('.')[0]

    print('starting inferring edge')
    predict(cfg)
    print('starting inferring binary map')
    predict_seg(cfg)
    out_shp_path = cfg.save_path + '_shp'
    image_root = cfg.data_path
    if os.path.exists(out_shp_path):
        rmtree(out_shp_path)

    mkdir(out_shp_path)
    mkdir(outRoot)

    imgList = [cfg.data_path]
    print('starting polygon')

    for imgPath in imgList:

        imageName = os.path.split(imgPath)[-1].split('.')[0]
        rasterPath = glob.glob(cfg.save_path + f'_ms/{imageName}*tif')[0]
        assert os.path.exists(imgPath), print(f'please cheack {imgPath}')
        assert os.path.exists(rasterPath), print(f'please cheack {rasterPath}')
        print(rasterPath)
        shpName = imageName + '_final.shp'
        shpPath = os.path.join(out_shp_path, shpName)
        out_skeleton_path = os.path.join(outRoot, 'skeleton.tif')

        im_proj, im_geotrans, im_width, im_height, edge_dn = gdalTools.read_img(rasterPath)
        ret2, th2 = cv.threshold(edge_dn, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        th2 = np.where(th2 > 0, 1, 0)
        skeleton = morphology.skeletonize(th2)
        skeleton = np.where(skeleton > 0, 255, 0).astype(np.uint8)
        gdalTools.write_img(out_skeleton_path, im_proj, im_geotrans, skeleton)

        polygonize(imgPath, out_skeleton_path, shpPath)

        assert os.path.exists(shpPath), f"please check your shapefile path:{shpPath}"

        rasterPath_seg = glob.glob(cfg.save_path + f'_seg/{imageName}*tif')[0]
        gdalTools.ZonalStatisticsAsTable(rasterPath_seg, shpPath)

        shp_baseName = os.path.basename(shpPath).split('.')[0]
        shp_root = os.path.split(shpPath)[0]
        shpList = glob.glob(f'{shp_root}/{shp_baseName}*')
        files = os.listdir(shp_root)
        for f in files:
            if f[-4:] == '.shp':
                shp_path = os.path.join(shp_root, f)
                ds = ogr.Open(shp_path, 0)
                layer = ds.GetLayer()
                layer.SetAttributeFilter("majority = 1 or majority = 2")

                driver = ogr.GetDriverByName('ESRI Shapefile')
                out_ds = driver.CreateDataSource(outPath)
                out_layer = out_ds.CopyLayer(layer, 'temp')
                del layer, ds, out_layer, out_ds

        polygonPath = outPath
        imgPath = rasterPath_seg
        linePath = os.path.join(outRoot, 'line.shp')
        out_polygon_path = os.path.join(outRoot, 'polygon.tif')
        out_line_path = os.path.join(outRoot, 'line.tif')

        out_line_dn_path = os.path.join(outRoot, 'line_dn.tif')
        gdalTools.pol2line(polygonPath, linePath)
        gdalTools.shp2Raster(linePath, imgPath, out_line_path, nodata=0)
        gdalTools.shp2Raster(polygonPath, imgPath, out_polygon_path, nodata=0)
        copyfile(rasterPath, out_line_dn_path)

        print(f'spend time:{time.time() - t1}s')

        os.system(f'python PatchRefinement.py')
