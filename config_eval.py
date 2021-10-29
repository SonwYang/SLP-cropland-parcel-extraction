import glob
import os

pwd = os.path.dirname(os.path.abspath(__file__))


class ConfigEval(object):
    # 影像切片的参数设置
    data_path = 'D:/2021/3/EESNet_test/test/test.tif'          # 测试数据路径,例如： "test"
    # data_path = glob.glob('J:/L1/科目一测试数据/地表分类/greenhouse/test2/*/*.tif')
    # data_path = 'test2'          # 测试数据路径,例如： "test"
    save_path = os.path.join(pwd, 'predict')      # 输出路径
    crop_size = 384         # 切片大小
    padding_size = 64       # 切片镜像延展大小

    # 模型参数设
    model_path = os.path.join(pwd, 'model_results')         # 模型路径,例如： "model_results"
    seg_model_path = 'D:/2021/3/EESNet_test/model_results/epoch113_model.pth'
    model_tta = True        # test time augmentation
    threshold = 0.5      # 概率图阈值

    # 影像后处理参数设置
    TTA = True
    remove_area = 1024      # 移除小物体时，面积阈值设置
    fill_area = 2048         # 填充孔洞时，面积阈值设置

