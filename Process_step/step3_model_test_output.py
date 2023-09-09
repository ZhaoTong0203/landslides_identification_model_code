"""
1.加载模型
2.加载测试数据
3.模型输出测试结果
4.将测试结果按照加载数据返回的名称保存到相应路径下
"""
import torch
import numpy as np
import os
import tifffile as tiff
from torch.utils.data import DataLoader

from Model.FCN import FCN8
from Model.UNet import UNet
from Model.SegNet import SegNet
from Model.DeeplabV3plus import DeepLabV3plusT
from Model.Attention_Deeplabv3plus import AttentionDeeplabV3plus

# 数据加载导入
from step1_data_load import ThreeNonTrainDataset


def model_result_output(device, model, dataset, path):

    test_model = model
    test_device = device
    test_dataset = dataset
    images_dir_save_path = path

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1
    )
    with torch.no_grad():
        model.eval()

        for image, image_name in test_dataloader:
            image = image.to(test_device)
            print(image_name)
            image_name = str(image_name)
            image_name = image_name[2:-3]
            image_save_path = os.path.join(str(images_dir_save_path), image_name)

            output = test_model(image)

            predicted = (output > 0.5).float()  # 打标签0，1。
            predicted = predicted * 255
            predicted = predicted.numpy()

            tiff.imsave(image_save_path, predicted)
            print(f"{image_name}保存成功")


if __name__ == "__main__":
    set_test_device = torch.device("cpu")

    set_test_landslides_dataset = ThreeNonTrainDataset(
        image_dir=r"",
        segmentation_label_dir=r"",
        state="test",
    )

    # 模型加载方式1：直接加载模型
    # set_test_model_file_path = r""
    # set_test_model = torch.load(set_test_model_file_path)

    # 模型加载方式2：先构建模型，再加载模型参数
    set_model_params_file_path = r""
    set_test_model = AttentionDeeplabV3plus(num_classes=1)
    set_test_model.load_state_dict(torch.load(set_model_params_file_path))

    result_save_path = r""

    model_result_output(
        device=set_test_device,
        model=set_test_model,
        dataset=set_test_landslides_dataset,
        path=result_save_path
    )
