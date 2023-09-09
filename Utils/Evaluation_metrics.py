import numpy as np


def accuracy_metrics(predicted_label, true_label):
    """这里默认真实标签和模型输出结果的像素值类别指示相同"""
    total_pixel_num = np.size(true_label)
    true_label_flat = true_label.flatten()
    predicted_label_flat = predicted_label.flatten()
    pixel_accuracy_num = np.sum(true_label_flat == predicted_label_flat)
    pixel_accuracy = pixel_accuracy_num / total_pixel_num

    return pixel_accuracy


def calculate_metrics(
        predicted_label,
        true_label,
        predicted_label_pixel_value,
        true_label_pixel_value
):

    """
    step1: 将输入的predicted_label, true_label的目标类别按照相应的类别像素值进行转换
    step2: 计算混淆矩阵
    step3: 按照混淆矩阵中的各个元素计算指标
    :param predicted_label: 预测标签，输入要求是numpy中的ndarray数据格式
    :param true_label: 真实标签，输入要求是numpy中的ndarray数据格式
    :param predicted_label_pixel_value: 目标类别在预测结果上对应的像素值
    :param true_label_pixel_value: 目标类别在标签上对应的像素值
    :return:precision, recall, f1_score, iou 这四个指标都是二级指标
    """
    # step1:数据格式转换
    predicted_label = (predicted_label == predicted_label_pixel_value).astype(float)
    predicted_label = predicted_label.flatten()
    true_label = (true_label == true_label_pixel_value).astype(float)
    true_label = true_label.flatten()

    # step2:混淆矩阵计算
    true_positive = np.sum((true_label == 1) & (predicted_label == 1))
    false_positive = np.sum((true_label == 0) & (predicted_label == 1))
    true_negative = np.sum((true_label == 0) & (predicted_label == 0))
    false_negative = np.sum((true_label == 1) & (predicted_label == 0))

    # step3:评价指标计算
    precision = true_positive / (true_positive + false_positive + 1e-5)
    recall = true_positive / (true_positive + false_negative + 1e-5)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-5)
    iou = true_positive / (true_positive + false_negative + false_positive + 1e-5)

    return precision, recall, f1_score, iou
