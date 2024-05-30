import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 模型导入
from Model.FCN import FCN8
from Model.UNet import UNet
from Model.SegNet import SegNet
from Model.DeeplabV3plus import DeepLabV3plusT
from Model.Attention_Deeplabv3plus import AttentionDeeplabV3plus

# 损失函数类导入
from Utils.Loss import IoULoss

# 数据打包类导入
from step1_data_load import ThreeTrainDataset, ThreeNonTrainDataset

# 结果评估方法导入
from Utils.Evaluation_metrics import accuracy_metrics, calculate_metrics


def model_train(
        train_model,
        train_epoch_num,
        train_device,
        train_model_save_base_path,
        train_log_save_path,
        train_dataset,
        valid_dataset,
        train_criterion
):
    model = train_model.to(train_device)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=8,                  # 这里设定batch_size超参数，具体大小按照具体任务，模型级硬件配置设定
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1
    )

    train_loss = 1.0      # 设定初始损失值

    for epoch in range(1, train_epoch_num + 1):
        model.train()
        # 参数设置
        start_time = time.time()
        if train_loss <= 0.001:
            print("训练损失值已小于0.001，中止训练")
            break

        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        for images, labels in train_dataloader:
            images = images.to(train_device)
            labels = labels.to(train_device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = train_criterion(outputs, labels)
            # print(f"epoch:{epoch}, loss:{loss.item():.5f}")
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)  # 对模型训练时的损失值进行累加

        train_loss /= len(train_dataset)    # 平均损失计算
        train_tem_time = time.time()
        train_time = train_tem_time - start_time     # 记录训练时间

        with torch.no_grad():  # 此处开始的计算不再与梯度有关
            model.eval()  # 将模型转换为验证模式
            val_loss = 0.0  # 初始化验证损失
            # 初始化目标类别和背景类别的各个评价指标参数
            iou_target, iou_background = 0.0, 0.0
            precision_target, precision_background = 0.0, 0.0
            recall_target, recall_background = 0.0, 0.0
            f1_target, f1_background = 0.0, 0.0
            image_accuracy = 0.0

            for image, label in valid_dataloader:
                image = image.to(train_device)
                label = label.to(train_device)

                output = model(image)
                val_loss += train_criterion(output, label).item()  # 计算总损失，但无优化操作

                predicted = (output > 0.5).float()  # 打标签0，1。
                predicted = predicted.to("cpu")
                label = label.to("cpu")
                predicted = predicted.numpy()
                label = label.numpy()

                accuracy = accuracy_metrics(predicted_label=predicted, true_label=label)
                image_accuracy = image_accuracy + accuracy

                target_precision, target_recall, target_f1_score, target_iou = calculate_metrics(
                    predicted_label=predicted,
                    true_label=label,
                    predicted_label_pixel_value=1,
                    true_label_pixel_value=1
                )
                iou_target = iou_target + target_iou
                precision_target = precision_target + target_precision
                recall_target = recall_target + target_recall
                f1_target = f1_target + target_f1_score

                background_precision, background_recall, background_f1_score, background_iou = calculate_metrics(
                    predicted_label=predicted,
                    true_label=label,
                    predicted_label_pixel_value=0,
                    true_label_pixel_value=0
                )
                iou_background = iou_background + background_iou
                precision_background = precision_background + background_precision
                recall_background = recall_background + background_recall
                f1_background = f1_background + background_f1_score

            val_loss /= len(valid_dataset)
            iou_target /= len(valid_dataset)
            precision_target /= len(valid_dataset)
            f1_target /= len(valid_dataset)
            recall_target /= len(valid_dataset)
            iou_background /= len(valid_dataset)
            precision_background /= len(valid_dataset)
            f1_background /= len(valid_dataset)
            recall_background /= len(valid_dataset)
            image_accuracy /= len(valid_dataset)

            mean_iou = (iou_target + iou_background) / 2.0
            mean_precision = (precision_target + precision_background) / 2.0
            mean_f1 = (f1_background + f1_target) / 2.0
            mean_recall = (recall_target + recall_background) / 2.0

            val_tem_time = time.time()  # 记录验证结束时间
            valid_time = val_tem_time - train_tem_time  # 记录验证所用时间
            total_time = train_time + valid_time  # 记录训练和验证总时间

            with open(train_log_save_path, "a") as log_file:  # 记录这一epoch下的训练和验证日志
                log_file.write(f"Epoch: [{epoch}/{train_epoch_num}]\n")
                log_file.write(f"   Train Time: {train_time:.2f}s\n")
                log_file.write(f"   Train Loss: {train_loss:.4f}\n")
                log_file.write(f"   Valid Time: {valid_time:.2f}s\n")
                log_file.write(f"   Valid Loss: {val_loss:.4f}\n")
                log_file.write(f"   Total Time: {total_time:.4f}s\n")
                log_file.write(
                    f"  Target Class: IoU={iou_target:.4f}, Precision={precision_target:.4f}, F1_score={f1_target:.4f}, Recall={recall_target:.4f}\n")
                log_file.write(
                    f"  Background Class: IoU={iou_background:.4f}, Precision={precision_background:.4f}, F1_score={f1_background:.4f}, Recall={recall_background:.4f}\n")
                log_file.write(
                    f"  Image: mIoU={mean_iou:.4f}, mPrecision={mean_precision:.4f}, mDice={mean_f1:.4f}, mRecall={mean_recall:.4f}\n")
                log_file.write(
                    f"  Accuracy={image_accuracy:.4f}\n")
                log_file.write("\n")

        if (epoch % 20) == 0:
            torch.save(model, train_model_save_base_path + f"/model_save{epoch}.pth")  # 保存当前状态下的模型及其参数
            model_params = model.state_dict()
            torch.save(model_params, train_model_save_base_path + f"/model_params_save{epoch}.pth")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_landslides_dataset = ThreeTrainDataset(
        image_dir="",
        segmentation_label_dir="",
    )
    valid_landslides_dataset = ThreeNonTrainDataset(
        image_dir="",
        segmentation_label_dir="",
        state="val",
    )

    model = AttentionDeeplabV3plus(num_classes=1)
    par_path = r"预训练.pth"
    model.load_state_dict(torch.load(par_path))

    model_train(
        train_model=model,
        train_epoch_num=100,
        train_device=device,
        train_model_save_base_path="",
        train_log_save_path="record_log.txt",
        train_criterion=IoULoss(),
        train_dataset=train_landslides_dataset,
        valid_dataset=valid_landslides_dataset,
    )
