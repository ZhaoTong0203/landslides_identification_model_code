import torch


def model_cpu2gpu(cpu_model_path, gpu_model_path, setting="single"):
    if setting == "single":
        temporary_code = int(input("即将进行模型转换，转换类别：cpu-gpu，是否继续：继续：1；退出：0"))
        if temporary_code == 0:
            return 0
        elif temporary_code == 1:
            cpu_model = torch.load(cpu_model_path)
            gpu_model = cpu_model.to("cuda")
            torch.save(gpu_model, gpu_model_path)
            return 0
        else:
            print("输入指令错误，请重新运行并输入正确的指令，正确指令示例：0 or 1")
            return 0
    elif setting == "batch":
        cpu_model = torch.load(cpu_model_path)
        gpu_model = cpu_model.to("cuda")
        torch.save(gpu_model, gpu_model_path)
        return 0
    else:
        print("函数变量输入错误，请重新运行代码；若处理单个模型，无需输入setting变量；若进行批处理，请输入setting='batch'")
        return 0


def model_gpu2cpu(gpu_model_path, cpu_model_path, setting="single"):
    if setting == "single":
        temporary_code = int(input("即将进行模型转换，转换类别：gpu-cpu，是否继续：继续：1；退出：0"))
        if temporary_code == 0:
            return 0
        elif temporary_code == 1:
            gpu_model = torch.load(gpu_model_path)
            cpu_model = gpu_model.to("cpu")
            torch.save(cpu_model, cpu_model_path)
            return 0
        else:
            print("输入指令错误，请重新运行并输入正确的指令，正确指令示例：0 or 1")
            return 0
    elif setting == "batch":
        gpu_model = torch.load(gpu_model_path)
        cpu_model = gpu_model.to("cpu")
        torch.save(cpu_model, cpu_model_path)
        return 0
    else:
        print("函数变量输入错误，请重新运行代码；若处理单个模型，无需输入setting变量；若进行批处理，请输入setting='batch'")
        return 0


if __name__ == "__main__":
    cpu_model_path = ""
    gpu_model_path = ""
    model_gpu2cpu(cpu_model_path=cpu_model_path, gpu_model_path=gpu_model_path)
