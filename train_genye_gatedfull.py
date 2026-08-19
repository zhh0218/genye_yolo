# 模型配置文件
# model_yaml_path = r"F:\\1YOLO\\YOLOv11RGBD\\ultralytics-main\\ultralytics\\cfg\\models\\11\\yolo11-seg.yaml"/workspace/ultralytics-main/ultralytics/cfg/models/11/yolo11-seg.yaml
model_yaml_path = r"/workspace/ultralytics-main_for_genye/ultralytics/cfg/models/11/yolo11-seg-gatedfull.yaml"
# 数据集配置文件
data_yaml_path = r"/workspace/ultralytics-main_for_genye/data-ubuntu-genye.yaml"
# data_yaml_path = r'F:\\1YOLO\\YOLOv11RGBD\\ultralytics-main\\data.yaml'
# 预训练模型
pre_model_name = r"/workspace/ultralytics-main_for_genye/yolo11l.pt"
# pre_model_name = r'/workspace/ultralytics-main/runs/segment/train56/weights/best.pt'
import warnings

warnings.filterwarnings("ignore")
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(model_yaml_path)
    model.load(pre_model_name)
    model.train(
        data=data_yaml_path,
        imgsz=640,
        epochs=300,
        batch=36,
        workers=2,
        device="0,1,2,3",
        optimizer="MuSGD",
        amp=True,
        project="YOLOv11-RGB-D-coord_attv2-genye",
        # name='MuSGD',
        name="coord_attv2-gatedfull",
    )
