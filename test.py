import os

from src.model_a import ModelA

if __name__ == '__main__':
    data_cfg = 'yolo-data-conf.yaml'
    # m1/m2 training, for gpu: [0]
    train_args = {
        # Options include SGD, Adam, AdamW, NAdam, RAdam, RMSProp etc., or auto
        # "optimizer": 'Adam',
        # "momentum": 0.938,
        "freeze": 10,  # the first N layers
        # "batch": -1,
        # "patience": 5,
        # "epochs": 5,
        # cmd tensorboard --logdir runs/train/ (default)
        "tensorboard": True,
        # "log_dir": 'runs/train/',
        "model_name": 'yolov8s_hg'
    }

    # train model, baseline: v8n | v8s | v8m | v8m | v8x
    model = ModelA(model='yolov8s.pt', data_cfg=data_cfg, device='cpu')
    model, result = model.train(project_dir=os.path.join(os.getcwd(), 'data_models', 'my_trained_modelsj', 'j'),
                        conf=train_args)
    print(result)
# Customize validation settings
# validation_results = model.val(data=data_cfg,
#                                imgsz=640,
#                                batch=16,
#                                conf=0.25,
#                                iou=0.6,
#                                device='0')
# print(validation_results)
#
#
# # Resume training
# model = YOLO('path/to/last.pt')
# results = model.train(resume=True)
#
#
# # Benchmark on GPU, such as "cpu", "cuda:0"
# benchmark(model='yolov8n.pt', data='coco8.yaml', imgsz=640, half=False, device=0)
