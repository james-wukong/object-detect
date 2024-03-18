import os

from src.model_a import ModelA

if __name__ == '__main__':
    data_cfg = 'yolo-data-conf.yaml'
    base_model_name = 'yolov8s'
    new_model_name = 'yolov8s_hg'
    owner = 'j'
    device = 'cpu'
    # m1/m2 training, for gpu: [0]
    train_args = {
        # Options include SGD, Adam, AdamW, NAdam, RAdam, RMSProp etc., or auto
        "optimizer": 'Adam',
        "learning_rate": 1e-4,
        "momentum": 0.938,
        "freeze": 10,  # the first N layers
        "batch": -1,
        "patience": 5,
        "epochs": 1,
        # cmd tensorboard --logdir runs/train/ (default)
        "tensorboard": True,
        # "log_dir": 'runs/train/',
        "model_name": new_model_name
    }

    # train model, baseline: v8n | v8s | v8m | v8m | v8x
    model = ModelA(model=f'{base_model_name}.pt', data_cfg=data_cfg, device=device)
    yolo, result = model.train(project_dir=os.path.join(os.getcwd(), 'data_models', 'my_trained_models', owner),
                               conf=train_args)

    metrics = model.val(yolo)
    print(metrics.box.map)

    # model.benchmark(model=os.path.join('data_models',
    #                                         'my_trained_models',
    #                                         owner, new_model_name, 'weights',
    #                                         'best.pt'))

    model.export(yolo)
