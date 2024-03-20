import os

basedir = os.path.abspath(os.path.dirname(__file__))


class Config:
    SECRET_KEY = '192b9bdd22ab9ed4d12e5d2134fcb9a393ec15f71bbf5dc987d54727823bcbf'

    # Specifies the device for inference (e.g., cpu, cuda:0 or 0).
    # Allows users to select between CPU, a specific GPU, or other compute devices
    # for model execution.
    DEVICE = 'cpu'
    # 'cpu', 'cuda'
    SD_DEVICE = 'cpu'

    BASE_PATH = basedir
    UPLOAD_PATH = os.path.join(basedir, 'uploads')
    VIDEO_PATH = os.path.join(basedir, 'videos')
    IMAGE_PATH = os.path.join(basedir, 'image')

    MODEL_BASE_PATH = os.path.join('data_models',
                                   'my_trained_models')
    IMAGE_BASE_PATH = os.path.join(os.getcwd(),
                                   'app', 'static'
                                          'images')
    MODELS = {'yolov8n': os.path.join(MODEL_BASE_PATH, 'j/yolov8n_hat_glass/weights/best.pt'),
              'yolov8s': os.path.join(MODEL_BASE_PATH, 'j/yolov8s_hat_glass/weights/best.pt'),
              'yolov8m': os.path.join(MODEL_BASE_PATH, 'j/yolov8m_hat_glass/weights/best.pt'),
              'yolov8l': os.path.join(MODEL_BASE_PATH, '1'),
              'yolov8x': os.path.join(MODEL_BASE_PATH, '1')
              }
    VIDEO_DEMO = 'app/static/videos/demo.mp4'