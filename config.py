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
    VIDEO_CONF = 0.35
    WEBCAM_CONF = 0.45

    BASE_PATH = basedir
    UPLOAD_PATH = os.path.join(basedir, 'uploads')
    VIDEO_PATH = os.path.join(basedir, 'videos')
    IMAGE_PATH = os.path.join(basedir, 'image')

    MODEL_BASE_PATH = os.path.join('data_models',
                                   'my_trained_models')
    IMAGE_BASE_PATH = os.path.join(os.getcwd(),
                                   'app', 'static', 'images')
    MODELS = {'yolov8n': os.path.join(MODEL_BASE_PATH, 'j/yolov8n/weights/best.pt'),
              'yolov8n_agmt': os.path.join(MODEL_BASE_PATH, 'j/yolov8n_agmt/weights/best.pt'),
              'yolov8s': os.path.join(MODEL_BASE_PATH, 'j/yolov8s/weights/best.pt'),
              'yolov8s_agmt': os.path.join(MODEL_BASE_PATH, 'j/yolov8s_agmt/weights/best.pt'),
              'yolov8m': os.path.join(MODEL_BASE_PATH, 'j/yolov8m/weights/best.pt'),
              'yolov8m_agmt': os.path.join(MODEL_BASE_PATH, 'j/yolov8m_agmt/weights/best.pt'),
              'yolov8x': os.path.join(MODEL_BASE_PATH, 'd/yolov8x_hat_glass_1/weights/best.pt'),
              }
    VIDEO_DEMOS = {'Recorded Video': 'app/static/videos/demo1.mp4',
                   'Realtime Video': 'app/static/videos/demo.mp4',}

    SD_MODEL_ID = 'stabilityai/stable-diffusion-xl-base-1.0'
    ENDPOINT = 'https://api-inference.huggingface.co/models/'