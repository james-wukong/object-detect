import os

basedir = os.path.abspath(os.path.dirname(__file__))


class Config:
    BASE_PATH = basedir
    UPLOAD_PATH = os.path.join(basedir, 'uploads')
    VIDEO_PATH = os.path.join(basedir, 'videos')
    IMAGE_PATH = os.path.join(basedir, 'image')