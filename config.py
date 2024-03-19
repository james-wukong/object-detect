import os

basedir = os.path.abspath(os.path.dirname(__file__))


class Config:
    BASE_PATH = basedir
    UPLOAD_PATH = os.path.join(basedir, 'uploads')
    VIDEO_PATH = os.path.join(basedir, 'videos')
    IMAGE_PATH = os.path.join(basedir, 'image')
    SECRET_KEY = '192b9bdd22ab9ed4d12e5d2134fcb9a393ec15f71bbf5dc987d54727823bcbf'