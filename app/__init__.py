from flask import Flask

from config import Config


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    app.config.update(
        TESTING=True,
        SECRET_KEY='192b9bdd22ab9ed4d12e236c78afcb9a393ec15f71bbf5dc987d54727823bcbf'
    )

    # Initialize Flask extensions here

    # Register blueprints here
    from app.main import bp as main_bp
    app.register_blueprint(main_bp)

    # @app.route('/test/')
    # def test_page():
    #     return '<h1>Testing the Flask Application Factory Pattern</h1>'

    return app
