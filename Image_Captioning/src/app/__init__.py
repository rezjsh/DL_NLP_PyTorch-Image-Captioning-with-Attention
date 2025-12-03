from flask import Flask
from src.utils.logging_setup import logger
from .routes import main_blueprint

def create_app():
    """
    Application factory for the Flask app.
    """
    logger.info("Creating Flask app...")
    app = Flask(__name__, static_folder='static', template_folder='templates')
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
    app.secret_key = "change_me_in_production"

    # Register blueprints
    
    app.register_blueprint(main_blueprint)

    logger.info("Flask app created.")
    return app