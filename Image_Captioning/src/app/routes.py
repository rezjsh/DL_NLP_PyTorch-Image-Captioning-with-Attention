# app/routes.py

from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app, jsonify
from werkzeug.utils import secure_filename
from ..constants.constants import ALLOWED_EXTENSIONS
from .services.predictor_service import predict_caption
import os

# Blueprint for the main web UI
main_blueprint = Blueprint('main', __name__)


def allowed_file(filename: str) -> bool:
    """Check if the uploaded filename has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _ensure_upload_dir() -> str:
    """Ensure the upload directory exists inside app/static/uploads and return its absolute path."""
    upload_dir = os.path.join(current_app.root_path, 'static', 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    return upload_dir


# No local predictor. Use the shared service.


@main_blueprint.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200


@main_blueprint.route('/', methods=['GET', 'POST'])
def upload_and_predict():
    if request.method == 'POST':
        file = request.files.get('image')
        if not file or file.filename == '':
            flash('No image uploaded')
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash('Unsupported file type. Allowed types: ' + ', '.join(sorted(ALLOWED_EXTENSIONS)))
            return redirect(request.url)

        filename = secure_filename(file.filename)
        upload_dir = _ensure_upload_dir()
        image_path = os.path.join(upload_dir, filename)
        file.save(image_path)

        # Predict caption via shared service
        caption = predict_caption(image_path)
        return render_template('index.html', filename=filename, caption=caption)

    # GET
    return render_template('index.html', filename=None, caption=None)


@main_blueprint.route('/display/<filename>')
def display_image(filename):
    # Serve uploaded images from the static/uploads directory
    return redirect(url_for('static', filename=f'uploads/{filename}'), code=301)
