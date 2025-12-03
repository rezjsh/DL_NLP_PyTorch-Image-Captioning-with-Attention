# run.py (at project root)
import os
from src.app import create_app

if __name__ == '__main__':
    app = create_app()
    host = os.environ.get('FLASK_RUN_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_RUN_PORT', '5000'))
    debug = os.environ.get('FLASK_DEBUG', '1') == '1'
    
    # Run the app
    # The debug reloader will now execute the script without re-running os.chdir()
    app.run(debug=debug, host=host, port=port)