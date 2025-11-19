# DL_NLP_PyTorch-Image-Captioning-with-Attention

Image Captioning system implemented in PyTorch with a CNN+Transformer architecture and attention. The project includes a modular training pipeline, configuration-driven setup, and a Flask web app for interactive caption generation with a modern UI.

Status: active development. Interfaces may change; refer to the repo’s issues for known gaps.

---

## Features

- End-to-end pipeline for image captioning on the Flickr8k dataset
- Clear, configuration-driven design (YAML configs for data and model params)
- Modular pipeline stages (ingestion, validation, preprocessing, dataset, dataloaders, encoder/decoder, training, evaluation)
- Transformer-based encoder-decoder with attention
- Logging to both console and file
- Flask app for simple image upload and caption generation

---

## Tech Stack

- Python, PyTorch, TorchVision
- spaCy for text processing
- Transformers (Hugging Face)
- Flask for web app (with a modern, responsive UI)
- Pandas, NumPy, Matplotlib, Seaborn, TQDM

---

## Repository Structure

```
DL_NLP_PyTorch-Image-Captioning-with-Attention/
│
├── .git/                          # Git repository
├── .gitignore                     # Git ignore rules
│
├── Image_Captioning/              # Main package directory
│     ├── main.py                    # Entry point for training pipeline
│     ├── params.yaml                # Model & training hyperparameters
│     ├── .env.example               # Environment variables template
│     ├── environment.yml            # Conda environment file
│     │
│     ├── app/                       # Flask web application
│     │   ├── app.py                 # Flask app with routes and model loading
│     │   └── template.html          # HTML template for image upload UI
│     │
│     ├── config/                    # Configuration files
│     │   └── config.yaml            # Dataset paths, artifact directories
│     │
│     ├── src/                       # Source code package
│     │   ├── __init__.py
│     │   │
│     │   ├── components/            # Core components
│     │   │   ├── __init__.py
│     │   │   ├── data_ingestion.py          # Download & extract dataset
│     │   │   ├── data_validation.py         # Validate data integrity
│     │   │   ├── data_loader.py             # PyTorch DataLoader wrapper
│     │   │   ├── dataset.py                 # CaptioningDataset class
│     │   │   ├── text_preprocessing.py      # TextPreprocessor (vocab, tokenization)
│     │   │   ├── image_preprocessing.py     # ImagePreprocessor (transforms)
│     │   │   ├── encoder.py                 # CNN encoder
│     │   │   ├── decoder.py                 # Transformer decoder
│     │   │   ├── encoder_decoder.py         # Combined encoder-decoder model
│     │   │   ├── model_trainer.py           # Trainer class
│     │   │   ├── model_evaluation.py        # Evaluator class
│     │   │   └── prediction.py              # Prediction utilities
│     │   │
│     │   ├── config/                # Configuration management
│     │   │   ├── __init__.py
│     │   │   └── configuration.py           # ConfigurationManager (singleton)
│     │   │
│     │   ├── constants/             # Constants
│     │   │   ├── __init__.py
│     │   │   └── constants.py               # Config/params file paths
│     │   │
│     │   ├── core/                  # Core utilities
│     │   │   ├── __init__.py
│     │   │   └── singleton.py                # Singleton metaclass
│     │   │
│     │   ├── entity/                # Data entities
│     │   │   ├── __init__.py
│     │   │   └── config_entity.py           # Config dataclasses
│     │   │
│     │   ├── modules/               # Reusable modules
│     │   │   ├── __init__.py
│     │   │   ├── attention.py               # Bahdanau attention mechanism
│     │   │   ├── encoder_cnn.py             # CNN backbone (EfficientNet)
│     │   │   ├── positional_encoder.py      # Positional encoding for Transformer
│     │   │   └── collate.py                 # Custom collate function for DataLoader
│     │   │
│     │   ├── pipeline/              # Pipeline stages
│     │   │   ├── __init__.py
│     │   │   ├── stage_01_data_ingestion.py
│     │   │   ├── stage_02_data_validation.py
│     │   │   ├── stage_03_text_preprocessing.py
│     │   │   ├── stage_04_image_preprocessing.py
│     │   │   ├── stage_05_dataset.py
│     │   │   ├── stage_06_data_loader.py
│     │   │   ├── stage_07_encoder.py
│     │   │   ├── stage_08_decoder.py
│     │   │   ├── stage_09_encoder_decoder.py
│     │   │   ├── stage_10_model_trainer.py
│     │   │   └── stage_11_model_evaluation.py
│     │   │
│     │   └── utils/                 # Utility functions
│     │       ├── __init__.py
│     │       ├── logging_setup.py           # Logger singleton
│     │       ├── device.py                  # Device selection (CPU/GPU)
│     │       └── helpers.py                 # Helper functions
│     │   
│     │
│     ├── data/                      # Dataset directory
│     │   └── 01_raw/                # Raw downloaded data
│     │
│     ├── models/                    # Model checkpoints
│     │
│     ├── notebooks/                 # Jupyter notebooks
│     │
│     ├── reports/                   # Reports and figures
│     │   └── figures/
│     │
│     ├── logs/                      # Log files
│     │   └── running_logs.log
│     │
│     └── docs/                      # Documentation
│   
├── requirements.txt               # Python dependencies
├── setup.py                       # Package setup configuration
├── LICENSE                        # MIT License
└── README.md                      # This file
```

### Key Directories Explained

- **Image_Captioning/src/components/**: Core ML components (data loading, preprocessing, models)
- **Image_Captioning/src/pipeline/**: Orchestration stages that wire components together
- **Image_Captioning/src/modules/**: Reusable neural network modules (attention, encoders)
- **Image_Captioning/app/**: Flask web application for interactive inference
- **Image_Captioning/data/**: Dataset storage (auto-populated by data ingestion)
- **Image_Captioning/logs/**: Training and runtime logs

---

## Getting Started

### 1) Prerequisites

- Python >= 3.8 (setup.py requires >=3.7; 3.8+ recommended)
- pip or conda
- CUDA-capable GPU (optional but recommended for training)

### 2) Create environment and install dependencies

Using pip and a venv (example):

```
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# optional: install the package in editable mode
pip install -e .
```

Notes:
- The spaCy English model is pinned via requirements and should be installed automatically.

---

## Configuration

There are two primary config files under Image_Captioning/:

- config/config.yaml: dataset locations and artifact directories (download_dir, images_dir, caption_path, model_dir, etc.). By default, it’s configured for Flickr8k and writes under data/ and artifacts/.
- params.yaml: preprocessing, model, and training hyperparameters (frequency threshold, image transforms, transformer dims, epochs, LR, etc.).

You can customize these before running.

---

## Dataset

- The pipeline is set up for Flickr8k. Download URLs and filenames are defined in config/config.yaml under dataset:.
- The data ingestion stage will download and prepare the data under data/01_raw by default.

If you already have the dataset, you can place it in the paths indicated by config/config.yaml and skip the automatic download.

---

## Training (Pipeline)

The training pipeline is orchestrated by Image_Captioning/main.py, which wires stages:
- Data ingestion
- Data validation
- Text preprocessing (vocab build/load)
- Image preprocessing (transforms)
- Dataset preparation (train/dev/test)
- DataLoaders
- Encoder
- Decoder
- Encoder–Decoder assembly
- Trainer

Run:

```
# from repo root
python Image_Captioning/main.py
```

Outputs and artifacts:
- Models saved under artifacts/models (see config/config.yaml: training.model_dir, model_name/model_save_prefix)
- Training logs under logs/running_logs.log
- Reports under artifacts/reports

Note: The codebase is under active development. If you encounter interface mismatches between stages (e.g., updated signatures), check recent commits or open an issue with the error trace.

---

## Evaluation and Inference (Programmatic)

A minimal programmatic example using the Evaluator class (Image_Captioning/src/pipeline/stage_11_model_evaluation.py):

```
from pathlib import Path
import torch
from src.pipeline.stage_11_model_evaluation import Evaluator
from src.components.encoder_decoder import TransformerImageCaptioningModel
from src.config.configuration import ConfigurationManager
from src.pipeline.stage_03_text_preprocessing import TextPreprocessingPipeline
from src.pipeline.stage_04_image_preprocessing import ImageProcessingPipeline

# 1) Build config and preprocessors
config_manager = ConfigurationManager()
text_preproc = TextPreprocessingPipeline(config_manager).run_pipeline()
image_preproc = ImageProcessingPipeline(config_manager).run_pipeline()

# 2) Load your trained model
# Replace with the correct model class/loader used during training
model = TransformerImageCaptioningModel(...)
state = torch.load("artifacts/models/<your_checkpoint>.pth", map_location="cpu")
model.load_state_dict(state)

# 3) Create evaluator
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
evaluator = Evaluator(model=model, text_preprocessor=text_preproc, device=DEVICE)

# 4) Generate a caption for a single image
caption_words = evaluator.generate_caption(
    image_path="/path/to/image.jpg",
    beam_size=3,
    image_transform=image_preproc.transform_pipeline,
)
print("Caption:", " ".join(caption_words))
```

Adjust the model loading to match the exact class used in your training stage.

---

## Web App (Flask) for Captioning

A lightweight Flask web application is included under `Image_Captioning/app/` for interactive image captioning. Users can upload images and receive generated captions in real-time.

### Flask App Setup

#### Step 1: Prepare the Model Checkpoint

Before running the Flask app, ensure you have a trained model checkpoint:

1. Train the model using the pipeline (see [Training (Pipeline)](#training-pipeline) section)
2. Locate your saved checkpoint (e.g., `artifacts/models/transformer_caption_model_20240715_090900.pth`)
3. Update `MODEL_CHECKPOINT_PATH` in `Image_Captioning/app/app.py`:

```python
# Line ~95 in app.py
MODEL_CHECKPOINT_PATH = "path/to/your/checkpoint.pth"  # Update this
```

#### Step 2: Prepare the HTML Template

The Flask app expects an HTML template. You have two options:

**Option A: Use the provided template.html (Recommended)**

1. Create a `templates/` folder inside `Image_Captioning/app/`:
   ```
   Image_Captioning/app/
   ├── app.py
   ├── template.html
   └── templates/
       └── index.html
   ```

2. Copy `template.html` to `templates/index.html`:
   ```bash
   # Windows
   copy Image_Captioning\app\template.html Image_Captioning\app\templates\index.html
   
   # macOS/Linux
   cp Image_Captioning/app/template.html Image_Captioning/app/templates/index.html
   ```

**Option B: Modify app.py to use template.html directly**

Edit `Image_Captioning/app/app.py` and change all `render_template('index.html', ...)` calls to `render_template('template.html', ...)`.

#### Step 3: Ensure Configuration is Accessible

The Flask app loads configuration from `Image_Captioning/config/config.yaml` at startup:

- Verify `config/config.yaml` exists and is readable
- Ensure `caption_path` in config points to your captions file (used for vocabulary building)
- The app will attempt to build vocabulary from captions at startup

If captions are unavailable, the app falls back to a minimal vocabulary (see app.py lines ~100-110).

### Running the Flask App

#### From Command Line

```bash
# Navigate to repo root
cd f:\PROJECTS\DL_NLP_PyTorch-Image-Captioning-with-Attention

# Windows (cmd)
set FLASK_APP=Image_Captioning/app/app.py
python Image_Captioning/app/app.py

# Windows (PowerShell)
$env:FLASK_APP = "Image_Captioning/app/app.py"
python Image_Captioning/app/app.py

# macOS/Linux
export FLASK_APP=Image_Captioning/app/app.py
python Image_Captioning/app/app.py
```

#### Direct Python Execution

```bash
# From repo root
python Image_Captioning/app/app.py
```

The app will start on `http://127.0.0.1:5000` (localhost, port 5000).

### Using the Web Interface

1. **Open the app**: Navigate to `http://127.0.0.1:5000` in your browser
2. **Upload an image**: Click "Choose Image" and select a JPG, PNG, or other image format
3. **Generate caption**: Click "Generate Caption"
4. **View results**: The uploaded image and generated caption will appear on the page

### Flask App Features

- **Image Upload**: Accepts common image formats (JPG, PNG, BMP, TIFF, GIF)
- **Real-time Captioning**: Generates captions using beam search (default beam_size=3)
- **Image Display**: Shows the uploaded image alongside the generated caption
- **Temporary Storage**: Uploaded images are stored in `Image_Captioning/app/static/uploads/` (auto-created)
- **Error Handling**: Displays user-friendly error messages if caption generation fails
- **Logging**: All operations logged to console and `logs/running_logs.log`

### Flask App Configuration

Key settings in `Image_Captioning/app/app.py`:

```python
# Model checkpoint path (line ~95)
MODEL_CHECKPOINT_PATH = "captioning_model_20240715_090900.pth"

# Flask server settings (line ~200)
app.run(debug=True, host='0.0.0.0', port=5000)
```

- `debug=True`: Enables auto-reload on code changes (disable for production)
- `host='0.0.0.0'`: Accessible from any network interface
- `port=5000`: Default Flask port (change if needed)

### Troubleshooting Flask App

| Issue                   | Solution                                                                  |
| ----------------------- | ------------------------------------------------------------------------- |
| **Model not found**     | Update `MODEL_CHECKPOINT_PATH` in app.py to correct path                  |
| **Template not found**  | Ensure `templates/index.html` exists or modify render_template calls      |
| **Vocabulary error**    | Verify `captions.txt` is accessible at path in config.yaml                |
| **Port already in use** | Change port in `app.run(port=XXXX)` or kill process using port 5000       |
| **CUDA out of memory**  | Model runs on CPU if CUDA unavailable; reduce batch size if needed        |
| **Slow inference**      | First request may be slow (model loading); subsequent requests are faster |


---

## Logging

Logging is configured via Image_Captioning/src/utils/logging_setup.py.
- Console and file logging
- Default file: logs/running_logs.log

---

## Tips and Troubleshooting

- CUDA/CPU: Training on GPU is recommended. The code automatically selects CUDA if available.
- Missing captions.txt: The app and preprocessing expect captions to be available at config.validation.captions_file or config.create_dataset.caption_path.
- Checkpoints: Ensure you pass the correct checkpoint path. Training saves under artifacts/models by default.
- Version drift: If a pipeline stage complains about a missing or extra parameter, check the corresponding stage file in Image_Captioning/src/pipeline/ and align the usage accordingly.

---

## License

MIT License. See LICENSE for details.

---

## Acknowledgements

- Flickr8k Dataset
- PyTorch and TorchVision
- spaCy
- Hugging Face Transformers

---

