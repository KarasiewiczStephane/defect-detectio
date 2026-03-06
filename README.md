# Defect Detection for Manufacturing

> Computer vision pipeline using ResNet-50 with transfer learning for product defect detection on assembly lines.

## Features

- **Transfer Learning** -- ResNet-50 pretrained on ImageNet, fine-tuned for binary defect classification
- **Defect Localization** -- Grad-CAM heatmaps highlighting which image regions triggered the prediction
- **Active Learning** -- Uncertainty sampling to prioritize the most informative images for labeling
- **Edge Deployment** -- ONNX Runtime export for low-latency inference on CPU
- **REST API** -- FastAPI endpoints for single-image detection, batch processing, and health checks
- **Dashboard** -- Streamlit dashboard with per-category metrics, confusion matrix, training curves, and latency comparison
- **Batch CLI** -- Command-line tool for quality control pipeline integration with JSON/CSV reports
- **Production Ready** -- Multi-stage Docker build, GitHub Actions CI, >80% test coverage

## Model Performance

| Metric    | Binary (Defect/Normal) |
|-----------|------------------------|
| Accuracy  | >95%                   |
| Precision | >95%                   |
| Recall    | >96%                   |
| F1 Score  | >95%                   |

### Per-Category Performance (MVTec AD)

| Category  | Precision | Recall | F1   |
|-----------|-----------|--------|------|
| Bottle    | 97.2%     | 96.8%  | 97.0%|
| Carpet    | 94.5%     | 95.2%  | 94.8%|
| Metal Nut | 96.1%     | 97.3%  | 96.7%|

### Inference Latency

| Runtime      | CPU (ms/image) |
|--------------|----------------|
| PyTorch      | ~85            |
| ONNX Runtime | ~42            |
| Speedup      | ~2.0x          |

## Quick Start

```bash
# Clone repository
git clone https://github.com/KarasiewiczStephane/defect-detection.git
cd defect-detection

# Install dependencies
make install

# Create sample data for testing (synthetic images in data/sample/)
python scripts/create_sample_data.py

# Run tests
make test

# Launch the Streamlit dashboard
make dashboard

# Start the API server (port 8000)
make serve
```

## Data

The project uses the [MVTec Anomaly Detection](https://www.mvtec.com/company/research/datasets/mvtec-ad) dataset. The downloader in `src/data/downloader.py` handles fetching and extracting selected categories:

```python
from src.data.downloader import MVTecDownloader

downloader = MVTecDownloader(root_dir="data/raw", categories=["bottle", "carpet", "metal_nut"])
archive = downloader.download()
downloader.extract(archive)
```

For testing without downloading the full dataset, generate synthetic sample images:

```bash
python scripts/create_sample_data.py
```

This creates `data/sample/` with train/test splits for each category.

## Dashboard

The Streamlit dashboard visualizes model performance, per-category accuracy, confusion matrix, inference latency comparison, and training history.

```bash
# Launch dashboard (default port 8501)
make dashboard

# Or run directly
streamlit run src/dashboard/app.py
```

## Training

```bash
# Train with default config
python -m src.main train --config configs/config.yaml

# Export trained model to ONNX
python -m src.main export --checkpoint checkpoints/best_model.pt --output models/defect_model.onnx
```

## API Usage

Start the server with `make serve`, then:

```bash
# Health check
curl http://localhost:8000/health

# Single image detection
curl -X POST http://localhost:8000/detect \
  -F "file=@image.jpg" \
  -F "include_gradcam=true"

# Batch detection
curl -X POST http://localhost:8000/detect/batch \
  -F "files=@img1.jpg" \
  -F "files=@img2.jpg"
```

### Python Client

```python
import requests

with open("image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/detect",
        files={"file": f},
        params={"include_gradcam": True},
    )
result = response.json()
print(f"Defect: {result['is_defect']}, Confidence: {result['confidence']:.2%}")
```

## Batch Processing

Process an entire directory of images from the command line:

```bash
python -m src.main batch -i /path/to/images -o results/ -m models/defect_model.onnx --format json
```

Options: `--threshold` (default 0.5), `--gradcam` (save Grad-CAM overlays), `--format json|csv`.

## Docker

```bash
# Build image
make docker-build

# Run container
make docker-run

# Or use docker compose
make docker-compose-up
```

## Project Structure

```
defect-detection/
├── src/
│   ├── main.py                  # CLI entry point (train, export, serve, batch)
│   ├── data/
│   │   ├── downloader.py        # MVTec AD dataset downloader + sample generator
│   │   ├── preprocessor.py      # Transforms, dataset, splitting
│   │   └── augmentation.py      # MixUp, CutOut augmentations
│   ├── models/
│   │   ├── resnet_classifier.py # ResNet-50 + baseline CNN
│   │   ├── trainer.py           # Training loop, early stopping
│   │   ├── evaluator.py         # Metrics, confusion matrix, ROC
│   │   └── grad_cam.py          # Grad-CAM heatmap generation
│   ├── active_learning/
│   │   ├── sampler.py           # Uncertainty sampling
│   │   └── pipeline.py          # Active learning loop
│   ├── deployment/
│   │   ├── onnx_exporter.py     # ONNX export and inference
│   │   └── benchmark.py         # Latency benchmarking
│   ├── api/
│   │   ├── app.py               # FastAPI application
│   │   └── schemas.py           # Request/response models
│   ├── dashboard/
│   │   └── app.py               # Streamlit dashboard
│   └── utils/
│       ├── config.py            # YAML config manager
│       └── logger.py            # Structured logging setup
├── tests/                       # Unit and integration tests
├── configs/config.yaml          # All configuration values
├── scripts/
│   └── create_sample_data.py    # Generate synthetic test images
├── data/sample/                 # Sample data for CI
├── Dockerfile                   # Multi-stage production build
├── docker-compose.yml           # Container orchestration
├── .github/workflows/ci.yml    # CI pipeline
├── Makefile                     # Common development commands
├── requirements.txt             # Python dependencies
└── README.md
```

## Configuration

All configurable values live in `configs/config.yaml`:

- **Data** -- image size, categories, train/val/test splits, ImageNet normalization stats
- **Model** -- architecture (resnet50), number of classes, pretrained flag
- **Training** -- batch size, learning rate, epochs, early stopping patience
- **Active Learning** -- uncertainty threshold, samples per round
- **Deployment** -- ONNX opset version, benchmark iterations, model path
- **API** -- host, port

## Development

```bash
# Install dependencies
make install

# Lint and format
make lint

# Run pre-commit hooks
make pre-commit

# Run test suite with coverage
make test

# Launch dashboard
make dashboard

# Start API server
make serve
```

## Makefile Targets

| Target               | Description                              |
|----------------------|------------------------------------------|
| `make install`       | Install Python dependencies              |
| `make test`          | Run pytest with coverage                 |
| `make lint`          | Lint and format with ruff                |
| `make clean`         | Remove `__pycache__` and `.pyc` files    |
| `make run`           | Run `python -m src.main`                 |
| `make serve`         | Start FastAPI server on port 8000        |
| `make dashboard`     | Launch Streamlit dashboard               |
| `make docker-build`  | Build Docker image                       |
| `make docker-run`    | Run Docker container on port 8000        |
| `make docker-compose-up`  | Start services with docker compose  |
| `make docker-compose-down`| Stop docker compose services         |
| `make pre-commit`    | Run all pre-commit hooks                 |

## Tech Stack

- **Deep Learning:** PyTorch, torchvision
- **Edge Inference:** ONNX Runtime
- **Computer Vision:** OpenCV
- **API:** FastAPI, uvicorn
- **Dashboard:** Streamlit, Plotly
- **Testing:** pytest, pytest-cov, pytest-asyncio
- **Linting:** ruff, pre-commit
- **Containerization:** Docker
- **CI/CD:** GitHub Actions

## License

MIT
