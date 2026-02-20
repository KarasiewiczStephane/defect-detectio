# Defect Detection for Manufacturing

> Computer vision pipeline using ResNet-50 with transfer learning for product defect detection on assembly lines.

## Features

- **Transfer Learning** — ResNet-50 pretrained on ImageNet, fine-tuned for binary defect classification
- **Defect Localization** — Grad-CAM heatmaps highlighting which image regions triggered the prediction
- **Active Learning** — Uncertainty sampling to prioritize the most informative images for labeling
- **Edge Deployment** — ONNX Runtime export for low-latency inference on CPU
- **REST API** — FastAPI endpoints for single-image detection, batch processing, and health checks
- **Batch CLI** — Command-line tool for quality control pipeline integration with JSON/CSV reports
- **Production Ready** — Multi-stage Docker build, GitHub Actions CI, >80% test coverage

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
git clone https://github.com/YOUR_USERNAME/defect-detection.git
cd defect-detection

# Install dependencies
pip install -r requirements.txt

# Create sample data for testing
python scripts/create_sample_data.py

# Run tests
make test

# Start API server
python -m src.main serve --port 8000

# Batch process a directory of images
python -m src.main batch -i /path/to/images -o results/ -m models/model.onnx
```

## Training

```bash
# Train with default config
python -m src.main train --config configs/config.yaml

# Export trained model to ONNX
python -m src.main export --checkpoint checkpoints/best_model.pt --output models/defect_model.onnx
```

## API Usage

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

## Docker

```bash
# Build image
make docker-build

# Run container
make docker-run

# Or use docker compose
docker compose up -d
```

## Project Structure

```
defect-detection/
├── src/
│   ├── main.py                  # CLI entry point
│   ├── data/
│   │   ├── downloader.py        # MVTec AD dataset downloader
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
│   └── utils/
│       ├── config.py            # YAML config manager
│       └── logger.py            # Structured logging setup
├── tests/                       # Unit and integration tests
├── configs/config.yaml          # All configuration values
├── scripts/                     # Utility scripts
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

- **Data** — image size, categories, train/val/test splits, ImageNet normalization stats
- **Model** — architecture, number of classes, pretrained flag
- **Training** — batch size, learning rate, epochs, early stopping patience
- **Active Learning** — uncertainty threshold, samples per round
- **Deployment** — ONNX opset version, benchmark iterations
- **API** — host, port

## Development

```bash
# Install dev dependencies
make install

# Lint and format
make lint

# Run pre-commit hooks
make pre-commit

# Run test suite with coverage
make test
```

## Tech Stack

- **Deep Learning:** PyTorch, torchvision
- **Edge Inference:** ONNX Runtime
- **Computer Vision:** OpenCV
- **API:** FastAPI, uvicorn
- **Testing:** pytest, pytest-cov
- **Linting:** ruff, pre-commit
- **Containerization:** Docker
- **CI/CD:** GitHub Actions

## License

MIT
