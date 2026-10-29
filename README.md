# RupeeVision AI 🇮🇳

**AI-Powered Indian ₹50 Banknote Detection System**

RupeeVision AI is a computer vision project designed to detect Indian ₹50 banknotes using a YOLO-based ONNX model, OpenCV DNN, and a FastAPI backend. It provides a browser-based interface for camera-based detection and image analysis.

## ✨ Features

- Real-time camera-based banknote detection interface
- ONNX model inference using OpenCV DNN
- FastAPI backend for detection requests
- Browser-based frontend built with HTML, CSS, and JavaScript
- Image upload and detection API support
- Configurable confidence and IoU thresholds
- API health and model information endpoints
- Automated backend tests

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Computer Vision | OpenCV |
| Model Format | ONNX |
| AI Model | YOLO-based detector |
| Backend | FastAPI |
| Frontend | HTML, CSS, JavaScript |
| API Server | Uvicorn |
| Testing | Pytest |

## 📁 Project Structure

```text
RupeeVision-AI/
├── backend/
│   ├── app/
│   │   ├── routers/
│   │   ├── config.py
│   │   ├── detector.py
│   │   ├── main.py
│   │   └── schemas.py
│   ├── tests/
│   ├── requirements.txt
│   └── requirements-dev.txt
├── frontend/
│   ├── assets/
│   ├── index.html
│   ├── script.js
│   └── style.css
├── models/
│   └── model.onnx
├── outputs/
├── src/
├── requirements.txt
├── test.py
└── README.md
```

## 💻 Requirements

- Python 3.11 recommended
- Git
- A modern web browser
- A webcam for live detection
- The ONNX model file at `models/model.onnx`

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Santha29/RupeeVision-AI.git
cd RupeeVision-AI
```

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
```

## ▶️ Run the Application

From the project root directory, run:

```powershell
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Open the application in your browser:

**Website:** http://127.0.0.1:8000

**API health:** http://127.0.0.1:8000/api/health

**Model information:** http://127.0.0.1:8000/api/model-info

**API documentation:** http://127.0.0.1:8000/docs

Keep the terminal running while using the application.

## 📷 How to Use

1. Start the application using the command above.
2. Open the website in your browser.
3. Allow camera access when prompted.
4. Start the camera.
5. Click **Start Detection**.
6. Show a ₹50 banknote to the camera and observe the detection output.

Detection results depend on the trained model, lighting conditions, image quality, and camera position.

## 🔌 API Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Check backend health |
| `GET /api/model-info` | Inspect model status and information |
| `POST /api/detect/frame` | Submit a frame for detection |

For request formats and interactive API documentation, start the backend and visit `/docs`.

## ⚙️ Configuration

Application settings can be configured through environment variables:

| Variable | Purpose |
|---|---|
| `RUPEEVISION_MODEL_PATH` | Override the ONNX model path |
| `RUPEEVISION_CONF` | Detection confidence threshold |
| `RUPEEVISION_IOU` | IoU threshold |
| `RUPEEVISION_MAX_UPLOAD_BYTES` | Maximum accepted upload size |
| `RUPEEVISION_CORS_ORIGINS` | Allowed frontend origins |

Default model path: `models/model.onnx`

## 🧪 Testing

Install development dependencies:

```powershell
pip install -r backend/requirements-dev.txt
```

Run the backend tests:

```powershell
python -m pytest backend/tests -v
```

Some tests may require the model file and a correctly configured environment.

## 🌐 Deployment

The application can be deployed to a Python-compatible hosting service that supports FastAPI and the required computer vision dependencies.

For production deployment:
- Configure the service to listen on the hosting provider's assigned port.
- Configure the frontend API URL for the deployed backend.
- Allow the deployed frontend origin in the backend CORS configuration.
- Use HTTPS for browser camera access.
- Verify that the model file is available after deployment.

## 🔒 Security and Privacy

- Do not commit API keys, passwords, or `.env` files.
- Review uploaded images and test data before publishing the repository.
- Camera access requires user permission.
- Configure CORS for trusted origins when deploying publicly.

## ⚠️ Limitations

- Detection accuracy depends on model training and real-world conditions.
- The model's predictions should be validated before practical use.
- This project is a computer vision demonstration and is not a banknote authentication or counterfeit-detection guarantee.

## 👨‍💻 Author

**Santha29**

GitHub: https://github.com/Santha29

## 📄 License

Add your chosen open-source license before distributing the project. Until a license is added, the repository should not be assumed to grant others permission to reuse its code.
