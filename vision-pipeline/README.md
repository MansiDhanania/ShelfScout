# Vision Pipeline

**Version**: 2.2.0 | **Architecture**: Microservice | **Framework**: FastAPI | **Inference**: Ollama Remote | **Validation**: Pydantic v2

A production-grade computer vision microservice that performs **open-vocabulary object detection** with **depth estimation**, **spatial guidance**, and **robust JSON parsing**. Designed for accessibility applications, the service uses **Qwen2.5-VL** vision-language model via remote **Ollama** inference for scalability and memory efficiency.

---

## 📋 Table of Contents

- [Architecture Overview](#architecture-overview)
- [Technical Stack](#technical-stack)
- [Key Features](#key-features)
- [Design Decisions](#design-decisions)
- [Setup & Deployment](#setup--deployment)
- [API Documentation](#api-documentation)
- [Implementation Highlights](#implementation-highlights)
- [Performance Optimization](#performance-optimization)
- [Error Handling](#error-handling)

---

## 🏗️ Architecture Overview

### Service Design Pattern: Microservice Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Vision Pipeline FastAPI Service              │
│                         (Port 5000)                              │
├─────────────────────────────────────────────────────────────────┤
│  HTTP Request Handler (FastAPI)                                 │
│  ├─ Request Validation (Pydantic)                               │
│  ├─ Image Decoding (Base64 → PIL)                               │
│  └─ Response Serialization (JSON)                               │
├─────────────────────────────────────────────────────────────────┤
│  Vision Processing Pipeline                                     │
│  ├─ Smart Image Resize (Qwen's proprietary algorithm)           │
│  ├─ Detection: Ollama → Qwen2.5-VL (Remote)                    │
│  ├─ JSON Repair & Parsing (Multi-strategy Pydantic parser)      │
│  ├─ Bounding Box Normalization & Scaling                        │
│  ├─ Depth Estimation: Depth-Anything-V2 (Optional)              │
│  └─ Spatial Guidance Generation                                 │
├─────────────────────────────────────────────────────────────────┤
│  Resource Management                                            │
│  ├─ Lazy Model Loading (On-demand)                              │
│  ├─ Auto-Unload Timer (Configurable, default 10 min)            │
│  ├─ Memory Monitoring & Reporting                               │
│  └─ CUDA/GPU Memory Management                                  │
├─────────────────────────────────────────────────────────────────┤
│  External Services                                              │
│  └─ Ollama Server (http://host.docker.internal:11434)           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technical Stack

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| **Web Framework** | FastAPI | ≥0.104.0 | Async HTTP API with OpenAPI docs |
| **Async Server** | Uvicorn | ≥0.24.0 | ASGI server for concurrent requests |
| **Validation** | Pydantic | ≥2.0.0 | Type-safe request/response handling + JSON repair |
| **Image Processing** | OpenCV, Pillow | 4.8.0+, 10.0.0+ | Image encoding/decoding and annotation |
| **Vision Models** | Transformers | ≥4.35.0 | HuggingFace model hub integration |
| **Remote Inference** | Ollama | Latest | Qwen2.5-VL model serving (remote) |
| **Depth Estimation** | Depth-Anything-V2 | Latest | Optional depth perception |
| **Numerical Computing** | NumPy | ≥1.24.0 | Array operations and calculations |
| **Containerization** | Docker | Latest | Reproducible deployment |
| **Base Image** | pytorch/pytorch | 2.1+ | Pre-installed PyTorch with CUDA support |

---

## ✨ Key Features

### 1. **Open-Vocabulary Object Detection**
- Detect any object by name without pre-training on fixed class sets
- Uses **Qwen2.5-VL** (3B parameter model) for efficient inference
- Supports custom detection prompts for specialized use cases

### 2. **Robust JSON Parsing with Pydantic**
- Multi-strategy JSON repair for handling malformed Qwen outputs
- Automatic field normalization (handles alternative key names)
- Type validation and coercion for all model outputs
- Graceful fallbacks for malformed data

### 3. **Depth Estimation**
- Integrated **Depth-Anything-V2** for spatial awareness
- Per-bounding-box depth calculation (median, mean, min, max methods)
- Depth included in detection results for 3D spatial understanding

### 4. **Spatial Guidance Generation**
- Converts 2D detections to natural language guidance
- Includes directional information (left, right, straight ahead)
- Vertical positioning relative to eye level
- Distance estimation in spatial units

### 5. **Memory-Efficient Inference**
- **Remote Ollama Integration**: Models run on separate server, not in-process
- **Lazy Model Loading**: Models load only on first inference request
- **Auto-Unload Timer**: Automatically frees resources after configurable idle timeout
- **Manual Unload Endpoint**: Explicit memory release when needed
- **Real-time Memory Monitoring**: Detailed RAM/GPU usage statistics

### 6. **Production-Ready API**
- AsyncIO-based concurrent request handling
- Comprehensive error handling with detailed error messages
- Health check endpoint with service diagnostics
- CORS support for cross-origin requests

---

## 🎯 Design Decisions

### Why Ollama for Inference?

**Problem**: Running large vision models in-process consumes massive GPU/CPU resources, making deployment inflexible and expensive.

**Solution**: Use **remote Ollama server** for model inference.

**Benefits**:
| Aspect | In-Process | Ollama Remote |
|--------|-----------|---------------|
| **Resource Usage** | High (GPU/CPU pinned) | Low (only network calls) |
| **Scalability** | Vertical only | Horizontal (multiple services) |
| **Model Updates** | Service restart needed | Zero downtime |
| **Cost** | GPU per replica | Single shared GPU server |
| **Failure Isolation** | Service dies if model crashes | Service continues, retry gracefully |

**Implementation**:
```python
def call_ollama(image_pil: Image.Image, prompt: str) -> str:
    """
    Encode image as base64, send to Ollama API, receive text response.
    - Timeout: 300s (configurable via OLLAMA_TIMEOUT)
    - Retry logic: Automatic connection verification with 3 retries
    - Connection pooling: Reuses HTTP connections via requests library
    """
```

### Why Pydantic for JSON Repair?

**Problem**: Qwen outputs sometimes produce malformed JSON (unclosed braces, single quotes, placeholder strings, etc.), causing parsing failures.

**Solution**: **Multi-strategy Pydantic validation** with automatic JSON repair.

**Implementation**:
```
Strategy 1: Direct JSON parse (json.loads)
   ↓ fails
Strategy 2: Repair JSON then parse (repair_json_string + json.loads)
   ↓ fails
Strategy 3: Extract objects array directly from malformed JSON
   ↓ fails
Strategy 4: Pydantic field validation with automatic coercion
```

**Repair Techniques**:
- Remove markdown code block markers: ` ```json ... ``` `
- Replace single quotes with double quotes (outside strings)
- Fix double double-quotes: `""` → `"`
- Remove trailing commas: `, }` → ` }`
- Fix missing commas: `}{ ` → `}, {`
- Convert placeholder strings to zeros: `"x_min"` → `0`
- Fix unquoted keys: `{key:` → `{"key":`
- Balance braces/brackets by counting and adding missing ones

**Pydantic Models**:
```python
class BoundingBox(BaseModel):
    coordinates: List[float]
    # Validator handles: dict format, list/tuple, string numbers
    
class DetectedObject(BaseModel):
    label: str              # Handles: None, empty, various field names
    bbox: List[float]       # Delegates to BoundingBox validator
    confidence: float       # Validates 0.0-1.0 range, coerces to float
    
class DetectionOutput(BaseModel):
    objects: List[DetectedObject]
    # Handles both single object and list formats
```

### Why Remote Model Loading?

**Problem**: Loading 3B+ parameter vision models takes 30+ seconds, blocking requests.

**Solution**: **Lazy loading singleton pattern** with background health checks.

```python
class LazyModelLoader:
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls):
        """Singleton: Returns same instance across all requests"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def get_model(self):
        """Load model on-demand, not at startup"""
        with self._lock:
            if self.depth_pipe is None:
                self._load_depth_model()  # Only for depth, not Qwen
        return None, None, self.depth_pipe
```

**Benefits**:
- Fast startup (seconds instead of minutes)
- Only loads if needed
- Singleton ensures single model instance
- Thread-safe with locks

---

## 🚀 Setup & Deployment

### Prerequisites

- **Docker** (20.10+)
- **Ollama Server** running separately (remote inference)
- **GPU** (optional, CPU works but slower)
- **Python** 3.9+ (for local development)

### 1. Local Development Setup

```bash
# Clone repository
git clone https://github.com/MansiDhanania/ShelfScout
cd SLIV/vision_pipeline

# Create Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start Ollama server (in another terminal)
ollama serve

# Pull Qwen model (in another terminal)
ollama pull qwen2.5-vl:3b

# Run FastAPI server
uvicorn vision_app:app --host 0.0.0.0 --port 5000 --reload

# Access API docs
# Open browser to http://localhost:5000/docs
```

### 2. Docker Deployment (Recommended)

```bash
# Build with CPU PyTorch (default)
docker build -t vision-pipeline .

# Or build with GPU PyTorch
docker build --build-arg BASE_IMAGE=pytorch/pytorch:2.1.0-cuda12.1-cudnn9-runtime \
  -t vision-pipeline:gpu .

# Run container
docker run -p 5000:5000 \
  --gpus all \
  -e OLLAMA_URL=http://host.docker.internal:11434 \
  -e OLLAMA_MODEL=qwen2.5vl:3b \
  vision-pipeline

# Check health
curl http://localhost:5000/health
```

### 3. Centralized Deployment (Production)
This service is deployed as part of a larger orchestrated system: DM for full deployment context.

```bash
cd ../centralized-deployment
python start.py essential  # Includes vision pipeline + core services

# Managed via docker-compose.yml
# Services auto-restart on failure
# Logs aggregated to ELK stack
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_URL` | `http://host.docker.internal:11434` | Ollama server endpoint |
| `OLLAMA_MODEL` | `qwen2.5vl:3b` | Model name in Ollama |
| `OLLAMA_TIMEOUT` | `300` | Inference timeout (seconds) |
| `OLLAMA_HEALTH_TIMEOUT` | `5` | Health check timeout (seconds) |
| `OLLAMA_STARTUP_RETRIES` | `3` | Connection retry attempts |
| `OLLAMA_RETRY_DELAY` | `2` | Retry delay (seconds) |
| `AUTO_UNLOAD_TIMEOUT` | `600` | Auto-unload timeout (seconds, 0=disabled) |
| `ENABLE_DEPTH` | `true` | Enable depth estimation |

---

## 📡 API Documentation

### 1. Object Detection (`POST /detect`)

**Purpose**: Detect a specific object in an image and provide spatial guidance.

**Request**:
```bash
curl -X POST http://localhost:5000/detect \
  -H "Content-Type: application/json" \
  -d '{
    "image": "data:image/jpeg;base64,iVBORw0KG...",
    "object": "banana",
    "score_threshold": 0.15,
    "user_pos": [960, 540, 0],
    "eye_level": 1.5,
    "prompt": null
  }'
```

**Response (200 - Success)**:
```json
{
  "success": true,
  "object": "banana",
  "confidence": 0.87,
  "bbox": [370, 120, 504, 293],
  "depth": 2.45,
  "guidance": "The object is approximately 3 units away, to your left, below eye level.",
  "annotated_image": "data:image/jpeg;base64,iVBORw0KG...",
  "all_detections": [
    {
      "label": "banana",
      "score": 0.87,
      "box": [370, 120, 504, 293],
      "depth": 2.45
    },
    {
      "label": "banana",
      "score": 0.72,
      "box": [482, 80, 599, 253],
      "depth": 3.12
    },
    {
      "label": "banana",
      "score": 0.61,
      "box": [250, 200, 350, 350],
      "depth": 2.89
    }
  ]
}
```

**Response (404 - Not Found)**:
```json
{
  "error": "Object 'banana' not found in image",
  "detections": [],
  "annotated_image": "data:image/jpeg;base64,iVBORw0KG...",
  "debug_info": "Total detections: 0, threshold: 0.15"
}
```

**Response (400 - Bad Request)**:
```json
{
  "error": "Invalid base64 image data"
}
```

**Features**:
- ✅ Handles various image formats (JPEG, PNG, WebP)
- ✅ Auto-resizes small images to minimum 224x224
- ✅ Smart image preprocessing per Qwen specifications
- ✅ Top-3 detections returned with color-coded annotations
- ✅ Optional depth estimation per detection
- ✅ Customizable confidence threshold
- ✅ Custom prompt support for specialized detection tasks

---

### 2. Route Details (`POST /route-details`)

**Purpose**: Analyze multiple images along a route to generate scene descriptions for navigation.

**Request**:
```bash
curl -X POST http://localhost:5000/route-details \
  -H "Content-Type: application/json" \
  -d '{
    "images": [
      "data:image/jpeg;base64,iVBORw0KG...",
      "data:image/jpeg;base64,iVBORw0KG..."
    ]
  }'
```

**Response**:
```json
{
  "success": true,
  "route_details": {
    "image_1": {
      "location": "produce section",
      "aisle": "fruits and vegetables aisle",
      "visible_products": ["apples", "bananas", "carrots", "lettuce"],
      "description": "You are in the produce aisle with fresh fruits and vegetables displayed on both sides."
    },
    "image_2": {
      "location": "dairy section",
      "aisle": "refrigerated aisle",
      "visible_products": ["milk", "cheese", "yogurt", "butter"],
      "description": "You are approaching the dairy section. Refrigerated items including milk and cheese are on your right."
    }
  ]
}
```

**Use Cases**:
- Store navigation assistance
- Wayfinding for accessibility
- Scene understanding for visually impaired users
- Route description generation

---

### 3. Health Check (`GET /health`)

**Purpose**: Monitor service health, model status, and resource usage.

**Request**:
```bash
curl http://localhost:5000/health
```

**Response**:
```json
{
  "status": "healthy",
  "service": "vision-pipeline",
  "version": "2.2.0",
  "inference_engine": "Ollama (remote)",
  "ollama": {
    "connected": true,
    "url": "http://host.docker.internal:11434",
    "model": "qwen2.5vl:3b"
  },
  "depth_model": {
    "loaded": true
  },
  "memory": {
    "system": {
      "total_gb": 31.89,
      "available_gb": 18.42,
      "used_gb": 13.47,
      "percent_used": 42.2
    },
    "gpu": {
      "allocated_gb": 5.821,
      "reserved_gb": 6.125,
      "max_allocated_gb": 5.821,
      "total_gb": 11.76,
      "free_gb": 5.939
    }
  }
}
```

**Monitoring**:
- Checks Ollama connectivity
- Reports depth model status
- Provides real-time memory statistics
- Used by Kubernetes/Docker health checks

---

### 4. Unload Models (`POST /unload`)

**Purpose**: Manually free GPU/CPU memory without restarting service.

**Request**:
```bash
curl -X POST http://localhost:5000/unload
```

**Response**:
```json
{
  "status": "unloaded",
  "unloaded_models": ["depth_pipe"],
  "memory_after_unload": {
    "system": {
      "total_gb": 31.89,
      "available_gb": 24.12,
      "used_gb": 7.77,
      "percent_used": 24.4
    },
    "gpu": {
      "allocated_gb": 0.0,
      "reserved_gb": 0.0,
      "max_allocated_gb": 5.821,
      "total_gb": 11.76,
      "free_gb": 11.76
    }
  },
  "note": "Model will be reloaded automatically on next inference request"
}
```

**Use Cases**:
- Free memory before heavy batch processing
- Release resources when service is idle
- Manual memory management before deployments

---

### 5. Hardware Info (`GET /hardware`)

**Purpose**: Get system and model hardware configuration.

**Request**:
```bash
curl http://localhost:5000/hardware
```

**Response**:
```json
{
  "hardware_info": {
    "device": "remote_ollama",
    "inference_engine": "ollama",
    "ollama_url": "http://host.docker.internal:11434"
  },
  "pytorch_version": "2.1.0",
  "inference_engine": "Ollama (remote)",
  "ollama_url": "http://host.docker.internal:11434",
  "model_name": "qwen2.5vl:3b",
  "depth_model_available": true,
  "depth_model_loaded": true,
  "system_info": {
    "platform": "Linux",
    "architecture": "x86_64",
    "python_version": "3.10.12"
  }
}
```

---

### 6. Qwen Preprocessing (`POST /qwen-preprocessing`)

**Purpose**: Resize images using Qwen's exact smart_resize algorithm for external API compatibility.

**Use Case**: When using Fireworks API or other external Qwen endpoints, pre-process images locally and use scale factors for denormalization.

**Request**:
```bash
curl -X POST http://localhost:5000/qwen-preprocessing \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/jpeg;base64,iVBORw0KG..."}'
```

**Response**:
```json
{
  "success": true,
  "resized_image": "data:image/jpeg;base64,iVBORw0KG...",
  "original_size": [1920, 1080],
  "resized_size": [1372, 784],
  "scale_x": 1.3994,
  "scale_y": 1.3776,
  "denormalize_instructions": "To convert Fireworks bbox [x1,y1,x2,y2] to original: [x1*1.3994, y1*1.3776, x2*1.3994, y2*1.3776]"
}
```

**Implementation Details**:
- Respects Qwen's min/max pixel constraints
- Uses BICUBIC resampling for quality
- Returns precise scale factors for coordinate transformation

---

### 7. Root Endpoint (`GET /`)

**Purpose**: Service information and API overview.

**Request**:
```bash
curl http://localhost:5000/
```

**Response**:
```json
{
  "service": "Vision Pipeline API",
  "version": "2.2.0",
  "description": "Object detection with Qwen2.5-VL (via Ollama) and depth estimation",
  "inference_engine": "Ollama (remote)",
  "ollama_url": "http://host.docker.internal:11434",
  "model_name": "qwen2.5vl:3b",
  "endpoints": {
    "detection": "/detect",
    "routing": "/route-details",
    "qwen-preprocessing": "/qwen-preprocessing",
    "health": "/health",
    "hardware": "/hardware",
    "unload": "/unload",
    "docs": "/docs"
  }
}
```

---

## 💻 Implementation Highlights

### 1. Smart Image Resizing (Qwen Algorithm)

```python
def smart_resize(height, width, factor=28, min_pixels=56*56, max_pixels=14*14*4*1280):
    """
    Implements Qwen's proprietary image resize logic.
    
    Algorithm:
    1. Round to nearest multiple of 'factor' (28)
    2. If total pixels exceed max_pixels, scale down
    3. If total pixels below min_pixels, scale up
    
    Ensures images maintain aspect ratio while fitting Qwen's constraints.
    """
```

**Why This Matters**:
- Qwen expects images in specific dimension ranges
- Smart resize maintains quality while optimizing inference speed
- Automatic calculation of scale factors for coordinate transformation

### 2. JSON Repair Strategy

```python
def repair_json_string(json_str: str) -> str:
    """
    Multi-step JSON repair for common Qwen output issues:
    
    1. Strip whitespace and markdown markers
    2. Fix quote issues (single → double)
    3. Remove trailing commas
    4. Balance braces/brackets
    5. Fix unquoted keys
    6. Convert placeholders to values
    7. Validate JSON structure
    """
```

**Real-World Example**:
```python
# Qwen malformed output
malformed = """
{
  'objects': [
    {'label': 'banana', 'bbox': [x_min, y_min, x_max, y_max], 'confidence': 0.95,},
    {'label': 'apple', 'bbox': [100, 50, 200, 150], 'confidence': 0.87}
  ]
}
"""

# After repair
repaired = repair_json_string(malformed)
# Now valid JSON for json.loads()
data = json.loads(repaired)
```

### 3. Pydantic Validation Pipeline

```python
class DetectedObject(BaseModel):
    label: str
    bbox: List[float]
    confidence: float
    
    @field_validator('label', mode='before')
    @classmethod
    def validate_label(cls, v):
        """Convert None/empty to 'unknown'"""
        if v is None:
            return "unknown"
        return str(v).strip() or "unknown"
    
    @model_validator(mode='before')
    @classmethod
    def normalize_field_names(cls, data):
        """Handle alternative field names from Qwen"""
        # Support: label, name, class, object, category, type
        # Support: bbox, bbox_2d, box, bounding_box, coordinates, rect
```

**Benefits**:
- Type safety (Python + IDE autocomplete)
- Automatic validation and coercion
- Clear error messages for invalid data
- Self-documenting API

### 4. Spatial Guidance Generation

```python
def generate_guidance(
    user_pos: Tuple[float, float, float],
    object_pos: Tuple[float, float, float],
    eye_level: float = 1.5
) -> str:
    """
    Convert 3D detection to natural language guidance.
    
    Outputs like:
    - "The object is approximately 3 units away, to your left, below eye level."
    - "The object is straight ahead at eye level."
    - "The object is to your right, above eye level."
    """
```

**Accessibility Applications**:
- Provides directions for visually impaired users
- Converts computer vision to natural language
- Works with text-to-speech systems

---

## ⚡ Performance Optimization

### Memory-Efficient Remote Inference

```
┌─────────────────────────────────────────────────────────┐
│ Vision Service (Container)        │  Ollama Server      │
│                                   │  (Separate)         │
│ Memory Usage:    ~200 MB          │  ~5-7 GB            │
│ ├─ FastAPI:      ~50 MB           │                     │
│ ├─ Dependencies: ~150 MB          │                     │
│ └─ Buffers:      ~10 MB           │                     │
│                                   │                     │
│ Can scale to 10+ replicas         │  Single GPU shared  │
│ Cost-effective                    │  between all        │
└─────────────────────────────────────────────────────────┘
```

### Benchmarks (RTX 3080 GPU)

| Operation | Time | Notes |
|-----------|------|-------|
| Ollama startup | 5-10s | One-time |
| Model load (Qwen 3B) | 3-5s | Via Ollama, remote |
| Single inference | 2-4s | Depends on image size |
| Depth estimation | 1-2s | Optional, per-detection |
| Full /detect response | 4-7s | Includes annotation |

### Optimization Techniques

1. **Remote Model Serving**: Models don't consume local resources
2. **Lazy Loading**: Only load on first request
3. **Auto-Unload**: Free memory after idle timeout
4. **Batch Processing**: Route-details endpoint handles multiple images efficiently
5. **Async Requests**: FastAPI handles concurrent requests without blocking
6. **Smart Image Resize**: Reduce pixels for faster inference
7. **JPEG Quality Trade-offs**: 95% JPEG quality balances speed vs quality

---

## 🛡️ Error Handling

### Graceful Degradation

```python
# JSON parsing failures don't crash service
try:
    data = json.loads(malformed_json)
except:
    # Fallback 1: Try repair
    try:
        repaired = repair_json_string(malformed_json)
        data = json.loads(repaired)
    except:
        # Fallback 2: Use Pydantic defaults
        data = DetectionOutput(objects=[]).model_dump()

# Ollama connection failures
try:
    response = requests.get(ollama_url, timeout=5)
except:
    return {"status": "degraded", "ollama": {"connected": False}}
```

### Error Response Examples

**Invalid Image**:
```json
{"error": "Invalid base64 image data"}
```

**Ollama Connection Failure**:
```json
{
  "error": "Failed to connect to Ollama at http://host.docker.internal:11434",
  "status": "degraded"
}
```

**Inference Timeout**:
```json
{
  "error": "Ollama request timed out after 300s",
  "recommendation": "Try again or increase OLLAMA_TIMEOUT"
}
```

---

## 🚀 Future Improvements

- [ ] **Batch Inference**: Process multiple images in parallel
- [x] **Model Quantization**: 8-bit/4-bit models for faster inference
- [ ] **Caching**: Cache detection results for identical images
- [ ] **Multi-Model Support**: Switch between Qwen, LLaVA, other vision models
- [ ] **WebSocket Support**: Streaming video frame detection
- [ ] **Distributed Inference**: Multiple Ollama servers with load balancing
- [ ] **Metrics & Monitoring**: Prometheus metrics, performance tracking
- [ ] **Request Batching**: Combine multiple detection requests
- [ ] **GPU Memory Optimization**: Dynamic batch sizing based on available VRAM

---

## 👤 Author

[**Mansi Dhanania**](https://github.com/MansiDhanania) | AI/ML Engineer

---

## 🏛️ Acknowledgements

This project was developed as part of M.Sc. thesis research at the **[Shared Reality Lab](https://srl.mcgill.ca/)** - **[Centre for Intelligent Machines](https://www.mcgill.ca/cim/)**, McGill University, under the supervision of **Prof. Jeremy Cooperstock**.

---

**Last Updated**: May 2026  
**Framework Version**: FastAPI 0.104+  
**Python Version**: 3.9+  
**Status**: Production Ready ✅
