"""
Vision Pipeline API - Standalone Object Detection Service with Qwen2.5-VL
Supports CPU and GPU inference with automatic hardware detection and lazy model loading.
Features auto-unload after idle timeout to free GPU memory.
"""

import os
import platform
import torch
import base64
import binascii
import uuid
import math
import re
import json
import gc
import time
import threading
import atexit
import requests
from io import BytesIO
from typing import List, Tuple, Optional, Any, Union
from pathlib import Path

import cv2
import numpy as np
import psutil
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from transformers import pipeline
from pydantic import BaseModel, Field, field_validator, model_validator


# ============================================================================
# PYDANTIC MODELS FOR ROBUST JSON PARSING
# ============================================================================

class BoundingBox(BaseModel):
    """Validated bounding box with automatic coordinate fixing."""
    coordinates: List[float] = Field(default_factory=lambda: [0, 0, 0, 0])
    
    @field_validator('coordinates', mode='before')
    @classmethod
    def validate_bbox(cls, v):
        """Convert various bbox formats to [x_min, y_min, x_max, y_max]."""
        if v is None:
            return [0, 0, 0, 0]
        
        # Handle dict format like {"x_min": 0, "y_min": 0, ...}
        if isinstance(v, dict):
            try:
                return [
                    float(v.get('x_min', v.get('xmin', v.get('x1', 0)))),
                    float(v.get('y_min', v.get('ymin', v.get('y1', 0)))),
                    float(v.get('x_max', v.get('xmax', v.get('x2', 0)))),
                    float(v.get('y_max', v.get('ymax', v.get('y2', 0))))
                ]
            except (TypeError, ValueError):
                return [0, 0, 0, 0]
        
        # Handle list/tuple format
        if isinstance(v, (list, tuple)):
            coords = []
            for item in v[:4]:  # Take only first 4 elements
                try:
                    if isinstance(item, str):
                        # Handle string numbers or placeholders
                        item = item.strip()
                        if item in ['x_min', 'y_min', 'x_max', 'y_max', '']:
                            coords.append(0.0)
                        else:
                            coords.append(float(item))
                    elif item is None:
                        coords.append(0.0)
                    else:
                        coords.append(float(item))
                except (TypeError, ValueError):
                    coords.append(0.0)
            
            # Pad with zeros if needed
            while len(coords) < 4:
                coords.append(0.0)
            
            return coords[:4]
        
        return [0, 0, 0, 0]
    
    def to_list(self) -> List[int]:
        """Return bbox as list of integers."""
        return [int(c) for c in self.coordinates]


class DetectedObject(BaseModel):
    """A single detected object with automatic field normalization."""
    label: str = Field(default="unknown")
    bbox: List[float] = Field(default_factory=lambda: [0, 0, 0, 0])
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    
    @field_validator('label', mode='before')
    @classmethod
    def validate_label(cls, v):
        """Ensure label is a non-empty string."""
        if v is None:
            return "unknown"
        return str(v).strip() or "unknown"
    
    @field_validator('bbox', mode='before')
    @classmethod
    def validate_bbox(cls, v):
        """Normalize bbox from various formats."""
        bbox_model = BoundingBox(coordinates=v)
        return bbox_model.coordinates
    
    @field_validator('confidence', mode='before')
    @classmethod
    def validate_confidence(cls, v):
        """Ensure confidence is a valid float between 0 and 1."""
        if v is None:
            return 0.0
        try:
            conf = float(v)
            return max(0.0, min(1.0, conf))
        except (TypeError, ValueError):
            return 0.0
    
    @model_validator(mode='before')
    @classmethod
    def normalize_field_names(cls, data):
        """Handle alternative field names from Qwen output."""
        if not isinstance(data, dict):
            return data
        
        normalized = {}
        
        # Handle label variations
        for key in ['label', 'name', 'class', 'object', 'category', 'type']:
            if key in data:
                normalized['label'] = data[key]
                break
        if 'label' not in normalized:
            normalized['label'] = 'unknown'
        
        # Handle bbox variations
        for key in ['bbox', 'bbox_2d', 'box', 'bounding_box', 'coordinates', 'rect']:
            if key in data:
                normalized['bbox'] = data[key]
                break
        if 'bbox' not in normalized:
            normalized['bbox'] = [0, 0, 0, 0]
        
        # Handle confidence variations
        for key in ['confidence', 'score', 'conf', 'probability', 'prob']:
            if key in data:
                normalized['confidence'] = data[key]
                break
        if 'confidence' not in normalized:
            normalized['confidence'] = 0.0
        
        return normalized


class DetectionOutput(BaseModel):
    """Complete detection output with automatic repair."""
    objects: List[DetectedObject] = Field(default_factory=list)
    
    @field_validator('objects', mode='before')
    @classmethod
    def validate_objects(cls, v):
        """Handle various object list formats."""
        if v is None:
            return []
        
        if isinstance(v, dict):
            # Single object passed as dict
            return [v]
        
        if isinstance(v, list):
            valid_objects = []
            for item in v:
                if isinstance(item, dict):
                    valid_objects.append(item)
                elif isinstance(item, DetectedObject):
                    valid_objects.append(item)
            return valid_objects
        
        return []


class RouteDetail(BaseModel):
    """Validated route detail output from Qwen."""
    location: str = Field(default="unknown location")
    aisle: str = Field(default="unknown aisle")
    visible_products: List[str] = Field(default_factory=list)
    description: str = Field(default="")
    
    @field_validator('location', 'aisle', 'description', mode='before')
    @classmethod
    def validate_string_field(cls, v):
        """Ensure string fields are valid."""
        if v is None:
            return ""
        return str(v).strip()
    
    @field_validator('visible_products', mode='before')
    @classmethod
    def validate_products(cls, v):
        """Ensure visible_products is a list of strings."""
        if v is None:
            return []
        if isinstance(v, str):
            # Handle comma-separated string
            return [p.strip() for p in v.split(',') if p.strip()]
        if isinstance(v, list):
            return [str(p).strip() for p in v if p]
        return []


def repair_json_string(json_str: str) -> str:
    """
    Attempt to repair malformed JSON string before parsing.
    Handles common Qwen output issues.
    """
    if not json_str:
        return '{}'
    
    original = json_str
    
    # Remove any leading/trailing whitespace
    json_str = json_str.strip()
    
    # Remove markdown code block markers
    json_str = re.sub(r'^```(?:json)?\s*', '', json_str)
    json_str = re.sub(r'\s*```$', '', json_str)
    
    # Replace single quotes with double quotes (but not inside strings)
    # This is a simplified approach - handles most cases
    json_str = re.sub(r"(?<![\\])\'", '"', json_str)
    
    # Fix double double-quotes
    json_str = json_str.replace('""', '"')
    
    # Remove trailing commas before } or ]
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    
    # Fix missing commas between objects in arrays
    json_str = re.sub(r'}\s*{', '},{', json_str)
    
    # Replace placeholder strings with 0
    for placeholder in ['x_min', 'y_min', 'x_max', 'y_max', 'xmin', 'ymin', 'xmax', 'ymax']:
        json_str = re.sub(f'"{placeholder}"', '0', json_str)
    
    # Fix unquoted keys (simple approach)
    json_str = re.sub(r'(\{|\,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
    
    # Fix boolean values
    json_str = re.sub(r'\bTrue\b', 'true', json_str)
    json_str = re.sub(r'\bFalse\b', 'false', json_str)
    json_str = re.sub(r'\bNone\b', 'null', json_str)
    
    # Try to find the JSON object/array boundaries
    # Find first { or [
    start_brace = json_str.find('{')
    start_bracket = json_str.find('[')
    
    if start_brace == -1 and start_bracket == -1:
        return '{}'
    
    if start_brace == -1:
        start = start_bracket
        end_char = ']'
    elif start_bracket == -1:
        start = start_brace
        end_char = '}'
    else:
        start = min(start_brace, start_bracket)
        end_char = '}' if start == start_brace else ']'
    
    # Find matching end
    end = json_str.rfind(end_char)
    if end <= start:
        return '{}'
    
    json_str = json_str[start:end + 1]
    
    # Balance braces/brackets if needed
    open_braces = json_str.count('{')
    close_braces = json_str.count('}')
    open_brackets = json_str.count('[')
    close_brackets = json_str.count(']')
    
    # Add missing closing braces/brackets
    json_str += '}' * (open_braces - close_braces)
    json_str += ']' * (open_brackets - close_brackets)
    
    return json_str


def parse_detection_output_pydantic(output_text: str) -> DetectionOutput:
    """
    Robust JSON parser using Pydantic with automatic repair.
    Handles malformed JSON from Qwen with multiple fallback strategies.
    """
    # Extract assistant response
    assistant_part = output_text.split("assistant", 1)[-1].strip()
    
    # Try to find JSON blocks
    json_matches = re.findall(r"```json\s*(.*?)\s*```", assistant_part, re.DOTALL)
    
    if not json_matches:
        # Try to find raw JSON without code blocks
        json_matches = re.findall(r'(\{[^{}]*"objects"[^{}]*\[.*?\][^{}]*\})', assistant_part, re.DOTALL)
    
    if not json_matches:
        # Try to find any JSON-like structure
        json_matches = re.findall(r'(\{.*\})', assistant_part, re.DOTALL)
    
    if not json_matches:
        print("No JSON found in output, returning empty result")
        return DetectionOutput(objects=[])
    
    # Try each match with repair strategies
    last_error = None
    for json_str in json_matches:
        # Strategy 1: Try direct parse
        try:
            data = json.loads(json_str)
            return DetectionOutput(**data)
        except (json.JSONDecodeError, Exception) as e:
            last_error = e
        
        # Strategy 2: Try with repair
        try:
            repaired = repair_json_string(json_str)
            data = json.loads(repaired)
            return DetectionOutput(**data)
        except (json.JSONDecodeError, Exception) as e:
            last_error = e
        
        # Strategy 3: Try to extract objects array directly
        try:
            objects_match = re.search(r'"objects"\s*:\s*\[(.*?)\]', json_str, re.DOTALL)
            if objects_match:
                objects_str = '[' + objects_match.group(1) + ']'
                repaired_objects = repair_json_string(objects_str)
                objects_list = json.loads(repaired_objects)
                return DetectionOutput(objects=objects_list)
        except (json.JSONDecodeError, Exception) as e:
            last_error = e
    
    print(f"All JSON parsing strategies failed: {last_error}")
    return DetectionOutput(objects=[])


def parse_route_detail_pydantic(output_text: str) -> RouteDetail:
    """
    Robust JSON parser for route details using Pydantic.
    """
    # Extract assistant response
    assistant_part = output_text.split("assistant", 1)[-1].strip()
    
    # Try to find JSON
    json_matches = re.findall(r"```json\s*(.*?)\s*```", assistant_part, re.DOTALL)
    
    if not json_matches:
        json_matches = re.findall(r'(\{.*\})', assistant_part, re.DOTALL)
    
    if not json_matches:
        print("No JSON found in route detail output")
        return RouteDetail()
    
    for json_str in json_matches:
        # Try direct parse
        try:
            data = json.loads(json_str)
            return RouteDetail(**data)
        except (json.JSONDecodeError, Exception):
            pass
        
        # Try with repair
        try:
            repaired = repair_json_string(json_str)
            data = json.loads(repaired)
            return RouteDetail(**data)
        except (json.JSONDecodeError, Exception) as e:
            print(f"Route detail parsing failed: {e}")
    
    return RouteDetail()


# ============================================================================
# LEGACY PARSER (kept for reference, but deprecated)
# ============================================================================

from fastapi import Request

# Ollama configuration
OLLAMA_TIMEOUT = int(os.getenv('OLLAMA_TIMEOUT', '300'))  # 5 minutes for inference
OLLAMA_HEALTH_TIMEOUT = int(os.getenv('OLLAMA_HEALTH_TIMEOUT', '5'))  # 5 seconds for health checks
OLLAMA_STARTUP_RETRIES = int(os.getenv('OLLAMA_STARTUP_RETRIES', '3'))
OLLAMA_RETRY_DELAY = int(os.getenv('OLLAMA_RETRY_DELAY', '2'))

class LazyModelLoader:
    """Ollama model loader for vision pipeline.
    Manages Ollama connection and depth estimation model.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self):
        self.depth_pipe = None
        self.model_name = os.getenv('OLLAMA_MODEL', 'qwen2.5vl:3b')
        self.ollama_url = os.getenv('OLLAMA_URL', 'http://host.docker.internal:11434')
        self.hardware_info = self._detect_hardware()
        self._verify_ollama_connection()
        
        # Qwen parameters (for image preprocessing)
        self.factor = 28
        self.min_pixels = 56 * 56
        self.max_pixels = 14 * 14 * 4 * 1280
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance of the model loader."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def _detect_hardware(self):
        """Detect available hardware for inference."""
        hardware_info = {
            "device": "remote_ollama",
            "inference_engine": "ollama",
            "ollama_url": self.ollama_url
        }
        return hardware_info
    
    def _verify_ollama_connection(self):
        """Verify that Ollama is running and model is available with retry logic."""
        for attempt in range(OLLAMA_STARTUP_RETRIES):
            try:
                response = requests.get(
                    f"{self.ollama_url}/api/tags", 
                    timeout=OLLAMA_HEALTH_TIMEOUT
                )
                response.raise_for_status()
                models = response.json().get("models", [])
                model_names = [m.get("name") for m in models]
                
                if self.model_name in model_names:
                    print(f"✓ Ollama connected: {self.ollama_url}")
                    print(f"✓ Model available: {self.model_name}")
                    return True
                else:
                    print(f"⚠ Model '{self.model_name}' not found in Ollama")
                    print(f"  Available models: {model_names}")
                    if attempt < OLLAMA_STARTUP_RETRIES - 1:
                        print(f"  Retrying in {OLLAMA_RETRY_DELAY}s...")
                        time.sleep(OLLAMA_RETRY_DELAY)
                        continue
            except Exception as e:
                print(f"⚠ Ollama connection attempt {attempt + 1}/{OLLAMA_STARTUP_RETRIES} failed: {str(e)}")
                if attempt < OLLAMA_STARTUP_RETRIES - 1:
                    print(f"  Retrying in {OLLAMA_RETRY_DELAY}s...")
                    time.sleep(OLLAMA_RETRY_DELAY)
                else:
                    print(f"⚠ Could not connect to Ollama after {OLLAMA_STARTUP_RETRIES} attempts")
                    print(f"  Attempting to connect to: {self.ollama_url}")
                    print(f"  Service will start but may fail on inference requests")
        
        return False
    
    def get_model(self):
        """Get depth pipe (Ollama doesn't need loading). For API compatibility."""
        with self._lock:
            if self.depth_pipe is None:
                self._load_depth_model()
        return None, None, self.depth_pipe  # Return None for qwen_model, processor (using Ollama)
    
    def _load_depth_model(self):
        """Depth model disabled for faster inference."""
        print("⊘ Depth estimation disabled (set ENABLE_DEPTH=true to enable)")
        self.depth_pipe = None
        return
        
        # Depth model loading code (disabled)
        # depth_model_name = os.getenv('DEPTH_MODEL', 'depth-anything/Depth-Anything-V2-Small-hf')
        # try:
        #     print(f"Loading depth model: {depth_model_name}...")
        #     self.depth_pipe = pipeline(
        #         task="depth-estimation",
        #         model=depth_model_name,
        #         device=-1  # Force CPU since Ollama uses GPU for vision
        #     )
        #     print("✓ Depth model loaded on CPU")
        # except Exception as e:
        #     print(f"⚠ Depth model failed to load: {str(e)}")
        #     self.depth_pipe = None
    
    def is_loaded(self):
        """Check if depth model is loaded (Ollama is remote)."""
        return self.depth_pipe is not None
    
    def unload_model(self):
        """Unload depth model to free resources."""
        with self._lock:
            unloaded = []
            
            if self.depth_pipe is not None:
                del self.depth_pipe
                self.depth_pipe = None
                unloaded.append("depth_pipe")
            
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            print(f"Models unloaded: {', '.join(unloaded) if unloaded else 'none'}")
            return {"status": "unloaded", "unloaded_models": unloaded}
    
    def get_memory_info(self) -> dict:
        """Get current memory usage information."""
        memory_info = {
            "system": {
                "total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
                "used_gb": round(psutil.virtual_memory().used / (1024**3), 2),
                "percent_used": psutil.virtual_memory().percent
            },
            "note": "Qwen model runs on remote Ollama server"
        }
        
        if torch.cuda.is_available():
            gpu_memory = {
                "allocated_gb": round(torch.cuda.memory_allocated() / (1024**3), 3),
                "reserved_gb": round(torch.cuda.memory_reserved() / (1024**3), 3),
                "max_allocated_gb": round(torch.cuda.max_memory_allocated() / (1024**3), 3)
            }
            
            try:
                total_memory = torch.cuda.get_device_properties(0).total_memory
                gpu_memory["total_gb"] = round(total_memory / (1024**3), 2)
                gpu_memory["free_gb"] = round((total_memory - torch.cuda.memory_allocated()) / (1024**3), 3)
            except Exception:
                pass
            
            memory_info["gpu"] = gpu_memory
        
        return memory_info


# Initialize lazy model loader singleton
MODEL_LOADER = LazyModelLoader.get_instance()

# Register cleanup on exit
atexit.register(lambda: None)  # No auto-unload thread for Ollama

# Initialize FastAPI
app = FastAPI(
    title="Vision Pipeline API",
    description="Standalone object detection service with Qwen2.5-VL and depth estimation",
    version="2.1.0"
)


def call_ollama(image_pil: Image.Image, prompt: str) -> str:
    """
    Call Ollama API for vision-language inference.
    Returns response text from the model.
    """
    try:
        # Encode image to base64
        buffer = BytesIO()
        image_pil.save(buffer, format="JPEG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Make request to Ollama (vision models use /api/generate)
        response = requests.post(
            f"{MODEL_LOADER.ollama_url}/api/generate",
            json={
                "model": MODEL_LOADER.model_name,
                "prompt": prompt,
                "images": [image_base64],
                "stream": False
            },
            timeout=OLLAMA_TIMEOUT
        )
        response.raise_for_status()
        
        # Extract response text
        result = response.json()
        return result.get("response", "")
    
    except requests.exceptions.ConnectionError:
        raise RuntimeError(f"Failed to connect to Ollama at {MODEL_LOADER.ollama_url}")
    except requests.exceptions.Timeout:
        raise RuntimeError(f"Ollama request timed out after {OLLAMA_TIMEOUT}s")
    except Exception as e:
        raise RuntimeError(f"Ollama inference failed: {str(e)}")


def smart_resize(height, width, factor=28, min_pixels=None, max_pixels=None):
    """Qwen's smart resize function."""
    if max_pixels is None:
        max_pixels = 14 * 14 * 4 * 1280
    if min_pixels is None:
        min_pixels = 56 * 56
    
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    
    return h_bar, w_bar


def parse_detection_output(output_text):
    """
    Parse JSON detection output from Qwen using Pydantic for robust handling.
    This is the main entry point - uses Pydantic-based parser with automatic repair.
    """
    result = parse_detection_output_pydantic(output_text)
    
    # Convert to dict format for backward compatibility
    return {
        "objects": [
            {
                "label": obj.label,
                "bbox": obj.bbox,
                "confidence": obj.confidence
            }
            for obj in result.objects
        ]
    }


def get_depth_map(depth_pipe, image_pil):
    """Generate depth map using depth estimation model."""
    if depth_pipe is None:
        return None
    
    try:
        output = depth_pipe(image_pil)
        depth_array = np.array(output["depth"], dtype=np.float32)
        
        if depth_array.shape != (image_pil.height, image_pil.width):
            depth_pil = output["depth"].resize(
                (image_pil.width, image_pil.height),
                Image.Resampling.BICUBIC
            )
            depth_array = np.asarray(depth_pil, dtype=np.float32)
        
        return depth_array
    except Exception as e:
        print(f"Depth estimation failed: {str(e)}")
        return None


def get_bbox_depth(depth_map, bbox, method='median'):
    """Calculate depth for bounding box."""
    if depth_map is None:
        return None
    
    x1, y1, x2, y2 = map(int, bbox)
    h, w = depth_map.shape
    
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    cropped_depth = depth_map[y1:y2, x1:x2]
    
    if cropped_depth.size == 0:
        return None
    
    if method == 'mean':
        return float(np.mean(cropped_depth))
    elif method == 'median':
        return float(np.median(cropped_depth))
    elif method == 'min':
        return float(np.min(cropped_depth))
    elif method == 'max':
        return float(np.max(cropped_depth))
    else:
        return float(np.median(cropped_depth))

def generate_guidance(
    user_pos: Tuple[float, float, float],
    object_pos: Tuple[float, float, float],
    eye_level: float = 1.5
) -> str:
    """Generate spatial guidance for object location relative to user position."""
    x1, y1, z1 = user_pos
    x2, y2, z2 = object_pos
    
    # Calculate distance and direction
    dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
    distance = math.sqrt(dx**2 + dy**2 + dz**2)
    
    # Determine horizontal direction
    if abs(dx) < 0.25:
        direction = "straight ahead"
    elif dx < 0:
        direction = "to your left"
    else:
        direction = "to your right"
    
    # Determine vertical position
    vertical_offset = y2 - eye_level
    if abs(vertical_offset) < 0.2:
        level = "at eye level"
    elif vertical_offset > 0:
        level = "above eye level"
    else:
        level = "below eye level"
    
    return f"The object is approximately {round(distance)} units away, {direction}, {level}."

def qwen_detection(image_pil, classes, output_path="output.jpg", score_threshold=0.2, custom_prompt: Optional[str] = None):
    """
    Perform object detection with Qwen2.5-VL (via Ollama) and depth estimation.
    Returns: output_path, detection_results, img_base64
    
    Args:
        image_pil: PIL Image to process
        classes: List of object classes to detect
        output_path: Path to save annotated image
        score_threshold: Minimum confidence threshold
        custom_prompt: Optional custom prompt (uses default if None)
    """
    # Get depth model
    _, _, depth_pipe = MODEL_LOADER.get_model()
    
    # Prepare for detection
    original_width, original_height = image_pil.size
    
    input_height, input_width = smart_resize(
        original_height, original_width,
        factor=MODEL_LOADER.factor,
        min_pixels=MODEL_LOADER.min_pixels,
        max_pixels=MODEL_LOADER.max_pixels
    )
    
    scale_x = original_width / input_width
    scale_y = original_height / input_height
    
    # Resize image for inference
    image_resized = image_pil.resize((input_width, input_height), Image.Resampling.BICUBIC)
    
    # Construct prompt for target object
    target_object = classes[0] if classes else "object"
    
    # Use custom prompt if provided, otherwise use default
    if custom_prompt:
        prompt = custom_prompt
    else:
        prompt = f"""You are a vision-language model.
Locate all visible instances of the object '{target_object}' in this image.
For each instance, provide a confidence score between 0 and 1.
Return only a JSON block in this format:
```json
{{"objects": [{{"label": "{target_object}", "bbox": [x_min, y_min, x_max, y_max], "confidence": 0.95}}]}}
```
The bbox coordinates should be absolute pixel values, not normalized."""
    
    # Call Ollama for inference
    output_text = call_ollama(image_resized, prompt)
    
    # Parse detections (format the output for parsing compatibility)
    formatted_output = f"assistant\n{output_text}"
    data = parse_detection_output(formatted_output)
    
    # Scale bboxes and filter by confidence
    detection_results = []
    for obj in data.get("objects", []):
        if "bbox" in obj:
            x_min, y_min, x_max, y_max = obj["bbox"]
            obj["bbox"] = [
                int(x_min * scale_x),
                int(y_min * scale_y),
                int(x_max * scale_x),
                int(y_max * scale_y)
            ]
        
        confidence = obj.get("confidence", 0.0)
        if confidence >= score_threshold:
            detection_results.append({
                "label": obj.get("label", target_object),
                "score": round(confidence, 2),
                "box": obj["bbox"]
            })
    
    # Sort by confidence and take top 3
    detection_results.sort(key=lambda x: x["score"], reverse=True)
    top_detections = detection_results[:3]
    
    # Get depth map
    depth_map = get_depth_map(depth_pipe, image_pil)
    
    # Add depth information to detections
    if depth_map is not None:
        for detection in top_detections:
            depth_value = get_bbox_depth(depth_map, detection["box"], method='median')
            if depth_value is not None:
                detection["depth"] = round(depth_value, 2)
    
    # Convert to OpenCV for annotation
    image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    
    # Draw bounding boxes with color coding
    colors = [(0, 255, 0), (0, 165, 255), (0, 100, 255)]  # Green, Orange, Blue
    
    for i, detection in enumerate(top_detections):
        box_int = detection["box"]
        xmin, ymin, xmax, ymax = box_int
        color = colors[i] if i < len(colors) else (128, 128, 128)
        
        # Draw rectangle
        cv2.rectangle(image_cv, (xmin, ymin), (xmax, ymax), color, 2)
        
        # Prepare label
        label_text = f"{detection['label']}: {detection['score']:.2f}"
        if "depth" in detection:
            label_text += f" [D:{detection['depth']:.1f}]"
        
        cv2.putText(
            image_cv, label_text, (xmin, max(0, ymin - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
    
    # Draw arrow to highest confidence detection
    if top_detections:
        best_detection = top_detections[0]
        box = best_detection["box"]
        xmin, ymin, xmax, ymax = box
        obj_center = (int((xmin + xmax) / 2), int((ymin + ymax) / 2))
        img_h, img_w = image_cv.shape[:2]
        img_center = (img_w // 2, img_h // 2)
        cv2.arrowedLine(image_cv, img_center, obj_center, (0, 0, 255), 4, tipLength=0.2)
    
    # Save and encode
    cv2.imwrite(output_path, image_cv)
    _, buffer = cv2.imencode('.jpg', image_cv)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return output_path, top_detections, img_base64

def qwen_routing_details(images: List[Image.Image]) -> dict:
    """
    Run Qwen vision-language model (via Ollama) on multiple images to generate descriptive JSON outputs.
    Return combined dict keyed by frame index.
    Uses Pydantic-based parsing for robust JSON handling.
    """
    combined_output = {}

    for idx, image in enumerate(images):
        # Resize & process image as per existing detection pipeline
        input_height, input_width = smart_resize(
            image.height, image.width,
            factor=MODEL_LOADER.factor,
            min_pixels=MODEL_LOADER.min_pixels,
            max_pixels=MODEL_LOADER.max_pixels
        )
        img_resized = image.resize((input_width, input_height), Image.Resampling.BICUBIC)

        # Construct prompt for scene understanding describing aisle, location, visible products
        prompt = (
                "You are a vision-language model. "
                "Please analyze this store aisle image and provide a JSON object describing it. "
                "Your JSON should have keys: "
                "\"location\" (short string describing where the user is), "
                "\"aisle\" (the aisle name or type), "
                "\"visible_products\" (a list of product types that are visible), "
                "and \"description\" (a brief natural language description of what the user might perceive). "
                "Output ONLY the JSON object with no extra text or explanation."
            )

        # Call Ollama for inference
        output_text = call_ollama(img_resized, prompt)

        # Parse the JSON from output_text using Pydantic-based parser
        try:
            formatted_output = f"assistant\n{output_text}"
            route_detail = parse_route_detail_pydantic(formatted_output)
            json_data = route_detail.model_dump()
        except Exception as e:
            print(f"Failed to parse route detail for image {idx+1}: {e}")
            json_data = RouteDetail().model_dump()

        combined_output[f"image_{idx+1}"] = json_data

    return combined_output

class DetectRequest(BaseModel):
    image: str
    object: str
    score_threshold: float = 0.15
    user_pos: Optional[List[float]] = None
    eye_level: float = 1.5
    prompt: Optional[str] = None  # Custom prompt for Qwen (uses default if None)


@app.post("/detect")
async def detect(request: DetectRequest):
    """
    Detect objects in an image and provide spatial guidance.
    Returns annotated image with bounding boxes and directional guidance.
    """
    try:
        # Trigger depth model loading
        _, _, depth_pipe = MODEL_LOADER.get_model()
        
        # Decode base64 image
        image_data = request.image
        if image_data.startswith("data:image"):
            image_data = image_data.split(",", 1)[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        
        # Ensure minimum image quality
        if image.width < 224 or image.height < 224:
            image = image.resize(
                (max(224, image.width), max(224, image.height)),
                Image.Resampling.LANCZOS
            )
        
        # Perform object detection (with optional custom prompt)
        classes = [request.object]
        output_path = f"output_{uuid.uuid4().hex[:8]}.jpg"
        _, detections, img_base64 = qwen_detection(
            image, classes, output_path, 
            score_threshold=request.score_threshold,
            custom_prompt=request.prompt
        )
        
        # Find the best detection
        best_detection = None
        for det in detections:
            if det["label"].lower() == request.object.lower():
                if best_detection is None or det["score"] > best_detection["score"]:
                    best_detection = det
        
        if not best_detection:
            return JSONResponse(
                {
                    "error": f"Object '{request.object}' not found in image",
                    "detections": detections,
                    "annotated_image": f"data:image/jpeg;base64,{img_base64}",
                    "debug_info": f"Total detections: {len(detections)}, threshold: {request.score_threshold}"
                },
                status_code=404
            )
        
        # Generate spatial guidance
        bbox = best_detection["box"]
        xmin, ymin, xmax, ymax = bbox
        object_center = ((xmin + xmax) / 2, (ymin + ymax) / 2, 0.0)
        
        if request.user_pos and len(request.user_pos) >= 2:
            user_pos = tuple(request.user_pos[:3]) if len(request.user_pos) >= 3 else tuple(request.user_pos + [0.0])
        else:
            width, height = image.size
            user_pos = (width / 2, height / 2, 0.0)
        
        guidance = generate_guidance(user_pos, object_center, request.eye_level)
        
        # Include depth in response if available
        response = {
            "success": True,
            "object": request.object,
            "confidence": best_detection["score"],
            "bbox": bbox,
            "guidance": guidance,
            "annotated_image": f"data:image/jpeg;base64,{img_base64}",
            "all_detections": detections
        }
        
        if "depth" in best_detection:
            response["depth"] = best_detection["depth"]
        
        return response
        
    except binascii.Error:
        return JSONResponse({"error": "Invalid base64 image data"}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": f"Detection failed: {str(e)}"}, status_code=500)

@app.post("/route-details")
async def route_details(request: Request):
    """
    Receive list of images (base64 strings) in JSON and return combined JSON details for routing.
    """
    data = await request.json()
    image_strs = data.get("images", [])

    if not image_strs:
        return JSONResponse({"error": "No images provided"}, status_code=400)

    # Decode base64 images, convert to PIL
    images = []
    for img_str in image_strs:
        if img_str.startswith("data:image"):
            img_str = img_str.split(",", 1)[1]
        img_bytes = base64.b64decode(img_str)
        images.append(Image.open(BytesIO(img_bytes)).convert("RGB"))

    # Process with qwen routing details function
    try:
        combined_details = qwen_routing_details(images)
        return {"success": True, "route_details": combined_details}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/qwen-preprocessing")
async def qwen_preprocessing(request: Request):
    """
    Resize image using exact Qwen smart_resize logic for Fireworks API compatibility.
    Returns base64 resized image + scale factors for bbox denormalization.
    
    Input:  {"image": "data:image/jpeg;base64,..."}
    Output: {"resized_image": "data:image/jpeg;base64,...", "scale_x": 1.23, "scale_y": 1.45}
    """
    try:
        data = await request.json()
        image_data = data["image"]
        
        # Decode base64 image (handle data URL prefix)
        if image_data.startswith("data:image"):
            image_data = image_data.split(",", 1)[1]
        
        image_bytes = base64.b64decode(image_data)
        image_pil = Image.open(BytesIO(image_bytes)).convert("RGB")
        
        # Get original dimensions
        original_width, original_height = image_pil.size
        
        # Use EXACT same resize logic as qwen_detection()
        input_height, input_width = smart_resize(
            original_height, original_width,
            factor=MODEL_LOADER.factor,  # 28
            min_pixels=MODEL_LOADER.min_pixels,  # 56*56
            max_pixels=MODEL_LOADER.max_pixels   # 14*14*4*1280
        )
        
        # Resize with BICUBIC (matches pipeline)
        resized_image = image_pil.resize(
            (input_width, input_height), 
            Image.Resampling.BICUBIC
        )
        
        # Calculate scale factors for bbox denormalization
        scale_x = original_width / input_width
        scale_y = original_height / input_height
        
        # Encode resized image as base64
        buffer = BytesIO()
        resized_image.save(buffer, format="JPEG", quality=95)
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return {
            "success": True,
            "resized_image": f"data:image/jpeg;base64,{img_base64}",
            "original_size": [original_width, original_height],
            "resized_size": [input_width, input_height],
            "scale_x": round(scale_x, 4),
            "scale_y": round(scale_y, 4),
            "denormalize_instructions": f"To convert Fireworks bbox [x1,y1,x2,y2] to original: [x1*{scale_x}, y1*{scale_y}, x2*{scale_x}, y2*{scale_y}]"
        }
        
    except KeyError:
        return JSONResponse({"error": "Missing 'image' field"}, status_code=400)
    except binascii.Error:
        return JSONResponse({"error": "Invalid base64 image data"}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": f"Preprocessing failed: {str(e)}"}, status_code=500)


@app.get("/hardware")
async def hardware_info():
    """Get detailed hardware information."""
    return {
        "hardware_info": MODEL_LOADER.hardware_info,
        "pytorch_version": torch.__version__,
        "inference_engine": "Ollama (remote)",
        "ollama_url": MODEL_LOADER.ollama_url,
        "model_name": MODEL_LOADER.model_name,
        "depth_model_available": MODEL_LOADER.depth_pipe is not None,
        "depth_model_loaded": MODEL_LOADER.is_loaded(),
        "system_info": {
            "platform": platform.system(),
            "architecture": platform.machine(),
            "python_version": platform.python_version()
        }
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint with memory usage and model status.
    Returns detailed information about system health, memory usage, and depth model state.
    """
    memory_info = MODEL_LOADER.get_memory_info()
    
    # Check Ollama connection
    ollama_connected = False
    try:
        response = requests.get(
            f"{MODEL_LOADER.ollama_url}/api/tags", 
            timeout=OLLAMA_HEALTH_TIMEOUT
        )
        ollama_connected = response.status_code == 200
    except:
        ollama_connected = False
    
    health_data = {
        "status": "healthy" if ollama_connected else "degraded",
        "service": "vision-pipeline",
        "version": "2.2.0",
        "inference_engine": "Ollama (remote)",
        "ollama": {
            "connected": ollama_connected,
            "url": MODEL_LOADER.ollama_url,
            "model": MODEL_LOADER.model_name
        },
        "depth_model": {
            "loaded": MODEL_LOADER.is_loaded()
        },
        "memory": memory_info
    }
    
    return health_data


@app.post("/unload")
async def unload_model():
    """
    Manually unload models from GPU/CPU memory to free resources.
    Use this endpoint when you want to release memory without stopping the service.
    The model will be automatically reloaded on the next /detect or /route-details request.
    """
    result = MODEL_LOADER.unload_model()
    
    # Get updated memory info after unload
    memory_after = MODEL_LOADER.get_memory_info()
    
    return {
        **result,
        "memory_after_unload": memory_after,
        "note": "Model will be reloaded automatically on next inference request"
    }


@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "Vision Pipeline API",
        "version": "2.2.0",
        "description": "Object detection with Qwen2.5-VL (via Ollama) and depth estimation",
        "inference_engine": "Ollama (remote)",
        "ollama_url": MODEL_LOADER.ollama_url,
        "model_name": MODEL_LOADER.model_name,
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
