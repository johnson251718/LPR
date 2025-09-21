# LPR

## YOLO-based license plate detection

The project now relies on [ONNX Runtime](https://onnxruntime.ai/) to execute
YOLO-based license plate detectors. Replace the previous OpenCV DNN backend by
installing one of the following packages before running inference:

```bash
pip install onnxruntime           # CPU inference
# or
pip install onnxruntime-gpu       # CUDA-enabled inference (where supported)
```

When constructing `YoloLicensePlateDetector` you can optionally provide the list
of execution providers passed to `onnxruntime.InferenceSession`. Falling back to
`ort.get_available_providers()` ensures the detector automatically selects a
suitable runtime on the current system.

The preprocessing pipeline still depends on NumPy for image manipulation. Make
sure it is available in your environment before running detections.
