"""Simple YOLO (Ultralytics) wrapper for this project.

Provides helpers to load a YOLO model and run inference on images or folders.

Usage examples:
    from yolo_wrapper import load_model, predict_image
    model = load_model('yolo26n.pt')
    results = predict_image(model, 'images/train/example.jpg', conf=0.25, save=True)

Run from CLI:
    python yolo_wrapper.py path/to/image.jpg --weights yolo26n.pt --save
"""
from pathlib import Path
from ultralytics import YOLO
import os


def load_model(weights: str = 'yolo26n.pt', device: str | None = None):
    """Load and return a YOLO model.

    Args:
        weights: path or name of weights file (will download if missing).
        device: device string like 'cpu' or 'cuda:0'. If None uses default.
    Returns:
        Ultralytics YOLO model instance.
    """
    if device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device
    model = YOLO(weights)
    return model


def predict_image(model, image_path, conf: float = 0.25, save: bool = True, imgsz: int = 640):
    """Run inference on a single image.

    Returns the raw Ultralytics results object (list-like).
    """
    image_path = str(Path(image_path))
    results = model.predict(source=image_path, conf=conf, imgsz=imgsz, save=save)
    return results


def predict_folder(model, folder_path, conf: float = 0.25, save: bool = True, imgsz: int = 640):
    """Run inference on all images in a folder. Results are saved if `save=True`.
    """
    folder_path = str(Path(folder_path))
    results = model.predict(source=folder_path, conf=conf, imgsz=imgsz, save=save)
    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='YOLO wrapper: load model and run inference')
    parser.add_argument('source', help='image file or folder to run inference on')
    parser.add_argument('--weights', default='yolo26n.pt', help='weights file or name')
    parser.add_argument('--conf', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--imgsz', type=int, default=640, help='inference image size')
    parser.add_argument('--no-save', dest='save', action='store_false', help="Don't save annotated results")
    parser.set_defaults(save=True)
    args = parser.parse_args()

    model = load_model(args.weights)
    print('Loaded model:', type(model))
    res = model.predict(source=args.source, conf=args.conf, imgsz=args.imgsz, save=args.save)
    print('Inference finished. Results object:', type(res))