"""
inference.py — прогон модели на произвольном видеофайле или папке с кадрами.

Использование:
  # Видеофайл → видеофайл
  python inference.py --config configs/retinex_cfc_sdsd.yaml \
                      --checkpoint experiments/checkpoints/cfc_sdsd/best.pth \
                      --input path/to/dark_video.mp4 \
                      --output path/to/enhanced.mp4

  # Папка с кадрами (.png/.jpg) → папка с enhanced кадрами
  python inference.py --config configs/retinex_cfc_sdsd.yaml \
                      --checkpoint best.pth \
                      --input path/to/frames/ \
                      --output path/to/output/

  # Папка с .npy кадрами (SDSD формат)
  python inference.py --config configs/retinex_cfc_sdsd.yaml \
                      --checkpoint best.pth \
                      --input path/to/LQ/pair1/ \
                      --output path/to/output/ \
                      --npy

Зависимости:
  обязательные: torch, torchvision, numpy, PIL
  опциональные: cv2 (для чтения/записи видео mp4)
"""

import argparse
import sys
import time
import yaml
import torch
import numpy as np
from pathlib import Path
from collections import deque
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision

sys.path.insert(0, str(Path(__file__).parent))

from models import RetinexLNNPipeline


# ──────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────

def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


# ──────────────────────────────────────────────
# Загрузка модели
# ──────────────────────────────────────────────

def build_model(cfg: dict, checkpoint: str, device: torch.device) -> RetinexLNNPipeline:
    mc = cfg['model']
    model = RetinexLNNPipeline(
        temporal_type=cfg['temporal_type'],
        n_feat=mc.get('n_feat', 32),
        stage=mc.get('stage', 1),
        num_blocks=mc.get('num_blocks', [1, 1, 1]),
        window_size=mc.get('window_size', 5),
        hidden_dim=mc.get('hidden_dim', 64),
        n_neurons=mc.get('n_neurons', 32),
        fps=mc.get('fps', 30.0),
    ).to(device)

    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    state = ckpt.get('model', ckpt)
    model.load_state_dict(state)
    model.eval()
    print(f"  Checkpoint: {checkpoint}")
    params = model.get_num_params()
    print(f"  Params: total={params['total']:,} | temporal={params['temporal']:,}")
    return model


# ──────────────────────────────────────────────
# Загрузка кадров
# ──────────────────────────────────────────────

def load_frame_png(path) -> torch.Tensor:
    """PNG/JPG → [3, H, W] float32 [0,1]"""
    return TF.to_tensor(Image.open(str(path)).convert('RGB'))


def load_frame_npy(path) -> torch.Tensor:
    """NPY (H,W,3) float32 → [3, H, W] float32 [0,1]"""
    arr = np.load(str(path)).astype(np.float32)
    if arr.max() > 1.5:
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    return torch.from_numpy(arr).permute(2, 0, 1)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """[3, H, W] float32 [0,1] → PIL Image"""
    t = t.clamp(0.0, 1.0).cpu()
    return TF.to_pil_image(t)


# ──────────────────────────────────────────────
# Sliding window inference
# ──────────────────────────────────────────────

class SlidingWindowInference:
    """
    Онлайн инференс со скользящим окном.
    Буферизует window_size кадров, выдаёт enhanced текущий кадр.
    Первые window_size-1 кадров дублируются для заполнения буфера (cold start).
    """

    def __init__(self, model: RetinexLNNPipeline, window_size: int,
                 fps: float, device: torch.device):
        self.model = model
        self.window_size = window_size
        self.dt = 1.0 / fps
        self.device = device
        self.buffer: deque = deque(maxlen=window_size)

    def reset(self):
        self.buffer.clear()

    @torch.no_grad()
    def process_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """
        frame: [3, H, W] float32 [0,1]
        Returns: enhanced [3, H, W] float32 [0,1]
        """
        self.buffer.append(frame)

        # Cold start: дублируем первый кадр пока буфер не заполнен
        while len(self.buffer) < self.window_size:
            self.buffer.appendleft(self.buffer[0])

        # Формируем батч [1, T, 3, H, W]
        window = torch.stack(list(self.buffer), dim=0).unsqueeze(0).to(self.device)
        timespans = torch.full((1, self.window_size), self.dt, device=self.device)

        enhanced = self.model(window, timespans)
        return enhanced[0].clamp(0.0, 1.0).cpu()


# ──────────────────────────────────────────────
# Источники входных данных
# ──────────────────────────────────────────────

def iter_frames_from_folder(folder: Path, npy: bool):
    """Генератор кадров из папки с изображениями или .npy файлами."""
    if npy:
        paths = sorted(folder.glob('*.npy'))
        loader = load_frame_npy
    else:
        exts = {'.png', '.jpg', '.jpeg', '.bmp'}
        paths = sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])
        loader = load_frame_png

    if not paths:
        raise FileNotFoundError(f"No {'npy' if npy else 'image'} files in {folder}")

    for p in paths:
        yield p.stem, loader(p)


def iter_frames_from_video(video_path: Path):
    """Генератор кадров из видеофайла через OpenCV."""
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "OpenCV не установлен. Установите: pip install opencv-python\n"
            "Или используйте папку с кадрами вместо видеофайла."
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Не удалось открыть видео: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # BGR → RGB → tensor [3, H, W] float32 [0,1]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(frame_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
        yield idx, tensor
        idx += 1

    cap.release()
    return fps, total, width, height


# ──────────────────────────────────────────────
# Запись результатов
# ──────────────────────────────────────────────

def save_to_folder(output_dir: Path, name: str, frame: torch.Tensor):
    output_dir.mkdir(parents=True, exist_ok=True)
    torchvision.utils.save_image(frame, str(output_dir / f"{name}.png"))


class VideoWriter:
    """Запись enhanced кадров в mp4 через OpenCV."""

    def __init__(self, output_path: Path, fps: float, width: int, height: int):
        try:
            import cv2
            self.cv2 = cv2
        except ImportError:
            raise ImportError("OpenCV не установлен: pip install opencv-python")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not self.writer.isOpened():
            raise RuntimeError(f"Не удалось создать видео: {output_path}")

    def write(self, frame: torch.Tensor):
        """frame: [3, H, W] float32 [0,1]"""
        arr = (frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        bgr = self.cv2.cvtColor(arr, self.cv2.COLOR_RGB2BGR)
        self.writer.write(bgr)

    def close(self):
        self.writer.release()


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='RetinexLNN inference')
    parser.add_argument('--config',     type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--input',      type=str, required=True,
                        help='Видеофайл (.mp4) или папка с кадрами')
    parser.add_argument('--output',     type=str, required=True,
                        help='Выходной файл (.mp4) или папка')
    parser.add_argument('--npy',        action='store_true',
                        help='Входные кадры в формате .npy (SDSD формат)')
    parser.add_argument('--fps',        type=float, default=None,
                        help='FPS для выходного видео (по умолчанию из конфига)')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Ограничить число кадров (для отладки)')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = get_device()
    print(f"\nRetinexLNN Inference")
    print(f"  Device: {device}")

    model = build_model(cfg, args.checkpoint, device)

    fps = args.fps or cfg['model'].get('fps', 30.0)
    window_size = cfg['model'].get('window_size', 5)
    inferencer = SlidingWindowInference(model, window_size, fps, device)

    input_path  = Path(args.input)
    output_path = Path(args.output)
    is_video_in  = input_path.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv'}
    is_video_out = output_path.suffix.lower() in {'.mp4', '.avi', '.mov'}

    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print(f"  FPS: {fps} | window_size: {window_size}\n")

    # Инференс
    t0 = time.time()
    frame_count = 0
    writer = None

    if is_video_in:
        frame_gen = iter_frames_from_video(input_path)
        first_frame = None
        all_enhanced = []

        for idx, frame in frame_gen:
            if args.max_frames and idx >= args.max_frames:
                break
            enhanced = inferencer.process_frame(frame)
            all_enhanced.append((idx, enhanced))

            if first_frame is None:
                first_frame = enhanced
                if is_video_out:
                    _, H, W = enhanced.shape
                    writer = VideoWriter(output_path, fps, W, H)

            if writer:
                writer.write(enhanced)
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"  Processed {frame_count} frames...")

        if writer:
            writer.close()
        elif all_enhanced:
            # Сохраняем как папку с кадрами
            for idx, frame in all_enhanced:
                save_to_folder(output_path, f"{idx:05d}", frame)

    else:
        # Папка с кадрами
        frame_gen = iter_frames_from_folder(input_path, npy=args.npy)
        for name, frame in frame_gen:
            if args.max_frames and frame_count >= args.max_frames:
                break
            enhanced = inferencer.process_frame(frame)
            save_to_folder(output_path, name, enhanced)
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"  Processed {frame_count} frames...")

    elapsed = time.time() - t0
    print(f"\n  Done: {frame_count} frames in {elapsed:.1f}s "
          f"({frame_count/elapsed:.1f} fps)")
    print(f"  Output: {output_path}")


if __name__ == '__main__':
    main()
