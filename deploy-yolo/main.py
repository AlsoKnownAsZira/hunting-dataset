from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io, asyncio, os

# ====== Tuning threads CPU (baik untuk shared CPU) ======
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
try:
    import torch
    torch.set_num_threads(2)
except Exception:
    pass

app = FastAPI(title="YOLOv11 FastAPI Streaming")

# Kalau nanti ingin host front-end di domain lain, tetap aman:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model sekali saat startup
MODEL_PATH = "yolo11n.pt"  # model default YOLOv11 kecil
model = YOLO(MODEL_PATH)
NAMES = model.names  # mapping id->nama kelas

@app.get("/")
def home():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    conf: float = Form(0.25),
    iou: float = Form(0.45),
    imgsz: int = Form(480),
    max_det: int = Form(50),
):
    """Terima 1 frame (JPEG/PNG), balas JSON bbox untuk digambar di browser."""
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    def _run():
        res = model.predict(
            img,
            imgsz=int(imgsz),
            conf=float(conf),
            iou=float(iou),
            device="cpu",
            verbose=False,
            max_det=int(max_det),
        )
        r = res[0]
        boxes = []
        if r.boxes is not None:
            for b in r.boxes:
                cls_id = int(b.cls[0])
                boxes.append({
                    "class_id": cls_id,
                    "class": NAMES.get(cls_id, str(cls_id)),
                    "conf": float(b.conf[0]),
                    "xyxy": [float(x) for x in b.xyxy[0].tolist()],
                })
        return {"boxes": boxes, "img_w": img.width, "img_h": img.height}

    data = await asyncio.to_thread(_run)  # non-blocking untuk event loop
    return JSONResponse(data)
