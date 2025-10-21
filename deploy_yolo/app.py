"""
YOLOv11 + Gradio (CPU) — siap untuk Hugging Face Spaces
- Upload gambar atau pakai webcam
- Keluaran: gambar beranotasi + JSON boks deteksi
"""

from ultralytics import YOLO
import gradio as gr
import numpy as np

# Unduh & cache model kecil (deteksi) saat pertama run
MODEL_PATH = "yolo11n.pt"
model = YOLO(MODEL_PATH)
NAMES = model.names  # dict id->class

def detect(image: np.ndarray, conf: float = 0.25, iou: float = 0.45):
    """
    image: RGB ndarray dari Gradio (upload/webcam)
    return: (annotated_rgb, list_of_boxes)
    """
    if image is None:
        return None, []

    # Spaces biasanya CPU-only → paksa device="cpu" agar stabil
    results = model.predict(
        image,
        imgsz=640,
        conf=conf,
        iou=iou,
        device="cpu",
        verbose=False
    )

    r = results[0]
    # Annotasi dari Ultralytics (np.ndarray BGR) → balik ke RGB untuk Gradio
    annotated_rgb = r.plot()[:, :, ::-1]

    boxes_json = []
    if r.boxes is not None:
        for b in r.boxes:
            cls_id = int(b.cls[0])
            boxes_json.append({
                "class": NAMES.get(cls_id, str(cls_id)),
                "conf": float(b.conf[0]),
                "xyxy": [float(x) for x in b.xyxy[0].tolist()]
            })

    return annotated_rgb, boxes_json

with gr.Blocks(title="YOLOv11 Web Demo (Gradio)") as demo:
    gr.Markdown("## YOLOv11 Web Demo — unggah gambar atau pakai webcam")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(
                label="Input (Upload / Webcam)",
                sources=["upload", "webcam"],
                type="numpy"
            )
            conf = gr.Slider(0.1, 0.9, value=0.25, step=0.05, label="Confidence")
            iou  = gr.Slider(0.1, 0.9, value=0.45, step=0.05, label="IoU (NMS)")
            btn  = gr.Button("Deteksi")
        with gr.Column():
            out_img  = gr.Image(label="Hasil Anotasi")
            out_json = gr.JSON(label="Deteksi (JSON)")

    btn.click(detect, inputs=[inp, conf, iou], outputs=[out_img, out_json])

# Untuk run lokal: python app.py
if __name__ == "__main__":
    demo.queue(concurrency_count=2)
    demo.launch()  # tambahkan share=True jika ingin URL sementara
