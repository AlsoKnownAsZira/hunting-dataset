# deploy yolov11 menggunakan model default: yolo11n.pt
## Bisa dicek di https://huggingface.co/spaces/AlsoKnownAsZira/zira-deploy-yolo
pada HF itu, bisa menggunakan stream via webcam/upload foto <br>
Untuk stream via webcam, frame masih agak patah patah karena mungkin cpu usage dilimit (akun gratisan hehe) <br>
Ada slider untuk confidence, IOU dan ukuran img (default 480) <br>
untuk source code bisa dicek di: https://huggingface.co/spaces/AlsoKnownAsZira/zira-deploy-yolo/tree/main
<br>
Deploy di railway dengan fastapi dan html on progress
<br>