# Rangkuman riset dari pengumpulan dataset serta pembuatan state of the art penelitian


## SCB-dataset
| Tahun     | Judul (tautan)                                                                                   | Varian dataset SCB                                             | Metode (inti)                                                                       | Hasil utama (mAP/temuan)                                                                                                           | Rangkuman SOTA (singkat)                                                                                                                               |
| --------- | ------------------------------------------------------------------------------------------------ | -------------------------------------------------------------- | ----------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 2023      | **Student Behavior Detection in the Classroom Based on Improved YOLOv8**                         | SCB (obj det)                                                  | YOLOv8 + C2f_Res2block (Res2Net), EMA attention, MHSA                               | mAP@0.5 di SCB **76.3%**; ↑ **+4.2%** vs YOLOv8 dasar; juga lebih baik di CrowdHuman                                               | Menunjukkan kombinasi backbone multi-skala + attention di YOLOv8 efektif untuk kelas padat & oklusi dalam skenario kelas nyata. ([MDPI][1])            |
| 2023      | **Student Classroom Behavior Detection based on YOLOv7-BRA and Multi-Model Fusion**              | SCB (8 perilaku)                                               | YOLOv7 + Bi-level Routing Attention; fusi YOLOv7 CrowdHuman + SlowFast + DeepSort   | mAP@0.5 **87.1%** (↑2.2% dari sebelumnya) pada SCB                                                                                 | SOTA awal di SCB berbasis YOLOv7 + attention, menunjukkan fusi multi-model membantu deteksi perilaku halus (raise hand, read, write). ([arXiv][2])     |
| 2023      | **SCB-Dataset3: A Benchmark for Detecting Student Classroom Behavior**                           | **SCB-Dataset3** (6 perilaku)                                  | Benchmark YOLOv5/7/8 pada data kelas riil                                           | Skor terbaik mencapai **mAP 80.3%**                                                                                                | Paper acuan benchmark SCB3—menetapkan tolok ukur & kelas target (hand-raise, read, write, phone, head-down, lean). ([arXiv][3])                        |
| 2024      | **CSB-YOLO: a rapid and efficient real-time algorithm for classroom student behavior detection** | SCB-Dataset3                                                   | BiFPN, ERD-Head, SCConv + pruning & distillation (real-time)                        | Model terdistil + dipruning **0.72M params, 4.3 GFLOPs** dengan akurasi “tinggi” pada SCB3                                         | Garis SOTA untuk **edge/real-time** di kelas; fokus ke efisiensi tanpa mengorbankan akurasi pada SCB3. ([SpringerLink][4])                             |
| 2025      | **A WAD-YOLOv8-based method for classroom student behavior detection**                           | SCB, **SCB2**, **SCB-S**, **SCB-U**                            | YOLOv8 + CA-C2f, 2DPE-MHA (attention), Dysample (dynamic sampling)                  | Peningkatan mAP@0.5 vs YOLOv8: **+2.2% (SCB)**, **+3.3% (SCB2)**, **+5.5% (SCB-S)**, **+18.7% (SCB-U)**; juga naik di mAP@0.5:0.95 | Menetapkan SOTA praktis lintas beberapa varian SCB; desain receptive-field & attention yang lebih kaya menangani target kecil/teroklusi. ([Nature][5]) |
| 2025      | **Real-time classroom student behavior detection based on improved YOLOv8s**                     | **SCB-Dataset3-S**, **SCB-Dataset3-U**                         | YOLOv8s + MLKCM (large-kernel multi-scale), PFOM (progressive feature optimization) | mAP **76.5%** (SCB3-S) & **95.0%** (SCB3-U), unggul dari baseline                                                                  | Menunjukkan **model ringan** bisa kompetitif dengan desain large-kernel + optimisasi fitur progresif untuk real-time. ([Nature][6])                    |
| 2023/2024 | **SCB-dataset: A Dataset for Detecting Student Classroom Behavior**                              | **SCB (5/3/2/… seri)** termasuk **SCB-Dataset5** (19–20 kelas) | Paper dataset + baseline YOLOv7 series                                              | Menyediakan data besar (hingga **7,4k gambar / 106k label**, 19–20 kelas) + baseline deteksi                                       | Fondasi data utama untuk riset tindakan siswa & guru di kelas nyata—jadi referensi wajib untuk definisi label & protokol evaluasi. ([arXiv][7])        |
| 2025      | **Classroom Behavior Detection Method Based on PLA-YOLO11n**                                     | **SCB2**                                                       | YOLOv11n + modul C3K2_PConv, LSKA, pengganti SPPF→AIFI, head resolusi tinggi        | **+3.8% mAP@0.5** vs YOLOv11 pada SCB2                                                                                             | Pembuktian bahwa arsitektur **YOLOv11** + partial conv & attention besar memberi kenaikan nyata di SCB2 (kecil/teroklusi). ([MDPI][8])                 |

[1]: https://www.mdpi.com/1424-8220/23/20/8385 "Student Behavior Detection in the Classroom Based on Improved YOLOv8"
[2]: https://arxiv.org/abs/2305.07825 "[2305.07825] Student Classroom Behavior Detection based on YOLOv7-BRA and Multi-Model Fusion"
[3]: https://arxiv.org/abs/2310.02522 "[2310.02522] SCB-Dataset3: A Benchmark for Detecting Student Classroom Behavior"
[4]: https://link.springer.com/article/10.1007/s11554-024-01515-8 "Csb-yolo: a rapid and efficient real-time algorithm for classroom student behavior detection | Journal of Real-Time Image Processing"
[5]: https://www.nature.com/articles/s41598-025-87661-w "A WAD-YOLOv8-based method for classroom student behavior detection | Scientific Reports"
[6]: https://www.nature.com/articles/s41598-025-99243-x "Real-time classroom student behavior detection based on improved YOLOv8s | Scientific Reports"
[7]: https://arxiv.org/abs/2304.02488?utm_source=chatgpt.com "SCB-dataset: A Dataset for Detecting Student Classroom Behavior"
[8]: https://www.mdpi.com/1424-8220/25/17/5386 "Classroom Behavior Detection Method Based on PLA-YOLO11n"

#### Catatan jika menggunakan scb dataset
<p>Mayoritas SOTA mengandalkan YOLO-family + attention (MHSA/SCConv/CA), receptive-field lebih luas (large kernels), dan feature-fusion yang ditingkatkan untuk menangani kelas padat, oklusi, target kecil.

Tren lain: model ringan + distillation + pruning (CSB-YOLO) untuk deploy on-edge; dan perbandingan lintas varian SCB (WAD-YOLOv8) agar hasil lebih generalizable. <p> 
#### Saran jika menggunakan scb dataset
<p>Ukuran & generalisasi lintas-ruang
Uji/modelkan pada beberapa varian SCB (SCB, SCB2, SCB-S, SCB-U) dengan skema label yang diseragamkan, lalu laporkan cross-domain transfer (train satu varian → test varian lain).

Temporal + tracking untuk engagement
Tambah tracker (mis. ByteTrack/OC-SORT) dan temporal smoothing untuk menghitung metrik Engaged-Time Ratio per siswa (mis. % waktu “menulis/membaca/mengangkat tangan” vs “head-down/phone”).

Ablasi oklusi/kemiringan kamera
Karena kamera kelas dipasang di depan/pojok, buat stress-test: oklusi berat, pencahayaan beragam, FoV sempit. Laporkan sensitivitas performa terhadap sudut kamera.

Ringan & real-time on-edge
Bandingkan YOLOv8/11 “n/s” + distillation vs model besar; laporkan latensi FPS, GFLOPs, & daya GPU/CPU ala CSB-YOLO. 
SpringerLink

Kebisingan anotasi & active learning
SCB menghimpun berbagai sumber; tambahkan uncertainty-driven relabeling/active learning untuk memperbaiki label noizy sebelum training final.

Etika & privasi
Sertakan pipeline blur wajah + rilis hanya bounding box/metadata untuk menjaga kepatuhan etika saat publikasi demo. <p>

## OUC-GCE
| Tahun | Judul (tautan)                                                                                                   | Varian Dataset | Metode                              | Hasil (Acc / Rec / Prec / F1 / AUC)  | Rangkuman SOTA singkat                                                                                                                                |
| ----- | ---------------------------------------------------------------------------------------------------------------- | -------------- | ----------------------------------- | ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2025  | [A Video Dataset for Classroom Group Engagement Recognition](https://www.nature.com/articles/s41597-025-04987-w) | OUC-CGE1       | **SLOW (ResNet-50)**                | **97.8 / 97.8 / 97.6 / 97.7 / 99.8** | Model sederhana dengan *low-freq bias* unggul untuk sinyal keterlibatan kelompok yang stabil; performa tertinggi & AUC ~0.998.  ([PubMed Central][1]) |
| 2025  | sama                                                                                                             | OUC-CGE1       | **SLOW-NLN (SLOW + NLN block)**     | **97.4 / 97.2 / 97.1 / 97.1 / 99.8** | Tambahan modul NLN menjaga akurasi sangat tinggi; mendekati plafon tugas 3-kelas berbasis isyarat spasial.                                            |
| 2025  | sama                                                                                                             | OUC-CGE1       | **X3D**                             | **96.8 / 96.5 / 96.3 / 96.4 / 99.8** | Arsitektur efisien spatio-temporal; hampir setara SLOW, menandakan dominasi fitur spasial.                                                            |
| 2025  | sama                                                                                                             | OUC-CGE1       | **I3D**                             | **94.3 / 94.0 / 93.7 / 93.8 / 98.8** | 3D conv klasik masih kuat namun tertinggal dari SLOW/X3D pada skenario 10-detik.                                                                      |
| 2025  | sama                                                                                                             | OUC-CGE1       | **C2D (2D Conv + temporal tricks)** | **93.3 / 92.8 / 92.9 / 92.8 / 98.7** | 2D murni sudah >93% → menegaskan label & tampilan ruang (postur, formasi kelompok) sangat informatif.                                                 |
| 2025  | sama                                                                                                             | OUC-CGE1       | **SlowFast**                        | **93.2 / 92.9 / 92.8 / 92.8 / 99.1** | Jalur “Fast” kurang relevan untuk dinamika lambat; SLOW mengungguli SlowFast pada tugas ini.                                                          |

[1]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12003871/?utm_source=chatgpt.com "A Video Dataset for Classroom Group Engagement ..."
