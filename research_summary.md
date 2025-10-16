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

## OUC-CGE
| Tahun | Judul (tautan)                                                                                                   | Varian Dataset | Metode                              | Hasil (Acc / Rec / Prec / F1 / AUC)  | Rangkuman SOTA singkat                                                                                                                                |
| ----- | ---------------------------------------------------------------------------------------------------------------- | -------------- | ----------------------------------- | ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2025  | [A Video Dataset for Classroom Group Engagement Recognition](https://www.nature.com/articles/s41597-025-04987-w) | OUC-CGE1       | **SLOW (ResNet-50)**                | **97.8 / 97.8 / 97.6 / 97.7 / 99.8** | Model sederhana dengan *low-freq bias* unggul untuk sinyal keterlibatan kelompok yang stabil; performa tertinggi & AUC ~0.998.  ([PubMed Central][1]) |
| 2025  | sama                                                                                                             | OUC-CGE1       | **SLOW-NLN (SLOW + NLN block)**     | **97.4 / 97.2 / 97.1 / 97.1 / 99.8** | Tambahan modul NLN menjaga akurasi sangat tinggi; mendekati plafon tugas 3-kelas berbasis isyarat spasial.                                            |
| 2025  | sama                                                                                                             | OUC-CGE1       | **X3D**                             | **96.8 / 96.5 / 96.3 / 96.4 / 99.8** | Arsitektur efisien spatio-temporal; hampir setara SLOW, menandakan dominasi fitur spasial.                                                            |
| 2025  | sama                                                                                                             | OUC-CGE1       | **I3D**                             | **94.3 / 94.0 / 93.7 / 93.8 / 98.8** | 3D conv klasik masih kuat namun tertinggal dari SLOW/X3D pada skenario 10-detik.                                                                      |
| 2025  | sama                                                                                                             | OUC-CGE1       | **C2D (2D Conv + temporal tricks)** | **93.3 / 92.8 / 92.9 / 92.8 / 98.7** | 2D murni sudah >93% → menegaskan label & tampilan ruang (postur, formasi kelompok) sangat informatif.                                                 |
| 2025  | sama                                                                                                             | OUC-CGE1       | **SlowFast**                        | **93.2 / 92.9 / 92.8 / 92.8 / 99.1** | Jalur “Fast” kurang relevan untuk dinamika lambat; SLOW mengungguli SlowFast pada tugas ini.                                                          |

[1]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12003871/?utm_source=chatgpt.com "A Video Dataset for Classroom Group Engagement ..."


#### Catatan jika menggunakan OUC-GCE
**AUC makro SLOW ≈ 0.99;** namun akurasi tinggi bisa mengindikasikan plafon/overfit spesifik dataset (kelas kasar & dominasi fitur spasial). Disarankan *cross-dataset validation*.  
**Sumber:** [PubMed Central (PMC12003871)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12003871/) • [Scientific Data – s41597-025-04987-w](https://www.nature.com/articles/s41597-025-04987-w)

**Analisis confusion matrix** memotret mis-read postur statis (mis. dagu bersandar disangka fokus) & interferensi gerakan kecil (putar pena vs angkat tangan).  
**Sumber:** [PubMed Central (PMC12003871)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12003871/) • [Scientific Data – s41597-025-04987-w](https://www.nature.com/articles/s41597-025-04987-w)

### Metode apa yang akan kita pakai??
<p>Refer ke tabel sebelumnya untuk cek perbandingan metode yang digunakan, nah berikut jika kita pakai metode lain selain yang ada di paper, terimakasih bantuannya chat gpt, you're the best.  <p>

| Keluarga metode                            | Contoh (tautan)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | Kenapa cocok untuk OUC-CGE                                                                                   | Kelebihan                                                                                                                                    | Kelemahan / Catatan                                                                                            |
| ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| **Video Transformer (global/ber-jendela)** | **TimeSformer** ([arXiv](https://arxiv.org/abs/2102.05095); [PMLR PDF](https://proceedings.mlr.press/v139/bertasius21a/bertasius21a.pdf)), **Video Swin** ([arXiv](https://arxiv.org/abs/2106.13230); [CVPR’22 PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Video_Swin_Transformer_CVPR_2022_paper.pdf)), **MViTv2** ([arXiv](https://arxiv.org/abs/2112.01526); [CVPR’22 PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_MViTv2_Improved_Multiscale_Vision_Transformers_for_Classification_and_Detection_CVPR_2022_paper.pdf)) | Dapat memodelkan konteks spasio-temporal panjang (diskusi kelompok, koordinasi gerak) dan *scaling* efisien. | Akurasi papan atas di Kinetics/SSv2; fleksibel urai temporal (divided attention / locality). ([Proceedings of Machine Learning Research][1]) | Butuh pretraining & compute; risiko overfit jika tidak *regularized*.                                          |
| **Dua tahap: Deteksi→Agregasi**            | **YOLOv8/11** untuk deteksi orang/gesture → agregasi fitur (proporsi “menghadap depan”, jarak antar siswa, intensitas gerak) menjadi skor engagement                                                                                                                                                                                                                                                                                                                                                                                                            | Cocok jika ingin **interpretabilitas** per-kelompok (indikator jelas).                                       | Explainable (fitur desain: % kepala menghadap meja, frekuensi angkat tangan); dapat multi-view dengan *late fusion*.                         | OUC-CGE cuma punya **label tingkat-kelompok** → perlu desain label turunan/heuristik; pipeline lebih kompleks. |
| **Pose/Graph (skeleton-based)**            | **ST-GCN** ([arXiv](https://arxiv.org/abs/1801.07455); [PDF](https://arxiv.org/pdf/1801.07455))                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | Agregasi pola postur/koordinasi antarmahasiswa → sinyal kolaborasi.                                          | Robust terhadap latar/pencahayaan; fitur manusiawi (*human-centric*). ([arXiv][2])                                                           | Perlu ekstraksi pose yang stabil (multi-view kelas bisa menantang); sinyal halus di jarak jauh bisa hilang.    |

[1]: https://proceedings.mlr.press/v139/bertasius21a/bertasius21a.pdf?utm_source=chatgpt.com "Is Space-Time Attention All You Need for Video Understanding?"
[2]: https://arxiv.org/abs/1801.07455?utm_source=chatgpt.com "Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition"





## Rekomendasi pemilihan metode (agar *alasan ilmiahnya* kuat)

**Tujuan**: kamu (D4) dan rekan S2 pakai **metode inti yang sama**, tapi *kedalaman kontribusi* berbeda.

**Metode inti yang disarankan (sama untuk D4 & S2):**

* **SLOW-ResNet50** sebagai baseline utama (selaras dengan karakter OUC-CGE; terbukti terbaik). Untuk bandingan efisiensi, sertakan **X3D-S**. Laporkan **Accuracy, macro-F1, macro-AUC** sesuai protokol paper. ([Nature][1])

**Ekstensi khusus per jenjang:**

* **D4 (aplikatif & solid)**:

  * Replikasi **SLOW** + **X3D** pada OUC-CGE; tambah **uji silang sudut/tata ruang** (train di view depan → test di samping/belakang; train checkerboard → test round-table).
  * Laporan *ablasi* panjang klip & fps (mis. 4 fps vs 8 fps) untuk menegaskan hipotesis frekuensi rendah. ([Nature][1])
* **S2 (kontribusi metodologis)** — pilih **satu** jalur:

  1. **Transformer Video** (mis. **Video Swin-T/B** atau **TimeSformer-B**): *pretrain* di Kinetics, *fine-tune* di OUC-CGE, tambahkan **ordinal loss** (Low<Med<High) + **calibration** (ECE/NLL) untuk keputusan pedagogis yang reliabel. ([arXiv][5])
  2. **Dua tahap YOLO→Agregasi**: deteksi orang/gesture sederhana lalu hitung **indikator kelompok** (mis. % kepala menghadap depan, jarak rata-rata, frekuensi angkat tangan) dan latih **classifier ordinal** untuk 3 level. Kuat di **interpretabilitas praktis** (butuh desain fitur & validasi pakar).
  3. **Pose/Graph (ST-GCN)**: ekstrak pose multi-view, bentuk **graph siswa** (node=orang, edge=jarak/arah tatapan) → klasifikasi engagement; tambahkan **temporal smoothing** untuk trajektori 20–30 detik. ([arXiv][4])

**Kenapa ini kuat untuk thesis S2?**
Ada **rasional teoretis** (OUC-CGE berfrekuensi rendah, multi-view, group-level) dan **gap** yang ditutup: (i) generalisasi lintas sudut/tata ruang, (ii) formulasi **ordinal + kalibrasi**, (iii) **interpretabilitas** indikator kelompok—sejalan dengan temuan & saran penulis dataset. ([Nature][2])



[1]: https://www.nature.com/articles/s41597-025-04987-w.pdf "A Video Dataset for Classroom Group Engagement Recognition"
[2]: https://www.nature.com/articles/s41597-025-04987-w "A Video Dataset for Classroom Group Engagement Recognition | Scientific Data"
[3]: https://proceedings.mlr.press/v139/bertasius21a/bertasius21a.pdf?utm_source=chatgpt.com "Is Space-Time Attention All You Need for Video Understanding?"
[4]: https://arxiv.org/abs/1801.07455?utm_source=chatgpt.com "Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition"
[5]: https://arxiv.org/abs/2106.13230?utm_source=chatgpt.com "Video Swin Transformer"


### Tadi coba prompt untuk dibuatkan template tabel eksperimen (SLOW/X3D/Video-Swin) + skrip evaluasi (accuracy/F1/AUC, uji silang view/layout) 

Karena tabel eksperimen sangat mantap! dan terlalu panjang, saya buatkan file baru, silakan refer ke file table_experiment.md [disini](table_experiment.md)