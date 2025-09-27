# Manga OCR & Typeset Tool v14.1.0

[![Version](https://img.shields.io/badge/version-14.1.0-blue)]() [![License](https://img.shields.io/badge/license-MIT-green)]()

## v14.1.0 (Current)

- **[MAJOR] OpenAI Prompt Caching (hemat biaya)**  
  - Menambahkan `cache_control: {"type": "ephemeral"}` pada **system prompt** di endpoint Chat Completions.  
  - Input tokens per request turun drastis (stabil ~300-an), karena system prompt tidak dihitung berulang.  
  - UI menampilkan **Input/Output Tokens** per request & total, agar efek caching terlihat real-time.

- **[MAJOR] API Status Panel — metrik lengkap & real-time**  
  - Tambahan label: **Provider**, **Model**, **Input Tokens**, **Output Tokens**, **Total Tokens**, **Rate Input/Output** per token, **Translated Snippets (counter)**.  
  - Tetap menampilkan **RPM / RPD**, **Cost (USD/IDR)**, dan **Active Workers**.

- **[MAJOR] Pricing & Costing akurat berbasis token (bukan karakter)**  
  - `add_api_cost()` kini memakai data `response.usage.prompt_tokens` & `completion_tokens` (OpenAI) dan tabel harga per 1K token (untuk semua model).  
  - Konversi **USD→IDR** dipertahankan; perhitungan granular per request + total.  
  - Penyesuaian rate untuk model murah (gpt-5-nano/mini) vs Gemini 2.5 (Flash/Lite/Pro).

- **[MAJOR] Guardrails Gemini (stabilitas & biaya)**  
  - Default `max_output_tokens=512` (bisa atur dari settings) untuk cegah over-output.  
  - Opsi **safety_settings** dilonggarkan untuk kasus manga agar tidak sering terblokir.  
  - **Fallback otomatis**: jika Gemini gagal (timeout/overthinking), langsung retry fallback ke OpenAI model yang ditentukan.  
  - Peringatan & pencegahan untuk **batch via Gemini** (opsional: blokir batch Gemini atau pakai model whitelist).

- **[MAJOR] Batch Routing Aware Provider (eksperimental, opsional)**  
  - Logika batch menyesuaikan endpoint terbaik per provider (mis. OpenAI batch/batches bila tersedia, atau tetap single-call berantai dengan queue control).  
  - Auto backoff & retry pada kasus rate-limit.

- **[IMPROVEMENT] Translator hygiene**  
  - Output sanitizer sederhana (hapus code fence & pembungkus) untuk memastikan hasil benar-benar “RAW plain text”.  
  - Temperatur otomatis dinonaktifkan untuk model yang tidak mendukung (gpt-5-mini/nano).

- **[FIX] Stability & UX**  
  - Counter **Translated Snippets** naik hanya saat terjemahan sukses.  
  - Minor typo pada changelog (“vv14.0.0”) **diperbaiki** → “v14.0.0”.  
  - Reset Undo/Redo aman setelah apply hasil dari dialog editor.

---

## Pipeline Detail: Manga OCR & Typeset Tool v14.0.1

Berikut adalah **pipeline detail** untuk alur sistem yang telah ditingkatkan:

### **1. Pipeline Utama Sistem**

```mermaid
flowchart LR
    A[Input] --> B[Preprocessing] --> C[OCR Multi-Engine] --> D[AI Enhancement] --> E[Translation] --> F[Glossary Integration] --> G[Typesetting] --> H[Output]
```

### **2. Pipeline Detail per Komponen**

#### **A. Input & Manajemen Proyek**

- **Tujuan**: Memuat sumber gambar/PDF dan mengelola proyek
- **Komponen**:

```mermaid
flowchart LR
    A1[File Dialog] --> A2[Load Image/PDF] --> A3[Deteksi Format] --> A4{PDF?}
    A4 -->|Ya| A5[Ekstrak Halaman dengan PyMuPDF]
    A4 -->|Tidak| A6[Konversi ke RGB dengan PIL]
    A5 & A6 --> A7[Setup Project Directory]
    A7 --> A8[Load Glossary dari JSON]
```

#### **B. Preprocessing Cerdas**

- **Tujuan**: Optimalkan gambar untuk OCR dengan deteksi orientasi otomatis
- **Komponen**:

```mermaid
flowchart LR
    B1[Konversi ke Grayscale] --> B2[Deteksi Orientasi Otomatis] --> B3[Rotasi Korektif] --> B4[Adaptive Thresholding] --> B5[Denoising]
```

#### **C. OCR Multi-Engine dengan Fallback**

- **Tujuan**: Ekstraksi teks dari gambar dengan engine terbaik untuk bahasa tertentu
- **Engine & Strategi**:

```mermaid
flowchart TD
    C1[Pilih Berdasarkan Bahasa] --> C2{Japanese?}
    C2 -->|Ya| C3[Manga-OCR Prioritas]
    C2 -->|Tidak| C4{Chinese/Korean?}
    C4 -->|Ya| C5[PaddleOCR Prioritas]
    C4 -->|Tidak| C6[Tesseract/EasyOCR]
    C3 & C5 & C6 --> C7[Gabungkan Hasil jika Enhanced Pipeline]
```

#### **D. Enhanced AI Pipeline**

- **Tujuan**: Tingkatkan akurasi dengan kombinasi Manga-OCR + Tesseract + Gemini
- **Alur**:

```mermaid
flowchart LR
    D1[Manga-OCR Result] --> D2[Tesseract Result] --> D3[Analisis & Merge dengan Gemini] --> D4[Koreksi OCR Errors] --> D5[Terjemahan Kontekstual]
```

#### **E. Sistem Glosarium AI-Powered**

- **Tujuan**: Deteksi otomatis istilah penting dan pertahankan konsistensi terjemahan
- **Alur**:

```mermaid
flowchart TD
    E1[Original Text] --> E2[Translated Text] --> E3[AI Analysis Gemini]
    E3 --> E4[Deteksi Proper Nouns] --> E5[Deteksi Unique Terms] --> E6[Deteksi Honorifics]
    E4 & E5 & E6 --> E7[Generate Suggestions] --> E8[Update Glossary]
    E8 --> E9[Integrasi ke Prompt Terjemahan]
```

#### **F. Dynamic Worker Pool**

- **Tujuan**: Pemrosesan paralel untuk throughput maksimal
- **Strategi**:

```mermaid
flowchart LR
    F1[Job Queue] --> F2{Queue Size > Threshold?}
    F2 -->|Ya| F3[Spawn New Worker]
    F2 -->|Tidak| F4[Wait]
    F3 --> F5[Process Job] --> F6[Signal Completion]
    F6 --> F7[Update UI Thread-safe]
```

#### **G. Typesetting & Output**

- **Tujuan**: Render teks terjemahan ke gambar dengan preservasi konteks visual
- **Komponen**:

```mermaid
flowchart LR
    G1[Inpainting Area Asli] --> G2[Deteksi Bubble Mask] --> G3[Font Scaling Adaptif] --> G4[Render Teks dengan Outline] --> G5[Simpan/Export]
```

---

## Fitur Utama

### 1. OCR Multi-Engine Cerdas
- **Manga-OCR** – Dioptimalkan untuk teks manga Jepang  
- **EasyOCR** – Deteksi multi-bahasa dengan dukungan GPU  
- **Tesseract** – Engine tradisional dengan dukungan bahasa luas  
- **PaddleOCR** – Khusus untuk bahasa China/Korea (opsional)  
- **Deteksi Orientasi Otomatis** dengan rotasi korektif  

### 2. AI-Powered Translation & Enhancement
- **Gemini Integration** untuk koreksi OCR dan terjemahan kontekstual  
- **Enhanced Pipeline** dengan kombinasi Manga-OCR + Tesseract + Gemini  
- **Style Selection**: Santai, Formal, Akrab, Vulgar/Dewasa, Sesuai Konteks Manga  
- **Auto-censorship** untuk konten eksplisit ("vagina" → "meong", "penis" → "burung")  

### 3. Dynamic Worker Pool
- **Scalable Processing** – Hingga 15 worker thread paralel  
- **Intelligent Scaling** – Worker baru dibuat berdasarkan ukuran antrian  
- **Resource Management** – Worker otomatis dihentikan saat idle  
- **Thread-safe UI Updates** – Pembaruan antarmuka yang aman dari thread  

### 4. Advanced Typesetting Tools
- **Selection Tools**: Rectangle, Pen tool (polygon bebas), opsi bubble toggle  
- **Advanced Text Editor Modal** – Kontrol font, ukuran, warna, alignment, spacing, margin, efek manga-style (curved, wavy, jagged), Bezier curve untuk teks melengkung, serta emoji  
- **Vertical & Horizontal Typesetting** – Dukungan orientasi teks ganda  
- **Inpainting Algorithms** – Navier-Stokes, Telea, dan Big-LAMA/Anime-Inpainting yang lebih natural  

### 5. Manajemen Proyek Lengkap
- **Save/Load Project** (`.manga_proj`) dengan semua metadata  
- **Autosave** setiap 5 menit  
- **Batch Processing** – Proses seluruh folder sekaligus  
- **PDF Support** – Buka, edit, dan ekspor PDF  

### 6. API & Resource Management
- **Multi-API Support** – Gemini (berbagai model) dan DeepL  
- **Rate Limit Management** – Monitoring RPM/RPD real-time  
- **Cost Tracking** – Pelacakan biaya API dalam USD dan IDR  
- **Automatic Retry** – Mekanisme antrian saat limit tercapai  

---

## Kelebihan v14.0.1

**Kelebihan**
- UI/UX lebih modern dan responsif, adaptif di semua ukuran layar (desktop, tablet, mobile)  
- Advanced text editor modal dengan kontrol font, warna, efek manga-style, dan styling per kata  
- Toggle bubble opsional untuk auto-render bubble putih dengan outline hitam  
- Worker pool dinamis yang lebih stabil dan efisien untuk pemrosesan batch besar  
- Pipeline rendering diperbarui untuk mendukung margin, alignment, spacing, serta teks vertikal maupun efek path-based  

**Pertimbangan**
- Konsumsi memori meningkat jika banyak worker aktif dan rich text digunakan  
- Dependency lebih kompleks, termasuk kebutuhan GPU untuk performa optimal pada EasyOCR/PaddleOCR  
- Biaya API berpotensi lebih tinggi dengan penggunaan AI translation intensif  
- Learning curve lebih tinggi karena fitur editing lebih advance dibanding versi sebelumnya  

---

## Instalasi

**Prasyarat**

- Python 3.8+
- Tesseract OCR (install terpisah)
- GPU recommended untuk EasyOCR/PaddleOCR

**Langkah Instalasi**

```bash
# Buat virtual environment
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate    # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install engines OCR opsional
pip install manga-ocr    # Untuk Manga-OCR
pip install paddleocr    # Untuk PaddleOCR
```

**File requirements.txt:**

```
PyQt5>=5.15
numpy>=1.21
opencv-python>=4.5
requests>=2.28
Pillow>=9.0
google-generativeai>=0.3.0
pytesseract>=0.3.10
easyocr>=1.6
PyMuPDF>=1.22
openai>=1.0.0
lama-cleaner>=1.0.0
```

---

## Konfigurasi (config.ini)

File konfigurasi dibuat otomatis saat pertama kali menjalankan aplikasi:

```ini
[API]
DEEPL_KEY = API-KEY
GEMINI_KEY = API-KEY
OPENAI_KEY = API-KEY

[PATHS]
TESSERACT_PATH = C:\Program Files\Tesseract-OCR\tesseract.exe

[MODELS]
BIG_LAMA_PATH = big-lama\models\best.ckpt
ANIME_INPAINT_PATH = models\lama_large_512px.ckpt
```

**Catatan**: Ganti placeholder dengan API key sebenarnya untuk menggunakan fitur AI.

---

## Cara Penggunaan

### 1. Setup Awal

- Jalankan aplikasi: `python main.py`
- Konfigurasi API keys di `config.ini`
- Muat folder project dengan gambar/PDF

### 2. Proses Translasi

1. Pilih engine OCR berdasarkan bahasa sumber
2. Tentukan style terjemahan yang diinginkan
3. Seleksi area teks dengan rectangle/pen tool
4. Review hasil OCR jika perlu
5. Terjemahkan dengan Gemini/DeepL

### 3. Kelola Glosarium

- Buat entri manual di tab Glossary
- Review saran otomatis dari AI
- Terapkan glosarium untuk konsistensi

### 4. Batch Processing

- Gunakan "Detect All Files" untuk proses otomatis
- Review dan konfirmasi detected bubbles
- Proses seluruh bubble sekaligus

### 5. Export Results

- Simpan individual image dengan "Save Image"
- Export batch dengan "Batch Save"
- Export ke PDF untuk kumpulan halaman

---

## Troubleshooting

**Masalah Umum dan Solusi:**

1. **Engine OCR tidak berjalan**:

   - Pastikan Tesseract terinstall dan path benar di config.ini
   - Untuk Manga-OCR/PaddleOCR: `pip install manga-ocr paddleocr`

2. **Error API Limit**:

   - Monitor usage di status bar bawah
   - Tunggu hingga limit reset atau upgrade plan API

3. **Performance lambat**:

   - Kurangi jumlah worker di pengaturan
   - Gunakan GPU untuk engine OCR yang mendukung

4. **Glossary tidak tersimpan**:
   - Pastikan folder project mempunyai permission write
   - Cek file `glossary.json` di folder project

---

## API Usage & Cost Management

Aplikasi mendukung berbagai model Gemini dengan karakteristik berbeda:

| Model                 | Kecepatan  | Biaya        | Recommended Use                  |
| --------------------- | ---------- | ------------ | -------------------------------- |
| Gemini 2.5 Flash Lite | ⚡⚡⚡⚡⚡ | $0.0001/1K   | Default, pemrosesan tinggi       |
| Gemini 2.5 Flash      | ⚡⚡⚡⚡   | $0.000125/1K | Fallback 1, akurasi lebih tinggi |
| Gemini 2.5 Pro        | ⚡⚡⚡     | $0.0025/1K   | Teks kompleks dan penting        |
| Gemini 2.0 Flash Lite | ⚡⚡⚡⚡⚡ | $0.0001/1K   | Darurat saat limit 2.5           |

---

## License

MIT License - lihat file LICENSE untuk detail lengkap.

---

## Contributing

Kontribusi dipersilakan! Untuk fitur besar, silakan buka issue terlebih dahulu untuk didiskusikan.

**Panduan kontribusi:**

1. Fork repository
2. Buat feature branch
3. Commit changes
4. Push ke branch
5. Buat Pull Request

---

# Changelog

## v14.0.1 (Current)

- **[MAJOR]** Added an “Add bubble” toggle in the manual selection toolkit to optionally auto-render a white bubble with a black outline on confirmation  
- **[MAJOR]** Replaced inline editor with **AdvancedTextEditDialog**, supporting font/size/color controls, bold/italic/underline, alignment, line spacing, character spacing, margins, Bezier curve editing, manga-style effects (curved/wavy/jagged), orientation switching, and emoji insertion with partial text styling  
- **[IMPROVEMENT]** Expanded `TypesetArea` to persist rich formatting metadata including bubble options, alignment, spacing, margins, effects, and per-segment styling with safe default fallbacks  
- **[IMPROVEMENT]** Reworked the rendering pipeline to honor bubbles, inner margins, alignment, vertical/path-based text effects, and rich-text segments  
- **[IMPROVEMENT]** Updated OCR worker flows to seed new `TypesetArea` instances with default advanced-layout options across batch and single-processing  
- **[FIX]** Hooked edit actions into the new modal, applying user selections back into stored segments and resetting undo/redo cleanly  

## vv14.0.0

- **[IMPROVEMENT]** Modernized UI/UX to be responsive and adaptive across all screen sizes (desktop, tablet, mobile), ensuring smooth navigation and consistent layout  
- **[MAJOR]** Added a simple text editing modal: when users select an existing text, a modal window opens to edit text content directly, and after confirmation updates are applied back to the canvas  
- **[MAJOR]** Added a new **AI Hardware** tab to organize and display hardware-related settings separately from the OCR and editing functions

## v13.0.1

- **[FIX]** Runtime crash dalam worker glossary (`RuntimeError: wrapped C/C++ object of type QThread has been deleted`)
- **[IMPROVEMENT]** Worker pool scaling yang lebih agresif (threshold diturunkan dari 5 ke 3)

## v13.0.0

- **[MAJOR]** Intelligent worker pooling dengan hingga 15 thread paralel
- **[MAJOR]** AI-powered glossary system dengan saran otomatis
- **[MAJOR]** Glossary manager UI dengan file-based persistence
- **[ENHANCEMENT]** Enhanced pipeline dengan kombinasi Manga-OCR + Tesseract + Gemini

## v12.0.3

- **[FIX]** Crash `cv2.mean` dengan validasi dimensi gambar
- **[IMPROVEMENT]** Auto-add glossary suggestions dengan pencegahan duplikat
- **[IMPROVEMENT]** Expanded translation styles dengan prompt yang lebih deskriptif

## v12.0.2

- **[FIX]** `AttributeError` pada Gemini API calls
- **[IMPROVEMENT]** Kembalikan opsi orientasi teks (Horizontal/Vertical)
- **[IMPROVEMENT]** Logika penguncian bahasa OCR yang disempurnakan

## v8.0-v12.0

- Integrasi Manga-OCR dan PaddleOCR
- Deteksi bubble otomatis dengan model DL
- Batch processing dan PDF export
- Manajemen proyek dan autosave
- Inpainting dan advanced typesetting tools

---

Untuk informasi lebih lanjut, issue, atau kontribusi, silakan kunjungi repository GitHub project ini.


