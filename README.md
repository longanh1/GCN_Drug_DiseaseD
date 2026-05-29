# PharmaLink — GCN Drug-Disease Prediction Platform

Hệ thống dự đoán mối quan hệ thuốc–bệnh sử dụng Graph Convolutional Network (GCN) kết hợp Graph Transformer và Fuzzy Logic.

---

## Kiến trúc tổng quan

```
GCN_Drug_Disease-main/
├── AI_ENGINE/      # FastAPI — mô hình GCN, inference, training
├── BACKEND/        # NestJS  — REST API, xác thực JWT, PostgreSQL
├── FRONTEND/       # Streamlit — giao diện web
└── AMDGT_main/     # Module nghiên cứu / tiền xử lý dữ liệu gốc
```

Khi chạy, 3 service hoạt động song song:

| Service      | Công nghệ | Cổng  |
|-------------|-----------|-------|
| AI Engine   | FastAPI   | 8000  |
| Backend API | NestJS    | 3000  |
| Frontend    | Streamlit | 8501  |

---

## Yêu cầu hệ thống

| Công cụ        | Phiên bản tối thiểu | Ghi chú |
|---------------|--------------------|-----------------------------------------|
| **Python**    | 3.10 – 3.12        | **Khuyến nghị 3.12**. Không dùng 3.13+ (DGL chưa hỗ trợ) |
| **Node.js**   | 18+                | Khuyến nghị v20 LTS hoặc v22 LTS |
| **npm**       | 9+                 | Đi kèm Node.js |
| **PostgreSQL**| 14+                | Cần chạy trước khi khởi động Backend |
| **pgAdmin 4** | Bất kỳ             | Tùy chọn, để quản lý DB |
| **Git**       | Bất kỳ             | Để clone repo |

> **Windows:** Đảm bảo Python và Node.js đã được thêm vào `PATH` khi cài đặt.

---

## Cài đặt từng bước

### 1. Clone repository

```bash
git clone <your-repo-url>
cd GCN_Drug_Disease-main
```

---

### 2. Cài Python dependencies (AI Engine + Training)

Tạo và kích hoạt virtual environment:

```powershell
# Windows PowerShell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
python3.12 -m venv .venv
source .venv/bin/activate
```

Cài các thư viện AI Engine:

```bash
pip install --upgrade pip setuptools wheel
```

> **Quan trọng:** PyTorch và DGL phải cài theo đúng thứ tự và đúng index URL.

#### Cài PyTorch (CPU):

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cpu
```

> Nếu máy có GPU NVIDIA, thay `cpu` bằng `cu118` hoặc `cu121` tùy phiên bản CUDA.

#### Cài DGL:

```bash
pip install dgl==1.1.2
```

#### Cài các thư viện còn lại của AI_ENGINE:

```bash
pip install -r AI_ENGINE/requirements.txt
```

Nội dung `AI_ENGINE/requirements.txt`:

```
torch>=1.13.0
dgl>=0.9.0
networkx>=2.8
scikit-learn>=1.0.0
scikit-fuzzy>=0.4.2
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
fastapi>=0.95.0
uvicorn>=0.22.0
rdkit-pypi>=2022.9.5
pillow>=9.0.0
requests>=2.28.0
python-multipart>=0.0.6
aiofiles>=22.1.0
pydantic>=1.10.0
matplotlib>=3.5.0
```

#### Cài thêm thư viện cho Frontend Streamlit:

```bash
pip install -r FRONTEND/requirements.txt
```

Nội dung `FRONTEND/requirements.txt`:

```
streamlit>=1.28.0
requests>=2.28.0
plotly>=5.15.0
pandas>=1.5.0
numpy>=1.21.0
rdkit-pypi>=2022.9.5
pillow>=9.0.0
python-dotenv>=1.0.0
```

---

### 3. Cài Node.js dependencies (Backend NestJS)

```bash
cd BACKEND
npm install
cd ..
```

Các package chính được cài tự động qua `npm install`:

| Package | Phiên bản | Mục đích |
|---------|----------|---------|
| `@nestjs/core` | ^10.0.0 | Framework NestJS |
| `@nestjs/jwt` | ^11.0.2 | Xác thực JWT |
| `@nestjs/typeorm` | ^11.0.1 | ORM kết nối PostgreSQL |
| `typeorm` | ^1.0.0 | ORM |
| `pg` | ^8.21.0 | PostgreSQL driver |
| `bcrypt` | ^6.0.0 | Hash mật khẩu |
| `passport-jwt` | ^4.0.1 | JWT strategy |
| `axios` | ^1.6.0 | HTTP client |
| `class-validator` | ^0.15.1 | Validation DTO |

---

### 4. Cấu hình PostgreSQL

#### 4a. Tạo database

Mở pgAdmin 4 hoặc `psql` và chạy:

```sql
CREATE DATABASE pharmalink;
```

#### 4b. Chạy script khởi tạo bảng và tài khoản admin

Kết nối vào database `pharmalink`, sau đó chạy nội dung file `BACKEND/database_setup.sql`:

```bash
psql -U postgres -d pharmalink -f BACKEND/database_setup.sql
```

Script này sẽ tạo bảng `users` và tài khoản admin mặc định:

| Field    | Giá trị mặc định       |
|----------|----------------------|
| Email    | admin@pharmalink.local |
| Username | admin                 |
| Password | `Admin@12345`         |
| Role     | admin                 |

#### 4c. Tạo file `.env` cho Backend

Tạo file `BACKEND/.env` với nội dung:

```env
# Database
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=your_password_here
DB_NAME=pharmalink

# JWT
JWT_SECRET=your_jwt_secret_key_here
JWT_EXPIRES_IN=7d

# AI Engine
AI_ENGINE_URL=http://localhost:8000
```

> Thay `your_password_here` bằng mật khẩu PostgreSQL thực tế của bạn.
> Thay `your_jwt_secret_key_here` bằng một chuỗi bí mật ngẫu nhiên (tối thiểu 32 ký tự).

---

### 5. Kiểm tra cài đặt

```bash
# Kiểm tra Python + DGL + PyTorch
.\.venv\Scripts\python.exe -c "import dgl, torch; print('DGL:', dgl.__version__, '| Torch:', torch.__version__)"

# Kiểm tra Node.js
node --version
npm --version
```

---

## Khởi động toàn bộ hệ thống

### Cách 1 — Script tự động (Windows PowerShell)

```powershell
.\start_all.ps1
```

Script này sẽ tự động khởi động 3 service trong 3 cửa sổ PowerShell riêng biệt.

### Cách 2 — Khởi động thủ công từng service

**Terminal 1 — AI Engine (FastAPI):**

```powershell
cd AI_ENGINE
.\..\venv\Scripts\python.exe api.py
```

**Terminal 2 — Backend (NestJS):**

```powershell
cd BACKEND
npm run start:dev
```

**Terminal 3 — Frontend (Streamlit):**

```powershell
cd FRONTEND
..\..\venv\Scripts\streamlit.exe run app.py
```

### Truy cập

| URL | Mô tả |
|-----|-------|
| http://localhost:8501 | Giao diện người dùng (Streamlit) |
| http://localhost:3000/api | Backend REST API |
| http://localhost:8000/docs | Tài liệu FastAPI (Swagger UI) |

---

## Training mô hình

Dùng script `train.ps1` để huấn luyện lại mô hình (script tự tạo `.venv` nếu chưa có):

```powershell
# Huấn luyện mô hình fuzzy trên C-dataset (mặc định)
.\train.ps1

# Tùy chỉnh dataset và mô hình
.\train.ps1 -dataset B-dataset -model fuzzy -epochs 1000

# Các lựa chọn model: base | fuzzy | gcn | ablation | both
.\train.ps1 -dataset F-dataset -model base
```

Kết quả lưu tại: `AI_ENGINE/data/results/` và `AI_ENGINE/data/models/`

---

## Các dataset có sẵn

| Dataset | Mô tả |
|---------|-------|
| `B-dataset` | Benchmark dataset |
| `C-dataset` | Clinical dataset |
| `F-dataset` | Fdataset |

---

## Lỗi thường gặp

### `import dgl` thất bại
- Nguyên nhân: DGL không hỗ trợ Python 3.13+. Dùng Python **3.10–3.12**.
- Giải pháp: Tạo lại venv với Python 3.12, cài lại theo thứ tự ở mục 2.

### Backend không kết nối được PostgreSQL
- Kiểm tra PostgreSQL đang chạy (port 5432).
- Kiểm tra file `BACKEND/.env` đúng thông tin DB.
- Đảm bảo database `pharmalink` đã được tạo.

### Frontend báo lỗi kết nối API
- Đảm bảo AI Engine (port 8000) và Backend (port 3000) đang chạy trước.
- Kiểm tra không có firewall chặn các cổng trên.

### `rdkit-pypi` không cài được
- Thử dùng `pip install rdkit` thay thế (phiên bản mới hơn đã đổi tên package).

---

## Tóm tắt nhanh (Quick Setup)

```powershell
# 1. Tạo venv và cài Python deps
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
pip install dgl==1.1.2
pip install -r AI_ENGINE/requirements.txt
pip install -r FRONTEND/requirements.txt

# 2. Cài Node deps
cd BACKEND ; npm install ; cd ..

# 3. Tạo BACKEND/.env (xem hướng dẫn mục 4c)

# 4. Khởi tạo database PostgreSQL
psql -U postgres -d pharmalink -f BACKEND/database_setup.sql

# 5. Chạy toàn bộ
.\start_all.ps1
```
