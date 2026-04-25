# Code Review — `tontide1/Customer-Purchase-Prediction-MLOPS` (branch `main`)

**Commit reviewed:** `5ea7e05` (Merge PR #1 from `dev`)
**Date:** 2026-04-25

---

## 1. Tổng quan

Repo là một dự án MLOps end-to-end **đang ở giai đoạn rất sớm**: mới hoàn thành "Week 1 — Data Foundation" (bronze + silver layer). Phần lớn nội dung repo là **tài liệu blueprint** (~3,700 dòng markdown) mô tả kiến trúc target-state; chỉ một phần nhỏ là code thực thi (~1,800 dòng Python).

| Thành phần | Trạng thái |
|---|---|
| Blueprint / docs | ✅ Rất chi tiết, 12 file chia theo chủ đề |
| Bronze pipeline (raw → bronze parquet, chunked) | ✅ Có, memory-safe |
| Silver pipeline (bronze → silver, dedup + sort) | ✅ Có, nhưng load full vào RAM |
| DVC + MinIO scaffold | ✅ Có |
| Tests (pytest) | ⚠️ **10 tests fail + 1 collection error** trên main |
| `requirements.txt` / `pyproject.toml` | ❌ Không có |
| `README.md` (top-level) | ❌ Không có |
| Pre-commit / lint / CI workflow | ❌ Không có |
| Feature engineering / training / serving / monitoring | ❌ Chưa triển khai (đúng như roadmap) |

### Cấu trúc thư mục
```
.
├── BLUEPRINT.md, AGENTS.md, docker-compose.yml, dvc.yaml, dvc.lock, .env.example
├── docs/                 # 16 file markdown (blueprint + week1 docs)
├── shared/               # constants.py (84 lòng), schemas.py (86 dòng)
├── training/
│   ├── src/              # bronze.py (505), silver.py (285), config.py (104)
│   └── tests/            # 3 test file (~770 dòng)
├── infra/minio/          # init-bucket.sh + README
├── notebooks/            # 3 notebook EDA
├── notebook-planned/     # 2 notebook trống (placeholder)
└── .opencode/            # skill packs (đã bị ignore nhưng vẫn commit)
```

---

## 2. Điểm mạnh

### 2.1 Thiết kế & tài liệu
- **Blueprint chi tiết, có chiều sâu**: 12 file phân chia rõ overview / architecture / features / pipelines / project structure / error handling / testing / security / explainability / performance / demo / roadmap.
- **Contract rõ ràng và locked**: timestamp (`event_time` ở raw vs `source_event_time` ở bronze trở đi), canonical `event_id`, dedup keys, prediction horizon 10 phút, fail-closed validation gate, online evaluation tách `demo_replay` vs `offline_backfill`.
- **AGENTS.md trung thực**: phân biệt rõ "target-state design" và "what is real today" — giúp người mới (và agent) không bị nhầm blueprint là code.
- **One data lake, multiple usage windows**: tách rõ training / replay / retraining như các "view" khác nhau trên cùng raw pool — đây là design đúng đắn.

### 2.2 Code engineering
- **Bronze pipeline chunked + streaming** (`bronze.py:284-398`): dùng `pd.read_csv(chunksize=...)` + `pyarrow.parquet.ParquetWriter` lazy-init + `gc.collect()` + memory telemetry qua `psutil`. Đây là **kỹ thuật đúng** cho file 1.7 GB CSV — chứng tỏ tác giả đã thực sự đo và xử lý OOM.
- **PyArrow schema-strict write**: `pa.Table.from_pandas(chunk_bronze, schema=BRONZE_SCHEMA, preserve_index=False)` đảm bảo schema enforcement ngay tại write.
- **Layer-aware constants**: `shared/constants.py` định nghĩa rõ `LAYER_RAW/BRONZE/SILVER/GOLD`, `ALLOWED_EVENT_TYPES`, `REQUIRED_FIELDS`, `DEDUP_KEY_FIELDS` — dễ tái sử dụng và test.
- **Cấu hình qua env var với defaults hợp lý** (`training/src/config.py`).
- **DVC-based reproducibility**: `dvc.lock` đã có MD5 hash + size — pipeline đã chạy thật, không phải scaffold giả.
- **Bảo vệ baseline khỏi lẫn dữ liệu simulation** (`bronze.py:120-139`, `ensure_not_simulation_raw_input`).

---

## 3. Vấn đề & Bugs (theo mức độ ưu tiên)

### 🔴 P0 — Bugs gây failure ngay

#### 3.1 `Config.get_all_settings()` raise `AttributeError`
`training/src/config.py:88-94` truy cập `cls.DATA_WINDOW_PROFILE`, `cls.TRAINING_WINDOW_START`, `cls.TRAINING_WINDOW_END`, `cls.DEV_SMOKE_WINDOW_START`, `cls.DEV_SMOKE_WINDOW_END`, `cls.REPLAY_WINDOW_START`, `cls.REPLAY_WINDOW_END` — **không có thuộc tính nào trong số này được khai báo trong class `Config`**. Verified bằng `python -c "Config.get_all_settings()"` → `AttributeError: type object 'Config' has no attribute 'DATA_WINDOW_PROFILE'`.

`.env.example` (lines 36-43) cũng liệt kê 7 biến này, nhưng chúng không được map vào `Config`. Đây là **contract drift giữa env contract và Python code**.

#### 3.2 Test suite hỏng — 10 fail + 1 collection error
Chạy `pytest training/tests/`:
- `test_silver_dataset_io.py`: **ImportError ngay lúc collect** vì `training.src.silver` không có `get_silver_sort_columns`, `normalize_category_code`, `validate_silver_sort_columns` (test import 5 hàm, code chỉ có 2).
- `test_raw_window_selection.py`: **9 / 9 fail** vì `training.src.bronze` không có `select_raw_files`, `resolve_raw_window_bounds`, `extract_raw_file_month`.
- `test_data_lake.py::TestDataPathConfig::test_data_strategy_config_defaults` fail do bug 3.1.

→ Test reference các hàm theo blueprint mới (window selection, dataset directory I/O, normalize_category_code) nhưng **các hàm này chưa được merge vào `main`**. Branch `dev` có thể đã có; merge bị thiếu hoặc bị revert (lịch sử commit có `Revert "fix: sửa lỗi OOM"` và `tmp` — gợi ý merge không hoàn chỉnh).

#### 3.3 Contract drift: bronze chưa filter theo window
`docs/RAW_DATA_INTAKE.md` và `docs/BLUEPRINT/12_ROADMAP.md` ghi:
> Bronze ingestion only considers files named `YYYY-Mon.csv` or `YYYY-Mon.csv.gz`; unsupported names are skipped. Bronze selects files by `--window-profile` (training, replay, dev_smoke, or all).

Nhưng `bronze.discover_raw_files()` (`bronze.py:90-117`) đọc **mọi** file `*.csv` / `*.csv.gz` không kiểm tra naming convention, không filter theo window. CLI cũng không có flag `--window-profile`. → Khi data lake có nhiều file tháng, bronze sẽ ăn hết tất cả → vi phạm contract "training window = Oct 2019 → Feb 2020".

### 🟠 P1 — Risk / smell nghiêm trọng

#### 3.4 Silver load toàn bộ bronze vào RAM
`silver.read_bronze_parquet` (`silver.py:47-67`) gọi `pq.read_table(bronze_path).to_pandas()`. Bronze artifact hiện tại = **1.18 GB**. Với `dev_smoke` (1 tháng) chấp nhận được, nhưng với full window (5 tháng) → xấp xỉ 6 GB RAM cho riêng silver. Trái với design intent "memory-safe dataset processing" trong blueprint.

→ Nên chuyển sang `pyarrow.dataset.dataset(bronze_dir).to_batches()` hoặc `pq.ParquetFile(...).iter_batches(batch_size=...)`.

#### 3.5 Không có dependency manifest
- Không có `requirements.txt`, `pyproject.toml`, `environment.yml`, `setup.py`.
- WEEK1_SETUP.md chỉ ghi `pip install pandas pyarrow dvc python-dotenv`, không có version pin.
- Không có cách reproduce môi trường ngoài "conda activate MLOPS" (env tồn tại trên máy tác giả).
- `dvc.lock` có MD5 hash → không reproduce được nếu lib version đổi (PyArrow schema casting nhạy theo version).

#### 3.6 Không có top-level `README.md`
Người clone repo vào sẽ thấy `BLUEPRINT.md` đầu tiên (200+ dòng tổng quan) nhưng không có quick-start. Người mới phải đọc `AGENTS.md` + `docs/WEEK1_SETUP.md` để biết cách chạy.

#### 3.7 Không có CI / pre-commit / lint
- Blueprint roadmap (Week 6) hứa hẹn GitHub Actions với `pytest-cov ≥ 70%`, nhưng `.github/` không tồn tại.
- Không có `.pre-commit-config.yaml`, `ruff.toml`, `pyproject.toml [tool.ruff]`, `mypy.ini`.
- → Test fail không bị block; merge từ `dev` vào `main` đã đẩy code broken lên `main` mà không ai phát hiện.

#### 3.8 `sys.path.insert(0, ...)` hack
`bronze.py:33-35`, `silver.py:21-23`, `tests/*.py` đều có `sys.path.insert(0, str(Path(__file__).parent.parent.parent))`. Đây là dấu hiệu repo chưa được install như package. Giải pháp đúng: thêm `pyproject.toml` + `pip install -e .`.

### 🟡 P2 — Code smell / nice-to-fix

#### 3.9 Backward-compat shims không cần thiết
`bronze.py:210-215, 276-281, 401-414` có 3 hàm `parse_event_time`, `transform_to_bronze`, `write_bronze_parquet` chỉ là wrapper gọi version `_chunk` "for tests and standalone script calls". Nên migrate test sang dùng API chính thức và xóa shim.

#### 3.10 `silver.check_required_fields` mutate `REQUIRED_FIELDS`
`silver.py:108-110` làm `required_fields.copy()` rồi `discard(FIELD_EVENT_TIME)` + `add(FIELD_SOURCE_EVENT_TIME)`. Hợp lý ở behavior nhưng gợi ý: `constants.REQUIRED_FIELDS` không phải single source — nên có `BRONZE_REQUIRED_FIELDS` và `SILVER_REQUIRED_FIELDS` riêng.

#### 3.11 DVC endpoint không khớp giữa host vs container
`.dvc/config` ghi `endpointurl = http://localhost:9000` (cho host) còn `docker-compose.yml` lại dùng tên service `http://minio:9000` (cho container). Nếu sau này có service chạy DVC trong container (training-runner) sẽ không kết nối được. Đề xuất: dùng env var `DVC_ENDPOINT_URL` hoặc 2 remote profile.

#### 3.12 `.opencode/` vừa ignore vừa commit
`.gitignore:3` có `.opencode/` nhưng thư mục `.opencode/skills/...` đã được tracked (~30+ file). Hoặc xóa khỏi `.gitignore` (nếu thực sự muốn commit) hoặc `git rm -r --cached .opencode/`.

#### 3.13 `notebook-planned/` chỉ có placeholder
2 file (`02_feature_experiment.ipynb`, `03_model_experiment.ipynb`) — nên có README giải thích hoặc xóa khỏi main.

#### 3.14 Logging f-string không có placeholder
`silver.py:265, 278`: `logger.info(f"\n6. Writing silver artifact...")` — f-string vô nghĩa. Ruff `F541` sẽ flag.

#### 3.15 Lịch sử commit không sạch
- `tmp` (commit `fe29ad6`)
- `Revert "fix: sửa lỗi OOM"` (commit `b11244c`) — tại sao revert một fix OOM mà không có giải pháp thay thế?
- `merge: resolve conflicts between dev and main branches` (commit `09544b0`) — gợi ý conflict resolution có thể đã làm mất code (xem 3.2).

→ Khuyến nghị: dùng `git rebase -i` clean lịch sử trước khi merge dev→main, và viết PR description giải thích revert.

### 🔵 P3 — Security / production concerns

#### 3.16 Default credentials hardcode
`config.py:46-47`: `AWS_ACCESS_KEY_ID = "minioadmin"`, `AWS_SECRET_ACCESS_KEY = "minioadmin"`. Cho local OK, nhưng `Config.get_all_settings()` (nếu sửa) cũng sẽ log secret vào log → cần redact hoặc dùng `pydantic-settings` với `SecretStr`.

#### 3.17 Không có data validation ngoài event_type
- Bronze chỉ validate event_type ∈ {view, cart, ...}.
- Không validate range của price, format UUID của user_session, format timestamp.
- Blueprint mention "fail-closed validation gate" nhưng chưa có Great Expectations / pandera schema runtime check.

---

## 4. Đối chiếu giữa code thực và blueprint

| Hứa hẹn trong blueprint | Trạng thái thực tế |
|---|---|
| Bronze chunked + memory-safe | ✅ Đúng |
| Bronze partitioned dataset (`year=2019/month=10/`) | ❌ Hiện chỉ output 1 file `events.parquet` |
| Silver dataset directory I/O | ❌ `silver.py` write 1 file đơn |
| Window selection (`--window-profile`) | ❌ Test có, code không có |
| Session split / gold layer | ⏳ Đúng roadmap, chưa làm |
| Training / model registry / SHAP | ⏳ Đúng roadmap, chưa làm |
| FastAPI serving | ⏳ Đúng roadmap, chưa làm |
| Stream processing (Quix + Kafka + Redis) | ⏳ Đúng roadmap, chưa làm |
| Monitoring (Prometheus + Grafana) | ⏳ Đúng roadmap, chưa làm |
| CI (GitHub Actions, pytest-cov ≥70%) | ❌ Chưa làm, test còn đang đỏ |
| Pre-commit hooks | ❌ Không có |

---

## 5. Đề xuất cải thiện (theo thứ tự ưu tiên)

### Ngay (P0) — fix lỗi đã có
1. **Sửa `Config`**: thêm các thuộc tính `DATA_WINDOW_PROFILE`, `TRAINING_WINDOW_START/END`, `DEV_SMOKE_WINDOW_START/END`, `REPLAY_WINDOW_START/END` đọc từ env. Hoặc nếu chưa cần, xóa khỏi `get_all_settings()` để khỏi crash.
2. **Đưa branch `dev` vào main đầy đủ**: code window-selection (`select_raw_files`, `resolve_raw_window_bounds`, `extract_raw_file_month`) và silver dataset I/O (`get_silver_sort_columns`, `normalize_category_code`, `validate_silver_sort_columns`) — test đang reference nhưng main thiếu. Diff `git log dev..main -- training/src/` để xác minh.
3. **Chạy `pytest` cho đến khi xanh** trước khi merge sang main.

### Tuần này (P1) — chặn regression
4. Thêm `pyproject.toml` với deps + dev-deps (pandas, pyarrow, pytest, ruff, mypy, dvc[s3]) và pin version. Cho phép `pip install -e .[dev]`.
5. Xóa toàn bộ `sys.path.insert(...)` sau khi cài package được.
6. Thêm `.pre-commit-config.yaml` với ruff + ruff-format + check-yaml + check-added-large-files.
7. Thêm `.github/workflows/ci.yml`: lint + pytest + dvc dag check.
8. Thêm `README.md` top-level: 30 dòng quick-start (clone → conda env → docker-compose up → dvc repro → pytest).
9. Convert silver sang dataset-aware streaming (`pq.ParquetFile.iter_batches`) để align với blueprint memory-safe.
10. Implement `bronze.select_raw_files` đúng spec window-profile để fix contract drift.

### Trước Week 2 (P2) — chuẩn hóa nền tảng
11. Tách `BRONZE_REQUIRED_FIELDS` / `SILVER_REQUIRED_FIELDS` thay vì mutate `REQUIRED_FIELDS`.
12. Xóa backward-compat shim sau khi test migrate xong.
13. Bronze write theo dataset directory `data/bronze/year=YYYY/month=MM/part-*.parquet` để chuẩn bị cho gold/session-split.
14. Đưa runtime data validation (pandera schema) vào bronze write path.
15. Clean `.opencode/` khỏi git index hoặc khỏi `.gitignore`.
16. Setup `pydantic-settings` cho `Config` thay vì `os.getenv`.

### Trong Q2 (P3) — production-readiness
17. DVC remote 2 profile (host vs container) qua env var.
18. Secret management: không hardcode `minioadmin`; dùng `.env` + git-secret hoặc Vault.
19. Lịch sử git: rebase / squash các commit `tmp`, revert chains trước khi merge.

---

## 6. Đánh giá tổng thể

| Tiêu chí | Điểm | Ghi chú |
|---|---|---|
| **Thiết kế / blueprint** | 8.5/10 | Tài liệu xuất sắc, contract rõ ràng |
| **Code quality (phần đã viết)** | 7/10 | Chunked bronze tốt, silver còn nguyên dạng load-all |
| **Tests** | 3/10 | 10/30 fail trên main, không có CI chặn |
| **Reproducibility** | 5/10 | Có DVC nhưng thiếu deps manifest |
| **Mức hoàn thiện theo roadmap** | 1/7 tuần (~14%) | Đúng kế hoạch, không over-promise |
| **Onboarding (README/setup)** | 4/10 | Thiếu top-level README, deps |
| **Production readiness** | 2/10 | Chưa có serving / monitoring / CI / security |

### Tóm tắt
Repo có **nền tảng kiến trúc rất tốt** và **bronze pipeline được engineer cẩn thận** (chunked, memory-aware, schema-strict). Đây là điểm sáng. **Vấn đề lớn nhất hiện tại là main branch đang ở trạng thái broken — tests fail, code thiếu hàm mà tests reference, contract giữa docs / env / code chưa khớp.** Cần fix gấp 3 vấn đề P0 trước khi tiếp tục Week 2; sau đó setup CI + deps manifest để phòng regression. Nếu duy trì chất lượng engineering như bronze pipeline cho các layer còn lại, đây sẽ là một dự án MLOps đáng tham khảo.
