# GPU-Aware Training Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make both smoke mode and the main training mode use the same GPU-first device policy.

**Architecture:** Keep `--smoke-mode` limited to Optuna budget selection. Add a shared device policy that is threaded into CatBoost, LightGBM, and XGBoost so the smoke and full runs behave the same way. Default to GPU-first execution, with an explicit CPU override for debugging and a small fallback path only for `auto` runs.

**Tech Stack:** Python, CatBoost, LightGBM, XGBoost, Optuna, MLflow, pytest, Ruff.

---

### Task 1: Add shared device configuration and CLI flags

**Files:**
- Modify: `training/src/config.py`
- Modify: `training/src/train.py`
- Test: `training/tests/test_train.py`

- [ ] **Step 1: Write the failing test**

```python
def test_main_accepts_device_flags(gold_data, monkeypatch):
    fake_mlflow = _FakeMlflow()
    monkeypatch.setattr("training.src.train.mlflow", fake_mlflow)
    monkeypatch.setattr("training.src.train.OPTUNA_SMOKE_TRIALS", 1)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "training.src.train",
            "--train",
            gold_data["train_path"],
            "--val",
            gold_data["val_path"],
            "--test",
            gold_data["test_path"],
            "--session-split-map",
            gold_data["split_map_path"],
            "--smoke-mode",
            "--device",
            "cpu",
            "--gpu-device-id",
            "0",
        ],
    )

    assert main() == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest training/tests/test_train.py::test_main_accepts_device_flags -v`
Expected: fail because `--device` and `--gpu-device-id` are not wired yet.

- [ ] **Step 3: Write minimal implementation**

```python
# training/src/config.py
TRAIN_DEVICE = os.getenv("TRAIN_DEVICE", "gpu")
GPU_DEVICE_ID = os.getenv("GPU_DEVICE_ID", "0")
```

```python
# training/src/train.py
parser.add_argument(
    "--device",
    default=Config.TRAIN_DEVICE,
    choices=["auto", "cpu", "gpu"],
)
parser.add_argument("--gpu-device-id", default=Config.GPU_DEVICE_ID)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest training/tests/test_train.py::test_main_accepts_device_flags -v`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add training/src/config.py training/src/train.py training/tests/test_train.py
git commit -m "feat: add gpu device configuration"
```

### Task 2: Thread the device policy into all three model trainers

**Files:**
- Modify: `training/src/train.py`
- Test: `training/tests/test_train.py`

- [ ] **Step 1: Write the failing test**

```python
def test_gpu_params_are_set_for_all_candidates():
    trial = optuna.trial.FixedTrial(
        {
            "iterations": 100,
            "depth": 4,
            "learning_rate": 0.1,
            "l2_leaf_reg": 1.0,
            "scale_pos_weight": 1.0,
            "max_depth": 4,
            "num_leaves": 15,
            "n_estimators": 50,
        }
    )

    cat_params = _catboost_params(trial, device="gpu", gpu_device_id="0")
    lgb_params = _lightgbm_params(trial, device="gpu", gpu_device_id="0")
    xgb_params = _xgboost_params(trial, device="gpu", gpu_device_id="0")

    assert cat_params["task_type"] == "GPU"
    assert cat_params["devices"] == "0"
    assert lgb_params["device_type"] == "gpu"
    assert xgb_params["device"] == "cuda:0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest training/tests/test_train.py::test_gpu_params_are_set_for_all_candidates -v`
Expected: fail because the helper functions do not accept device selection yet.

- [ ] **Step 3: Write minimal implementation**

```python
def _catboost_params(trial: optuna.Trial, device: str, gpu_device_id: str) -> dict:
    params = {
        "iterations": trial.suggest_int("iterations", 100, 300),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 5.0),
        "loss_function": "Logloss",
        "random_seed": 42,
        "allow_writing_files": False,
        "verbose": False,
    }
    if device in {"auto", "gpu"}:
        params["task_type"] = "GPU"
        params["devices"] = gpu_device_id
    return params
```

```python
def _lightgbm_params(trial: optuna.Trial, device: str, gpu_device_id: str) -> dict:
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "random_state": 42,
        "n_jobs": -1,
    }
    if device in {"auto", "gpu"}:
        params["device_type"] = "gpu"
        params["gpu_device_id"] = int(gpu_device_id)
    return params
```

```python
def _xgboost_params(trial: optuna.Trial, device: str, gpu_device_id: str) -> dict:
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 5.0),
        "random_state": 42,
        "tree_method": "hist",
        "enable_categorical": True,
        "eval_metric": "aucpr",
        "n_jobs": -1,
    }
    if device in {"auto", "gpu"}:
        params["device"] = f"cuda:{gpu_device_id}"
    else:
        params["device"] = "cpu"
    return params
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest training/tests/test_train.py::test_gpu_params_are_set_for_all_candidates -v`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add training/src/train.py training/tests/test_train.py
git commit -m "feat: thread gpu params through all candidates"
```

### Task 3: Make smoke and full runs share the same device policy

**Files:**
- Modify: `training/src/train.py`
- Test: `training/tests/test_train.py`

- [ ] **Step 1: Write the failing test**

```python
def test_smoke_and_full_runs_pass_same_device_policy(gold_data, monkeypatch):
    fake_mlflow = _FakeMlflow()
    monkeypatch.setattr("training.src.train.mlflow", fake_mlflow)
    monkeypatch.setattr("training.src.train.OPTUNA_SMOKE_TRIALS", 1)
    monkeypatch.setattr("training.src.train.OPTUNA_TARGET_TRIALS", 1)

    recorded = []

    def fake_train(*args, **kwargs):
        recorded.append(kwargs["device"])
        return _FakeModel(), {"pr_auc": 0.8, "confusion_matrix": np.array([[1, 0], [0, 1]])}

    monkeypatch.setattr("training.src.train.train_catboost_candidate", fake_train)
    monkeypatch.setattr("training.src.train.train_lightgbm_candidate", fake_train)
    monkeypatch.setattr("training.src.train.train_xgboost_candidate", fake_train)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "training.src.train",
            "--train",
            gold_data["train_path"],
            "--val",
            gold_data["val_path"],
            "--test",
            gold_data["test_path"],
            "--session-split-map",
            gold_data["split_map_path"],
            "--smoke-mode",
            "--device",
            "gpu",
            "--gpu-device-id",
            "0",
        ],
    )
    assert main() == 0
    assert recorded == ["gpu", "gpu", "gpu"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest training/tests/test_train.py::test_smoke_and_full_runs_pass_same_device_policy -v`
Expected: fail because the device policy is not yet threaded through `main`.

- [ ] **Step 3: Write minimal implementation**

```python
device = args.device
gpu_device_id = args.gpu_device_id

catboost_model, catboost_metrics = train_catboost_candidate(
    ...,
    n_trials,
    device=device,
    gpu_device_id=gpu_device_id,
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest training/tests/test_train.py::test_smoke_and_full_runs_pass_same_device_policy -v`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add training/src/train.py training/tests/test_train.py
git commit -m "feat: thread gpu policy through training modes"
```

### Task 4: Add GPU fallback behavior for `auto`

**Files:**
- Modify: `training/src/train.py`
- Test: `training/tests/test_train.py`

- [ ] **Step 1: Write the failing test**

```python
def test_auto_device_falls_back_to_cpu_when_gpu_fails(monkeypatch, gold_data):
    fake_mlflow = _FakeMlflow()
    monkeypatch.setattr("training.src.train.mlflow", fake_mlflow)
    monkeypatch.setattr("training.src.train.OPTUNA_SMOKE_TRIALS", 1)

    calls = {"cat": 0}

    def fake_train(*args, **kwargs):
        calls["cat"] += 1
        if calls["cat"] == 1:
            raise RuntimeError("GPU unavailable")
        return _FakeModel(), {"pr_auc": 0.8, "confusion_matrix": np.array([[1, 0], [0, 1]])}

    monkeypatch.setattr("training.src.train.train_catboost_candidate", fake_train)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "training.src.train",
            "--train",
            gold_data["train_path"],
            "--val",
            gold_data["val_path"],
            "--test",
            gold_data["test_path"],
            "--session-split-map",
            gold_data["split_map_path"],
            "--smoke-mode",
            "--device",
            "auto",
        ],
    )

    assert main() == 0
    assert calls["cat"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest training/tests/test_train.py::test_auto_device_falls_back_to_cpu_when_gpu_fails -v`
Expected: fail because `auto` fallback is not implemented yet.

- [ ] **Step 3: Write minimal implementation**

```python
def _train_with_device_policy(train_fn, *, device: str, ...):
    try:
        return train_fn(..., device=device, ...)
    except Exception:
        if device != "auto":
            raise
        logger.warning("GPU training failed, retrying on CPU")
        return train_fn(..., device="cpu", ...)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest training/tests/test_train.py::test_auto_device_falls_back_to_cpu_when_gpu_fails -v`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add training/src/train.py training/tests/test_train.py
git commit -m "feat: add auto gpu fallback"
```

### Task 5: Verify the updated training path end to end

**Files:**
- Modify: none
- Test: `training/tests/test_train.py`

- [ ] **Step 1: Run the focused test file**

Run: `pytest training/tests/test_train.py -q`
Expected: all training tests pass.

- [ ] **Step 2: Run static checks on touched files**

Run: `ruff check training/src/train.py training/src/config.py training/tests/test_train.py`
Expected: no lint errors.

- [ ] **Step 3: Run a manual smoke command with the device policy**

Run:
```bash
conda run -n MLOPS python -m training.src.train \
  --train data/gold/train.parquet \
  --val data/gold/val.parquet \
  --test data/gold/test.parquet \
  --session-split-map data/gold/session_split_map.parquet \
  --smoke-mode \
  --device gpu
```

Expected: smoke mode and the main mode share the same GPU-first policy; the command should use GPU when the runtime has a working NVIDIA setup.

**Assumptions:**
- `--smoke-mode` remains only a budget switch.
- GPU usage is a training-policy concern, not a smoke-mode concern.
- LightGBM GPU support depends on the installed build, so the code will request GPU there but the runtime still needs a GPU-capable wheel/build.
- The repo default should prefer GPU, with `auto` available only for fallback behavior when needed.
