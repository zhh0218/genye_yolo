import os
import warnings
from pathlib import Path

# 模型配置文件
#model_yaml_path = r"F:\\1YOLO\\YOLOv11RGBD\\ultralytics-main\\ultralytics\\cfg\\models\\11\\yolo11-seg.yaml"/workspace/ultralytics-main/ultralytics/cfg/models/11/yolo11-seg.yaml
default_model_yaml_path = r"/workspace/ultralytics-main_for_genye/ultralytics/cfg/models/11/yolo11-seg.yaml"
model_yaml_path = os.getenv('MODEL_YAML_PATH', default_model_yaml_path).strip() or default_model_yaml_path
# 数据集配置文件
# data_yaml_path = r'/workspace/ultralytics-main/data-ubuntu-selective-os01.yaml'
default_data_yaml_path = r'/workspace/ultralytics-main_for_genye/data-ubuntu-genye.yaml'
data_yaml_path = os.getenv('DATA_YAML_PATH', default_data_yaml_path).strip() or default_data_yaml_path
# data_yaml_path = r'/workspace/ultralytics-main/data-ubuntu.yaml'
# data_yaml_path = r'F:\\1YOLO\\YOLOv11RGBD\\ultralytics-main\\data.yaml'
# 预训练模型
default_pre_model_name = r'/workspace/ultralytics-main_for_genye/yolo11l.pt'
pre_model_name = os.getenv('PRE_MODEL_NAME', default_pre_model_name).strip() or default_pre_model_name
#pre_model_name = r'/workspace/ultralytics-main/runs/segment/train56/weights/best.pt'
warnings.filterwarnings('ignore')
from ultralytics import YOLO
from ultralytics.utils import LOGGER, SETTINGS, YAML


def setup_wandb(local_project, local_name, train_seed=None):
    use_wandb = os.getenv('USE_WANDB', '').strip().lower() in {'1', 'true', 'yes'}
    if not use_wandb:
        return None

    try:
        import wandb
        api_key = os.getenv('WANDB_API_KEY', '').strip()
        if api_key:
            wandb.login(key=api_key, relogin=True)

        wandb_project = os.getenv('WANDB_PROJECT', 'rgbd-yolo11-ablation').strip() or 'rgbd-yolo11-ablation'
        wandb_name = os.getenv('WANDB_RUN_NAME', local_name).strip() or local_name
        wandb_tags = [x.strip() for x in os.getenv('WANDB_TAGS', '').split(',') if x.strip()]

        if wandb.run is None:
            wandb.init(
                project=wandb_project,
                name=wandb_name,
                tags=wandb_tags or None,
                config={
                    'model_yaml_path': model_yaml_path,
                    'data_yaml_path': data_yaml_path,
                    'pre_model_name': pre_model_name,
                    'train_seed': train_seed,
                    'local_project': local_project,
                    'local_name': local_name,
                },
            )
    except Exception as e:
        SETTINGS['wandb'] = False
        message = f"W&B disabled because setup failed: {e}"
        if 'protobuf' in str(e).lower():
            message += " Please align the installed protobuf version with the wandb package version."
        LOGGER.warning(message)
        return None

    SETTINGS['wandb'] = True
    return wandb


def _env_flag(name):
    value = os.getenv(name, '').strip().lower()
    if not value:
        return None
    return value in {'1', 'true', 'yes', 'on'}


def _env_int(name, default):
    value = os.getenv(name, '').strip()
    return int(value) if value else default


def _env_float(name, default):
    value = os.getenv(name, '').strip()
    return float(value) if value else default


def _env_str(name, default):
    value = os.getenv(name, '').strip()
    return value if value else default


def build_runtime_model_yaml(base_path, run_name):
    str_overrides = {
        'RGBD_FUSION': 'rgbd_fusion',
    }
    bool_overrides = {
        'P3_WAVELET_GUIDED': 'p3_wavelet_guided',
        'P3_WAVELET_HF': 'p3_wavelet_HF',
        'P3_WAVELET_ADAPTIVE': 'p3_wavelet_adaptive',
        'P4_WAVELET_GUIDED': 'p4_wavelet_guided',
        'DEPTH_FPN': 'depth_fpn',
        'MODALITY_ADAPTIVE_GATE': 'modality_adaptive_gate',
        'RGBD_PMG_AUX_LOSS': 'rgbd_pmg_aux_loss',
        'PROG_LOSS': 'prog_loss',
        'STAL': 'stal',
    }
    float_overrides = {
        'RGBD_PMG_AUX_LOSS_WEIGHT': 'rgbd_pmg_aux_loss_weight',
        'RGBD_PMG_AUX_POS_WEIGHT': 'rgbd_pmg_aux_pos_weight',
        'COF_EMBED_RATIO': 'cof_embed_ratio',
        'COF_PMG_ALPHA_INIT': 'cof_pmg_alpha_init',
    }
    int_overrides = {
        'COF_MIN_EMBED': 'cof_min_embed',
    }

    runtime_overrides = {}
    for env_name, yaml_key in str_overrides.items():
        value = os.getenv(env_name, '').strip()
        if value:
            runtime_overrides[yaml_key] = value
    for env_name, yaml_key in bool_overrides.items():
        value = _env_flag(env_name)
        if value is not None:
            runtime_overrides[yaml_key] = value
    for env_name, yaml_key in float_overrides.items():
        value = os.getenv(env_name, '').strip()
        if value:
            runtime_overrides[yaml_key] = float(value)
    for env_name, yaml_key in int_overrides.items():
        value = os.getenv(env_name, '').strip()
        if value:
            runtime_overrides[yaml_key] = int(value)

    if not runtime_overrides:
        return base_path

    model_cfg = YAML.load(base_path)
    model_cfg.update(runtime_overrides)
    runtime_dir = Path('LOG') / 'runtime_cfg'
    runtime_dir.mkdir(parents=True, exist_ok=True)
    safe_run_name = run_name.replace('/', '_').replace(' ', '_')
    runtime_path = runtime_dir / f'{safe_run_name}.yaml'
    YAML.save(runtime_path, model_cfg)
    return str(runtime_path)

if __name__ == '__main__':
    local_project_default = 'runs_yolo11n/YOLOv11-RGB-D-wavelet_ablation'
    local_name_default = 'coordatt-nodfl-p3hf'
    local_project = os.getenv('LOCAL_PROJECT', local_project_default).strip() or local_project_default
    local_name = os.getenv('LOCAL_NAME', local_name_default).strip() or local_name_default
    train_seed_env = os.getenv('TRAIN_SEED', '').strip()
    train_seed = int(train_seed_env) if train_seed_env else None
    if train_seed is not None:
        local_name = f'{local_name}-seed{train_seed}'
    model_yaml_path = build_runtime_model_yaml(model_yaml_path, local_name)
    wandb_run = setup_wandb(local_project, local_name, train_seed)
    model = YOLO(model_yaml_path)
    model.load(pre_model_name)
    amp_env = _env_flag('AMP')
    train_kwargs = dict(data=data_yaml_path,
                        imgsz=_env_int('IMGSZ', 640),
                        epochs=_env_int('EPOCHS', 300),
                        patience=_env_int('PATIENCE', 100),
                        batch=_env_int('BATCH', 32),
                        workers=_env_int('WORKERS', 2),
                        device=_env_str('DEVICE', '0,1,2,3'),
                        optimizer=_env_str('OPTIMIZER', 'MuSGD'),
                        lr0=_env_float('LR0', 0.01),
                        lrf=_env_float('LRF', 0.001),
                        weight_decay=_env_float('WEIGHT_DECAY', 0.0001),
                        amp=True if amp_env is None else amp_env,
                        project=local_project,
                        #name='MuSGD',
                        name=local_name)
    if train_seed is not None:
        train_kwargs['seed'] = train_seed
        train_kwargs['deterministic'] = True
    model.train(**train_kwargs)
    if wandb_run is not None:
        wandb_run.finish()
