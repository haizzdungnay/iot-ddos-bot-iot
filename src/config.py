"""
File cấu hình cho dự án phát hiện DDoS với LSTM
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class DataConfig:
    """Cấu hình dữ liệu"""
    # Đường dẫn
    raw_data_path: str = "data/raw/bot_iot.csv"
    processed_data_path: str = "data/processed/bot_iot_preprocessed.csv"

    # Preprocessing
    binary_classification: bool = True  # True: Normal vs Attack, False: Multi-class
    test_size: float = 0.2
    validation_split: float = 0.1
    random_state: int = 42

    # Cột nhãn trong Bot-IoT
    label_column: str = "attack"  # hoặc "category" tùy dataset


@dataclass
class ModelConfig:
    """Cấu hình mô hình LSTM"""
    # LSTM architecture
    time_steps: int = 1  # Số time steps (1 cho stateless LSTM)
    lstm_units: int = 64  # Số units trong LSTM layer
    dropout_rate: float = 0.3
    dense_units: int = 32

    # Training
    learning_rate: float = 1e-3
    batch_size: int = 256
    epochs: int = 50
    early_stopping_patience: int = 10

    # Class imbalance handling
    use_class_weight: bool = True  # Tự động tính class weight
    use_smote: bool = False  # Sử dụng SMOTE để balance data (chỉ cho training)

    # Model saving
    model_dir: str = "models"
    model_name: str = "lstm_ddos_model.h5"
    save_best_only: bool = True


@dataclass
class TrainingConfig:
    """Cấu hình tổng thể cho training"""
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()

    # Paths
    log_dir: str = "logs"
    results_dir: str = "results"

    # Experiment tracking
    experiment_name: Optional[str] = None
    save_metrics: bool = True
    plot_results: bool = True

    def __post_init__(self):
        """Tạo thư mục nếu chưa tồn tại"""
        os.makedirs(self.model.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)


# Default config
DEFAULT_CONFIG = TrainingConfig()


# Các config thử nghiệm khác
@dataclass
class LightweightConfig(TrainingConfig):
    """Config cho model nhẹ, training nhanh"""
    def __post_init__(self):
        super().__post_init__()
        self.model.lstm_units = 32
        self.model.dense_units = 16
        self.model.batch_size = 512
        self.model.epochs = 20


@dataclass
class DeepConfig(TrainingConfig):
    """Config cho model sâu hơn (có thể cần nhiều data)"""
    def __post_init__(self):
        super().__post_init__()
        self.model.lstm_units = 128
        self.model.dense_units = 64
        self.model.dropout_rate = 0.4
        self.model.epochs = 100
        self.model.early_stopping_patience = 15


@dataclass
class SequenceConfig(TrainingConfig):
    """Config cho LSTM với sequence (time_steps > 1)"""
    def __post_init__(self):
        super().__post_init__()
        self.model.time_steps = 10  # Sử dụng 10 flows liên tiếp
        self.model.lstm_units = 128
        self.model.epochs = 100


def get_config(config_name: str = "default") -> TrainingConfig:
    """
    Lấy config theo tên

    Args:
        config_name: "default", "lightweight", "deep", hoặc "sequence"

    Returns:
        TrainingConfig object
    """
    configs = {
        "default": TrainingConfig(),
        "lightweight": LightweightConfig(),
        "deep": DeepConfig(),
        "sequence": SequenceConfig()
    }

    if config_name not in configs:
        print(f"Warning: Config '{config_name}' không tồn tại. Dùng 'default'.")
        return configs["default"]

    return configs[config_name]
