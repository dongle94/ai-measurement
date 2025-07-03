import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigError(Exception):
    """설정 관련 예외"""
    pass


class Config:
    """설정 관리 클래스"""
    
    def __init__(self):
        self._config_data = {}
        self._is_loaded = False
    
    def load_from_file(self, config_file: str) -> None:
        """YAML 파일에서 설정을 로드합니다."""
        try:
            config_path = Path(config_file)
            if not config_path.exists():
                raise ConfigError(f"설정 파일을 찾을 수 없습니다: {config_file}")
            
            with open(config_path, 'r', encoding='utf-8') as f:
                self._config_data = yaml.safe_load(f)
            
            if not self._config_data:
                raise ConfigError("설정 파일이 비어있습니다.")
            
            self._validate_config()
            self._is_loaded = True
            
        except yaml.YAMLError as e:
            raise ConfigError(f"YAML 파싱 오류: {e}")
        except Exception as e:
            raise ConfigError(f"설정 파일 로드 오류: {e}")
    
    def _validate_config(self) -> None:
        """필수 설정 키들을 검증합니다."""
        # 유연한 검증: 필수 섹션이 있는지만 확인
        required_sections = ['ENV', 'LOG']
        
        for section in required_sections:
            if section not in self._config_data:
                raise ConfigError(f"필수 섹션이 없습니다: {section}")
        
        # ENV 섹션 필수 키 검증
        if 'DEVICE' not in self._config_data.get('ENV', {}):
            raise ConfigError("ENV.DEVICE 키가 필요합니다.")
        
        # LOG 섹션 필수 키 검증
        log_section = self._config_data.get('LOG', {})
        required_log_keys = ['LOG_LEVEL', 'LOGGER_NAME']
        for key in required_log_keys:
            if key not in log_section:
                raise ConfigError(f"LOG.{key} 키가 필요합니다.")
    
    def get(self, key: str, default: Any = None) -> Any:
        """점 표기법으로 설정값을 가져옵니다. (예: 'ENV.DEVICE')"""
        if not self._is_loaded:
            raise ConfigError("설정이 로드되지 않았습니다. load_from_file()을 먼저 호출하세요.")
        
        keys = key.split('.')
        value = self._config_data
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            if default is not None:
                return default
            raise ConfigError(f"설정 키를 찾을 수 없습니다: {key}")
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """전체 섹션을 반환합니다."""
        return self.get(section, {})
    
    def update(self, key: str, value: Any) -> None:
        """설정값을 업데이트합니다."""
        if not self._is_loaded:
            raise ConfigError("설정이 로드되지 않았습니다.")
        
        keys = key.split('.')
        config = self._config_data
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """전체 설정을 딕셔너리로 반환합니다."""
        return self._config_data.copy()
    
    def save_to_file(self, config_file: str) -> None:
        """현재 설정을 YAML 파일로 저장합니다."""
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.safe_dump(self._config_data, f, default_flow_style=False, 
                              allow_unicode=True, indent=2)
        except Exception as e:
            raise ConfigError(f"설정 파일 저장 오류: {e}")
    
    # 기존 코드와의 호환성을 위한 속성들
    # Env
    @property
    def device(self) -> str:
        return self.get('ENV.DEVICE')
    
    @property
    def gpu_num(self) -> int:
        return self.get('ENV.GPU_NUM', 0)
    
    # Log
    @property
    def log_level(self) -> str:
        return self.get('LOG.LOG_LEVEL')
    
    @property
    def logger_name(self) -> str:
        return self.get('LOG.LOGGER_NAME')
    
    @property
    def console_log(self) -> bool:
        return self.get('LOG.CONSOLE_LOG')
    
    @property
    def console_log_interval(self) -> int:
        return self.get('LOG.CONSOLE_LOG_INTERVAL')
    
    @property
    def file_log(self) -> bool:
        return self.get('LOG.FILE_LOG')
    
    @property
    def file_log_dir(self) -> str:
        return self.get('LOG.FILE_LOG_DIR')
    
    @property
    def file_log_counter(self) -> int:
        return self.get('LOG.FILE_LOG_COUNTER')
    
    @property
    def file_log_rotate_time(self) -> str:
        return self.get('LOG.FILE_LOG_ROTATE_TIME')
    
    @property
    def file_log_rotate_interval(self) -> int:
        return self.get('LOG.FILE_LOG_ROTATE_INTERVAL')


# 전역 설정 인스턴스 (기존 코드 호환성)
_global_config = Config()


def set_config(file: str) -> None:
    """설정 파일을 로드합니다."""
    _global_config.load_from_file(file)


def get_config() -> Config:
    """전역 설정 인스턴스를 반환합니다."""
    return _global_config


# 편의 함수들
def get_config_value(key: str, default: Any = None) -> Any:
    """설정값을 직접 가져옵니다."""
    return _global_config.get(key, default)


if __name__ == '__main__':
    # 테스트 코드 - configs/config.yaml 파일 사용
    try:
        print("=== Config 테스트 시작 ===")
        
        # 1. Config 인스턴스 생성 및 로드
        config = Config()
        config.load_from_file('configs/config.yaml')
        print("✅ 설정 파일 로드 성공")
        
        # 2. 기본 설정 값들 확인
        print(f"Device: {config.device}")
        print(f"GPU Num: {config.gpu_num}")
        print(f"Logger Name: {config.logger_name}")
        print(f"Log Level: {config.log_level}")
        
        # 3. 전역 함수 테스트
        print("\n=== 전역 함수 테스트 ===")
        set_config('configs/config.yaml')
        global_config = get_config()
        print(f"Global Config Device: {global_config.device}")
        
        track_model = get_config_value('TRACK.TRACK_MODEL_TYPE', 'default')
        print(f"Track Model Type: {track_model}")
        
        print("\n🎉 모든 테스트 성공!")
        
    except ConfigError as e:
        print(f"❌ 설정 오류: {e}")
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")
