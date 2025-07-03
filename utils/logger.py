import os
import sys
import logging
from logging import StreamHandler
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime
from pathlib import Path


class LoggerManager:
    """로거 관리 클래스 - 싱글톤 패턴으로 구현"""
    _instance = None
    _loggers = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_logger(cls, name="default", cfg=None, loglevel="DEBUG"):
        """로거 인스턴스를 반환합니다."""
        if name not in cls._loggers:
            cls._loggers[name] = cls._create_logger(name, cfg, loglevel)
        return cls._loggers[name]
    
    @staticmethod
    def _create_logger(name, cfg=None, loglevel="DEBUG"):
        """새로운 로거를 생성합니다."""
        logger = logging.getLogger(name)
        
        # 기존 핸들러 제거 (중복 방지)
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # 로그 레벨 설정
        log_level = getattr(logging, loglevel.upper(), logging.DEBUG)
        logger.setLevel(log_level)
        
        # 로그 포맷
        log_format = "[%(asctime)s]-[%(levelname)s]-[%(name)s]-[%(module)s](%(process)d): %(message)s"
        formatter = logging.Formatter(log_format)
        
        if cfg is not None:
            # 설정 기반 로거 생성
            logger = LoggerManager._setup_configured_logger(logger, cfg, formatter)
        else:
            # 기본 로거 생성
            logger = LoggerManager._setup_default_logger(logger, name, formatter, log_level)
        
        logger.info(f"Logger '{name}' initialized successfully")
        return logger
    
    @staticmethod
    def _setup_configured_logger(logger, cfg, formatter):
        """설정 파일 기반 로거 설정"""
        # 로그 레벨 재설정
        if hasattr(cfg, 'log_level'):
            log_level = getattr(logging, cfg.log_level.upper(), logging.DEBUG)
            logger.setLevel(log_level)
        
        # 콘솔 핸들러
        if getattr(cfg, 'console_log', True):
            console_handler = StreamHandler()
            console_handler.setLevel(logger.level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # 파일 핸들러
        if getattr(cfg, 'file_log', False):
            try:
                file_handler = LoggerManager._create_file_handler(cfg, formatter, logger.level)
                logger.addHandler(file_handler)
            except Exception as e:
                logger.warning(f"파일 로그 설정 실패: {e}")
        
        return logger
    
    @staticmethod
    def _setup_default_logger(logger, name, formatter, log_level):
        """기본 로거 설정"""
        # 콘솔 핸들러
        console_handler = StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 기본 파일 핸들러
        try:
            timestamp = datetime.now().strftime("%Y%m%d")
            log_dir = Path("./logs")
            log_dir.mkdir(exist_ok=True)
            
            filename = log_dir / f"{name}_{timestamp}.log"
            file_handler = TimedRotatingFileHandler(
                filename=str(filename),
                when="D",
                interval=1,
                backupCount=10,
                encoding='utf8'
            )
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"기본 파일 로그 설정 실패: {e}")
        
        return logger
    
    @staticmethod
    def _create_file_handler(cfg, formatter, log_level):
        """파일 핸들러 생성"""
        timestamp = datetime.now().strftime("%Y%m%d")
        log_dir = Path(getattr(cfg, 'file_log_dir', './logs'))
        log_dir.mkdir(parents=True, exist_ok=True)
        
        name = getattr(cfg, 'logger_name', 'default')
        filename = log_dir / f"{name}_{timestamp}.log"
        
        when = getattr(cfg, 'file_log_rotate_time', 'D')
        interval = getattr(cfg, 'file_log_rotate_interval', 1)
        backup_count = getattr(cfg, 'file_log_counter', 10)
        
        file_handler = TimedRotatingFileHandler(
            filename=str(filename),
            when=when,
            interval=interval,
            backupCount=backup_count,
            encoding='utf8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        
        return file_handler


# 편의 함수들
def init_logger(cfg=None, name="default", loglevel="DEBUG"):
    """로거를 초기화합니다."""
    return LoggerManager.get_logger(name, cfg, loglevel)


def get_logger(name="default"):
    """로거 인스턴스를 반환합니다."""
    return LoggerManager.get_logger(name)


if __name__ == '__main__':
    # 테스트 코드 - config.yaml과 연동 테스트
    try:
        print("=== Logger 기본 테스트 ===")
        
        # 1. 기본 로거 테스트
        logger = init_logger(name="test")
        logger.info("Hello World!")
        logger.debug("디버그 메시지")
        logger.warning("경고 메시지")
        logger.error("에러 메시지")
        
        # 2. Config와 연동 테스트
        print("\n=== Config 연동 테스트 ===")
        
        # Config 모듈 임포트 (동적 임포트로 순환 참조 방지)
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent))
        
        from config import Config
        
        # Config 로드
        config = Config()
        config.load_from_file('configs/config.yaml')
        print(f"✅ Config 로드 성공")
        
        # Config 기반 로거 생성
        config_logger = init_logger(cfg=config, name="config-test")
        config_logger.info("Config 기반 로거 테스트")
        config_logger.debug("Config 로거 디버그 메시지")
        config_logger.warning("Config 로거 경고 메시지")
        
        print(f"Config Logger Level: {config_logger.level}")
        print(f"Config Logger Name: {config_logger.name}")
        
        # 3. 로그 파일 확인
        print("\n=== 로그 파일 확인 ===")
        log_dir = Path("./logs")
        if log_dir.exists():
            log_files = list(log_dir.glob("*.log"))
            print(f"생성된 로그 파일: {[f.name for f in log_files]}")
        
        # 4. Config LOG 섹션 값들 확인
        print("\n=== Config LOG 섹션 확인 ===")
        print(f"LOG_LEVEL: {config.get('LOG.LOG_LEVEL')}")
        print(f"LOGGER_NAME: {config.get('LOG.LOGGER_NAME')}")
        print(f"CONSOLE_LOG: {config.get('LOG.CONSOLE_LOG')}")
        print(f"CONSOLE_LOG_INTERVAL: {config.get('LOG.CONSOLE_LOG_INTERVAL')}")
        print(f"FILE_LOG: {config.get('LOG.FILE_LOG')}")
        print(f"FILE_LOG_DIR: {config.get('LOG.FILE_LOG_DIR')}")
        print(f"FILE_LOG_COUNTER: {config.get('LOG.FILE_LOG_COUNTER')}")
        print(f"FILE_LOG_ROTATE_TIME: {config.get('LOG.FILE_LOG_ROTATE_TIME')}")
        print(f"FILE_LOG_ROTATE_INTERVAL: {config.get('LOG.FILE_LOG_ROTATE_INTERVAL')}")
        
        print("\n🎉 모든 테스트 성공!")
        
    except Exception as e:
        print(f"❌ 테스트 오류: {e}")
        import traceback
        traceback.print_exc()
