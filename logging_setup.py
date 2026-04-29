import logging
import sys


class ForceFilter(logging.Filter):
    def __init__(self, threshold_level):
        super().__init__()
        self.threshold_level = threshold_level

    def filter(self, record):
        if getattr(record, 'force', False):
            # 'force'フラグがあれば無条件で通す
            return True
        return record.levelno >= self.threshold_level


def to_log_level(log_level_str: str):
    log_level = getattr(logging, log_level_str, logging.INFO)
    return log_level


def setup_app_logging(log_level_str: str, log_file_path="app.log"):
    """
    アプリケーション全体のロギング設定を行います。
    ルートロガーにハンドラを設定するのが最も確実です。
    """

    # 既存のハンドラをクリア（二重設定を防ぐため）
    root_logger = logging.getLogger()
    if root_logger.handlers:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)

    # ロガー自体のレベルは一番低いもの
    root_logger.setLevel(logging.DEBUG)

    # フォーマッター
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ロガーのレベル設定
    log_level = to_log_level(log_level_str)
    log_filter = ForceFilter(log_level)

    # 画面出力 (StreamHandler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(log_filter)
    root_logger.addHandler(console_handler)

    # ファイル出力 (FileHandler)
    file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.addFilter(log_filter)
    root_logger.addHandler(file_handler)

    # 外部ライブラリのログが大量に出る場合は、必要に応じてここでレベルを設定
    # logging.getLogger('requests').setLevel(logging.WARNING)

    # logging.info("ロギング設定が完了しました。")
