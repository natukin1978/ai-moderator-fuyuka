import logging
import sys


def setup_app_logging(log_level=logging.INFO, log_file_path="app.log"):
    """
    アプリケーション全体のロギング設定を行います。
    ルートロガーにハンドラを設定するのが最も確実です。
    """

    # 既存のハンドラをクリア（二重設定を防ぐため）
    root_logger = logging.getLogger()
    if root_logger.handlers:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)

    # ロガーのレベル設定
    root_logger.setLevel(log_level)

    # フォーマッター
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 1. 画面出力 (StreamHandler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 2. ファイル出力 (FileHandler)
    file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # 外部ライブラリのログが大量に出る場合は、必要に応じてここでレベルを設定
    # logging.getLogger('requests').setLevel(logging.WARNING)

    # logging.info("ロギング設定が完了しました。")
