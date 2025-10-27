import re


def clean_and_extract_alt(text: str) -> str:
    """
    1. HTMLタグを除去し、その前後に空白を入れる。
    2. <img>タグのalt属性があれば、その値に置き換える。

    Args:
        text: 処理対象の文字列。

    Returns:
        処理後の文字列。
    """

    # 処理ステップ1: alt属性を持つimgタグを、そのalt属性値に置き換える
    # パターン: <img ... alt="値" ... />
    # (?P<alt_value>...) でalt属性の値を名前付きグループとしてキャプチャ
    # \s*alt=["'](?P<alt_value>[^"']+)["']\s* を含むように修正
    # alt属性がないimgタグも考慮し、alt属性はオプションにする

    # imgタグの処理 (alt属性がある場合、その値に置き換え)
    # <img ... alt="capture_group" ...> または <img ... alt='capture_group' ...> に対応
    # alt属性の有無に関わらず、タグ全体を処理対象とする
    img_pattern = r'<img\s+[^>]*?\s*alt=["\']([^"\']+?)["\'][^>]*?/?>'

    def replace_img_with_alt(match):
        """alt属性の値があればそれに置き換え、なければタグ全体を除去（空白追加）"""
        # グループ1がalt属性の値
        alt_value = match.group(1)
        # alt属性値の前後に空白を付加して返す
        return f" {alt_value} "

    # alt属性を持つimgタグを先に処理
    # 例外処理として、alt属性を持つタグのみを先に置換
    processed_text = re.sub(
        img_pattern, replace_img_with_alt, text, flags=re.IGNORECASE
    )

    # 処理ステップ2: alt属性を持たないimgタグを含む、残りの全てのHTMLタグを除去し、空白で置換
    # パターン: <...>（属性や内容を含む）
    # 非貪欲マッチで、タグ内部の文字を最小限でキャプチャ
    tag_pattern = r"<[^>]+>"

    # タグを「 [空白] 」に置換
    # これにより、タグがあった位置に空白が挿入され、前後の文字と結合される
    final_text = re.sub(tag_pattern, " ", processed_text)

    # 処理ステップ3: 連続する複数の空白を1つの空白に置き換え、前後の空白を除去
    final_text = re.sub(r"\s+", " ", final_text).strip()

    return final_text
