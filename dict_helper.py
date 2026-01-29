from typing import Any, Dict, List


def remove_keys_by_value(
    data: Dict[str, Any], target_keys: List[str], target_value: Any
) -> None:
    """
    指定されたキーのリストの中で、特定の値を持つ要素のみを削除します。

    Args:
        data: 操作対象の辞書
        target_keys: 削除対象として検討するキーのリスト
        target_value: 削除の条件となる値
    """
    # 削除対象となるキーを抽出する
    keys_to_delete = []

    for key in target_keys:
        # 指定されたキーが辞書に存在するか確認
        if key in data:
            value = data[key]
            # 値が一致し、かつ型も一致するか確認
            if value == target_value and type(value) is type(target_value):
                keys_to_delete.append(key)

    # まとめて削除
    for key in keys_to_delete:
        del data[key]
