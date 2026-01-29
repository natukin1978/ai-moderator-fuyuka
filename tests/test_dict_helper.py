import unittest

from dict_helper import remove_keys_by_value


class TestDictHelper(unittest.TestCase):
    def test_remove_by_value_success(self):
        """指定したキーと値が一致する場合、正しく削除されること"""
        data = {
            "A": "A",
            "B": True,
            "C": False,
            "D": False,
        }
        target_keys = ["C", "D"]
        target_value = False

        remove_keys_by_value(data, target_keys, target_value)

        # CとDが消え、AとBが残っていることを確認
        self.assertEqual(data, {"A": "A", "B": True})

    def test_remove_by_value_type_strict(self):
        """値が等価でも型が異なる場合は削除されないこと (False と 0 の区別)"""
        data = {
            "is_active": False,
            "count": 0,
        }
        # Falseを指定して削除を試みる
        remove_keys_by_value(data, ["is_active", "count"], False)

        # is_activeは消えるが、0であるcountは残るべき
        self.assertIn("count", data)
        self.assertNotIn("is_active", data)
        self.assertEqual(data["count"], 0)

    def test_remove_by_value_missing_key(self):
        """辞書に存在しないキーを指定してもエラーにならないこと"""
        data = {"A": "A"}
        target_keys = ["A", "NON_EXISTENT_KEY"]

        # エラーが起きずに実行できることを確認
        remove_keys_by_value(data, target_keys, "A")
        self.assertEqual(data, {})

    def test_remove_by_value_no_match(self):
        """キーは存在するが値が一致しない場合、削除されないこと"""
        data = {"status": "pending"}
        remove_keys_by_value(data, ["status"], "completed")

        self.assertEqual(data, {"status": "pending"})
