import unittest

from text_cleaner import clean_and_extract_alt


class TestTextCleaner(unittest.TestCase):
    def test_basic_tag_removal(self):
        # 1. 基本的なタグ除去のテスト (bタグ)
        input_text = "みなさん<b>こんにちは</b>です。"
        expected_output = "みなさん こんにちは です。"
        self.assertEqual(clean_and_extract_alt(input_text), expected_output)

    def test_img_alt_extraction(self):
        # 2. alt属性の抽出テスト
        input_text = 'この<img src="https://files.kick.com/emotes/1730770/original" alt="emojiDown" />アイコン'
        expected_output = "この emojiDown アイコン"
        self.assertEqual(clean_and_extract_alt(input_text), expected_output)

    def test_mixed_tags_and_alt(self):
        # 3. 複数のタグとalt属性が混在するテスト
        input_text = '<b>重要:</b> <a href="#">詳細</a><img src="x" alt="👍"/>を確認。'
        expected_output = "重要: 詳細 👍 を確認。"
        self.assertEqual(clean_and_extract_alt(input_text), expected_output)

    def test_no_tags(self):
        # 4. タグがない場合のテスト
        input_text = "ただのテキストです。"
        expected_output = "ただのテキストです。"
        self.assertEqual(clean_and_extract_alt(input_text), expected_output)

    def test_img_no_alt(self):
        # 5. alt属性がないimgタグのテスト
        input_text = '画像<img src="x" />は無視。'
        expected_output = "画像 は無視。"
        self.assertEqual(clean_and_extract_alt(input_text), expected_output)

    def test_leading_and_trailing_spaces(self):
        # 6. 先頭と末尾の空白除去のテスト
        input_text = " <p>テスト</p> "
        expected_output = "テスト"
        self.assertEqual(clean_and_extract_alt(input_text), expected_output)

    def test_various_quotes_and_spacing(self):
        # 7. alt属性の引用符とスペースの多様性テスト
        input_text = 'a<img alt=\'b\' src="x">c<img alt="d" src="y">e'
        expected_output = "a b c d e"
        self.assertEqual(clean_and_extract_alt(input_text), expected_output)
