import msvcrt
import time


def input_with_timeout(prompt, timeout=3):
    print(prompt, end="", flush=True)
    start_time = time.time()
    input_str = ""

    while True:
        if msvcrt.kbhit(): # キーが押されたかチェック
            char = msvcrt.getche().decode("utf-8")
            if char in ("\r", "\n"): # エンターキー
                break
            input_str += char

        if (time.time() - start_time) > timeout:
            print("\nタイムアウト：デフォルト(n)で進行します。")
            return "n"

    return input_str.strip()
