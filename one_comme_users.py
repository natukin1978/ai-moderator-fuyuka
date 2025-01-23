import os
import pickle

import global_value as g
from cache_helper import get_cache_filepath
from csv_helper import read_csv_to_list


class OneCommeUsers:
    FILENAME_MAP_IS_FIRST_ON_STREAM = get_cache_filepath(
        "ai_moderator_fuyuka_map_is_first_on_stream.pkl"
    )

    @staticmethod
    def read_one_comme_users():
        pathUsersCsv = g.config["oneComme"]["pathUsersCsv"]
        if not pathUsersCsv:
            return None

        return read_csv_to_list(pathUsersCsv)

    @staticmethod
    def get_nickname(displayName: str) -> str:
        if not g.one_comme_users:
            return None

        filtered_rows = list(
            filter(lambda row: row[1] == displayName, g.one_comme_users)
        )
        for filtered_row in filtered_rows:
            return filtered_row[4]

        return None

    @staticmethod
    def update_nickname(json_data: dict[str, any]) -> None:
        nickname = OneCommeUsers.get_nickname(json_data["displayName"])
        if nickname:
            json_data["nickname"] = nickname

    @classmethod
    def load_is_first_on_stream(cls) -> bool:
        if not os.path.isfile(cls.FILENAME_MAP_IS_FIRST_ON_STREAM):
            return False
        with open(cls.FILENAME_MAP_IS_FIRST_ON_STREAM, "rb") as f:
            g.map_is_first_on_stream = pickle.load(f)
            return True

    @classmethod
    def save_is_first_on_stream(cls) -> None:
        with open(cls.FILENAME_MAP_IS_FIRST_ON_STREAM, "wb") as f:
            pickle.dump(g.map_is_first_on_stream, f)

    @staticmethod
    def update_is_first_on_stream(json_data: dict[str, any]) -> None:
        name = json_data["id"]
        val = None
        if name not in g.map_is_first_on_stream:
            val = True
        else:
            val = g.map_is_first_on_stream[name]
        json_data["isFirstOnStream"] = val
        g.map_is_first_on_stream[name] = False
        OneCommeUsers.save_is_first_on_stream()

    @staticmethod
    def update_additional_requests(
        json_data: dict[str, any], answer_length: int
    ) -> None:
        json_data["additionalRequests"] = " ".join(
            [
                g.ADDITIONAL_REQUESTS_PROMPT.format(answerLength=answer_length),
            ]
        )

    @staticmethod
    def update_message_json(json_data: dict[str, any]) -> None:
        OneCommeUsers.update_is_first_on_stream(json_data)
        OneCommeUsers.update_nickname(json_data)
        OneCommeUsers.update_additional_requests(json_data, 30)
