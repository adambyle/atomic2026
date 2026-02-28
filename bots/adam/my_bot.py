from ..sushi_go_client import SushiGoClient

HOST = "localhost"
PORT = 7878


class MyBot(SushiGoClient):
    def __init__(self):
        super().__init__(HOST, PORT)

    def choose_card(self, hand: list[str]) -> int:
        return 0
