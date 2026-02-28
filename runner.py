import socket
import threading
from typing import Callable

import requests

from bots.adam.iterfour import Iter4
from bots.jacob.chopstick_hater import ChopstickHater
from bots.sushi_go_client import SushiGoClient

# HOST = "10.8.1.191"
HOST = "localhost"
PORT = 7878
GAMES_URL = "http://localhost:8080/api/games"

print(f"Connecting to {HOST}:{PORT}...")
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((HOST, PORT))
sock_file = sock.makefile("r", encoding="utf-8", errors="replace")
print("Connected!")


def create_game(player_count: int):
    response = requests.post(GAMES_URL, json={"max_players": player_count})
    print(response.text)


def send(cmd: str):
    print(f">>> {cmd}")
    sock.sendall((cmd + "\n").encode())


def recv() -> str:
    line = sock_file.readline()
    if line == "":
        raise ConnectionError("Server closed connection")
    msg = line.strip()
    print(f"<<< {msg}")
    return msg


def recv_until(predicate):
    while True:
        msg = recv()
        if not msg:
            continue
        if predicate(msg):
            return msg


def list_games() -> list[dict]:
    send("GAMES")
    games = recv_until(lambda s: s.startswith("GAMES"))
    games = eval(games[6:])
    return games


def newest_game() -> str:
    return list_games()[-1]["id"]


def run_game(players: list[SushiGoClient]):
    create_game(len(players))
    game = newest_game()
    threads = []
    for i, player in enumerate(players):
        name = player.name
        if not name:
            name = f"Player{i}"
        t = threading.Thread(
            target=player.run,
            args=(game, name),
            daemon=True,
        )
        t.start()
        threads.append(t)
    for thread in threads:
        thread.join()


def faceoff(count: int, players: Callable[[], list[SushiGoClient]]):
    wins = {}
    for _ in range(count):
        round_players = players()
        run_game(round_players)
        winner = round_players[0].winner
        if winner not in wins:
            wins[winner] = 0
        wins[winner] += 1
    return wins


if __name__ == "__main__":
    tourney_id = "tasty-nori-172423pD"
    threads = []
    for i in range(32):
        player = Iter4()
        t = threading.Thread(
            target=player.run_tournament,
            args=(tourney_id, f"ChopstickHater{i}"),
            daemon=True,
        )
        t.start()
        threads.append(t)
    for thread in threads:
        thread.join()

# if __name__ == "__main__":
#     player = Iter4()
#     game_id = "sweet-salmon-31733vy0"
#     player.run(game_id, "ChopstickHater")

# if __name__ == "__main__":
#     player = Iter4()
#     rejoin_token = "MOjkXBz7gUhg6nhksMPmUslVXou71fxg"
#     player.rejoin(rejoin_token)
