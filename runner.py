import socket
import threading

import requests

from bots.adam.claudetwo import ClaudeTwo
from bots.adam.iterone import IterOne
from bots.sushi_go_client import SushiGoClient

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


if __name__ == "__main__":
    run_game(
        [
            SushiGoClient(HOST, PORT),
            SushiGoClient(HOST, PORT),
            IterOne(),
        ]
    )
