# Python Client

## Requirements

- Python 3.10+
- Standard library only — no external packages needed

## Files

| File | Description |
|------|-------------|
| `sushi_go_client.py` | Full-featured client with state tracking and a priority-based strategy |
| `first_card_bot.py` | Minimal bot (~30 lines of logic) that always plays the first card |

## Usage

The two scripts have different argument orders:

```bash
# first_card_bot.py — game_id and name first, host/port optional
python first_card_bot.py <game_id> <player_name> [host] [port]
python first_card_bot.py abc123 MyBot
python first_card_bot.py abc123 MyBot 192.168.1.50 7878

# sushi_go_client.py — host and port first
python sushi_go_client.py <host> <port> <game_id> <player_name>
python sushi_go_client.py localhost 7878 abc123 MyBot
```

## Implementing Your Strategy

Edit the `choose_card` method in `sushi_go_client.py`:

```python
def choose_card(self, hand: list[str]) -> int:
    """
    Choose which card to play.

    Args:
        hand: List of card names (e.g., ["Tempura", "Salmon Nigiri", "Pudding"])

    Returns:
        Index of the card to play (0-based)
    """
    # Your strategy here!
    return 0
```

The default implementation uses a simple priority list. Replace it with your own logic.

## Key Patterns

### Line-buffered reading

`first_card_bot.py` uses `socket.makefile('r')` for reliable line-by-line reading:

```python
sock_file = sock.makefile('r')
msg = sock_file.readline().strip()
```

### HAND = your turn

Only send `PLAY` when you receive a `HAND` message. The server sends `HAND` exactly when it's time for you to act — not as a status update.

### State tracking

`sushi_go_client.py` tracks played cards, chopsticks, and wasabi state for you. Use `self.state` to make smarter decisions.

## Protocol

See [../PROTOCOL.md](../PROTOCOL.md) for the full protocol specification.
