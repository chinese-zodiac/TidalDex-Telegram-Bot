# TidalDex Telegram Bot

A Telegram bot that monitors trading pairs on TidalDex and sends real-time alerts for buy transactions with custom-generated DALL-E images.

## Features

- 🔄 Real-time monitoring of trading pairs
- 🖼️ AI-generated unique pair images using DALL-E
- 💱 Automatic detection of buy transactions
- 💰 Minimum trade amount filtering
- 🎯 Base asset priority system
- 🔒 Admin-only commands
- 📊 USD value calculation for trades

## Commands

- `/addpair <address>` - Add a new trading pair to monitor
- `/removepair <address>` - Remove a trading pair from monitoring
- `/listpairs` - List all tracked trading pairs
- `/regenimg <address>` - Regenerate the DALL-E image for a pair
- `/setimage <address>` - Set a custom image for a pair

## Setup

### Prerequisites

- Python 3.8+
- Telegram Bot Token
- OpenAI API Key
- BSC RPC URL

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/TidalDex-Telegram-Bot.git
cd TidalDex-Telegram-Bot
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file with the following variables:

```env
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
OPENAI_API_KEY=your_openai_api_key
BSC_RPC_URL=your_bsc_rpc_url
DEFAULT_TOKEN_LIST=your_token_list_url
```

### Running the Bot

```bash
python main.py
```

## Configuration Files

- `baseassetlist.json`: Define base assets and their minimum trade quantities
- `tidaldex_tokens.json`: Token list with symbols, decimals, and logos (auto-updated)
- `tracked_pairs.json`: Stores tracked pairs and their configurations

## Features in Detail

### Trade Detection

- Monitors swap events on tracked trading pairs
- Identifies buy transactions based on token priorities
- Filters trades based on minimum quantities for base assets

### Image Generation

- Creates unique underwater-themed images for each pair using DALL-E
- Automatically overlays token logos
- Supports custom image uploads via `/setimage`

### Trade Alerts

- Real-time notifications for buy transactions
- Includes trade details (amounts, prices, USD value)
- Links to BSCScan transaction and TidalDex swap interface
- Displays pair-specific images with alerts

## Security

- Admin-only access for sensitive commands
- Environment variables for all sensitive data
- No hardcoded secrets
- Secure error handling

## Error Handling

- Graceful handling of network issues
- Automatic reconnection for dropped connections
- Detailed error logging
- Timeout handling for image generation

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot)
- [OpenAI DALL-E](https://openai.com/dall-e-3)
- [Web3.py](https://web3py.readthedocs.io/)
- [TidalDex](https://tidaldex.com)
