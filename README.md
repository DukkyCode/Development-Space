## Development-Space: Discord Risk Metric Bot

This project is a Discord bot that fetches historical Bitcoin price data from Yahoo Finance, computes a custom **risk metric**, and sends various interactive plots (line chart, heatmap, and prediction table) to your selected Discord channel.

---

### Features

* 📈 Computes BTC risk levels based on long-term moving averages and log-return scaling.
* 🧠 Predicts future BTC prices based on risk level buckets.
* 🎨 Sends 3 visualizations:

  * Risk over time (dual-axis plot)
  * BTC price colored by risk level
  * Risk-level vs. predicted price table
* 🤖 Command-based interaction in Discord (`!riskgraph`).

---

### Installation

1. **Clone this repository**:

   ```bash
   git clone https://github.com/yourusername/Development-Space.git
   cd Development-Space
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your Discord bot**:

   * Go to [Discord Developer Portal](https://discord.com/developers/applications)
   * Create an application → Add a Bot
   * Enable `MESSAGE CONTENT INTENT` and `SERVER MEMBERS INTENT`
   * Copy your **bot token**

4. **Run the bot**:

   Edit `bot.py` and replace the placeholder token with your actual token.

   ```python
   bot.run("YOUR_BOT_TOKEN")
   ```

   Then run:

   ```bash
   python bot.py
   ```

---

### Usage in Discord

Once the bot is running and added to your server:

* Type `!riskgraph` in any channel where the bot has access.
* The bot will compute the data and send graphs and a table with insights directly to that channel.

### Dependencies

All dependencies are managed via `requirements.txt`. Includes:

* `discord.py`
* `yfinance`
* `pandas`
* `numpy`
* `plotly`
* `matplotlib` (optional for headless image generation)



