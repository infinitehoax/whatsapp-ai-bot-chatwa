# WhatsApp AI Bot with OpenRouter

This Python bot connects to WhatsApp via a webhook, processes incoming messages (text and images), interacts with AI models via OpenRouter (currently configured for Quasar Alpha), and sends back responses.

## Features

-   **Chat-wa.com API**: Uses the chat-wa.com API for WhatsApp communication.
-   **WhatsApp Integration**: Receives messages via webhook using Flask.
-   **AI Integration**: Uses OpenRouter to access AI models (text and vision).
-   **Multimodal**: Handles both text and image messages.
-   **Conversation History**: Stores chat history per contact in `history.json`.
-   **Rate Limiting**:
    -   Daily limit per user (default: 150 messages/day).
    -   Anti-spam throttle (default: 5 messages/60 seconds).
-   **User Blocking**: Blocks users listed in `blocked_users.txt`.
-   **Concurrency**: Uses threading and a queue to handle multiple requests.
-   **Async API Calls**: Uses `httpx` for non-blocking calls to the OpenRouter text model.
-   **Ngrok Integration**: Uses `pyngrok` to expose the local Flask server for WhatsApp webhooks.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/infinitehoax/whatsapp-ai-bot-chatwa.git
    cd whatsapp-api
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv .venv
    # On Windows
    .\.venv\Scripts\activate
    # On macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables:**
    -   Copy or rename `.env.example` to `.env` (if an example file exists) or create a new `.env` file.
    -   Fill in the required values:
        -   `NGROK_AUTHTOKEN`: Your ngrok authentication token.
        -   `WHATSAPP_API_KEY`: Your WhatsApp Business API key.
        -   `WHATSAPP_DEVICE_ID`: Your WhatsApp device ID.
        -   `WHATSAPP_BASE_URL`: The base URL for your WhatsApp API provider.
        -   `OPENROUTER_API_KEY`: Your OpenRouter API key.
        -   `AI_MODEL`: (Optional) OpenRouter model ID for vision (defaults to `openrouter/quasar-alpha`).
        -   `YOUR_SITE_URL`: (Optional) Your site URL for OpenRouter headers.
        -   `YOUR_SITE_NAME`: (Optional) Your site name for OpenRouter headers.

5.  **Configure Blocked Users:**
    -   Edit `blocked_users.txt` and add any phone numbers (one per line) you wish to block.

## Running the Bot

```bash
python bot.py
```

-   The script will start the Flask server and create an ngrok tunnel.
-   It will print the public ngrok URL. Use this URL (`<public_url>/webhook`) as the callback URL in your WhatsApp API provider (chat-wa.com) settings.

## Files

-   `bot.py`: The main application script.
-   `requirements.txt`: Python dependencies.
-   `history.json`: Stores conversation history (auto-generated, ignored by git).
-   `rate_limits.json`: Stores daily rate limit counts (auto-generated, ignored by git).
-   `user_throttle.json`: Stores timestamps for spam throttling (auto-generated, ignored by git).
-   `blocked_users.txt`: List of blocked phone numbers (ignored by git).
-   `README.md`: This file.
-   `LICENSE`: Project license file.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.