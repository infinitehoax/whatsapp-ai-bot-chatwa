import os
import requests
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIError, APIConnectionError # Import specific errors for retry
import json
from datetime import datetime, timezone
import time # Import time for retry delays
from flask import Flask, request, jsonify
from pyngrok import ngrok, conf
import re
import threading
import collections
import httpx
import asyncio


# Load environment variables from .env file
load_dotenv()

# --- Ngrok Configuration ---
NGROK_AUTHTOKEN = os.getenv("NGROK_AUTHTOKEN")
if NGROK_AUTHTOKEN:
    ngrok.set_auth_token(NGROK_AUTHTOKEN)
    print("Configured pyngrok with authtoken from environment variable.")
else:
    print("WARN: NGROK_AUTHTOKEN not found in .env file. ngrok tunnel may be limited or fail.")

# WhatsApp API Configuration
WHATSAPP_API_KEY = os.getenv("WHATSAPP_API_KEY")
WHATSAPP_DEVICE_ID = os.getenv("WHATSAPP_DEVICE_ID")
WHATSAPP_BASE_URL = os.getenv("WHATSAPP_BASE_URL")

# OpenRouter Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
YOUR_SITE_URL = os.getenv("YOUR_SITE_URL")
YOUR_SITE_NAME = os.getenv("YOUR_SITE_NAME")
# Placeholder - Replace with the actual Gemini vision model identifier on OpenRouter
AI_MODEL = os.getenv("AI_MODEL", "openrouter/quasar-alpha")
# --- Request Queue Management ---
MAX_CONCURRENT_REQUESTS = 3  # Adjust as needed
request_queue = collections.deque()
active_requests = 0
queue_lock = threading.Lock()
# --- Blocked Users ---
BLOCKED_USERS_FILE = "blocked_users.txt"

import time as time_module

_blocked_cache = {
    "users": set(),
    "last_loaded": 0
}
BLOCKED_CACHE_TTL = 300  # seconds (5 minutes)

def load_blocked_users():
    now = time_module.time()
    if now - _blocked_cache["last_loaded"] > BLOCKED_CACHE_TTL:
        # Refresh cache
        if not os.path.exists(BLOCKED_USERS_FILE):
            _blocked_cache["users"] = set()
        else:
            try:
                with open(BLOCKED_USERS_FILE, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    _blocked_cache["users"] = set(line.strip() for line in lines if line.strip())
            except IOError:
                _blocked_cache["users"] = set()
        _blocked_cache["last_loaded"] = now
    return _blocked_cache["users"]

# --- Rate Limiting ---
RATE_LIMIT_FILE = "rate_limits.json"
MAX_REQUESTS_PER_DAY = 150

def load_rate_limits():
    if not os.path.exists(RATE_LIMIT_FILE):
        return {}
    try:
        with open(RATE_LIMIT_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content:
                return {}
            return json.loads(content)
    except (json.JSONDecodeError, IOError):
        return {}

def save_rate_limits(data):
    try:
        with open(RATE_LIMIT_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    except IOError:
        pass

def is_user_rate_limited(phone_number):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    data = load_rate_limits()
    user_data = data.get(phone_number, {})
    last_date = user_data.get("date")
    count = user_data.get("count", 0)

    if last_date != today:
        # Reset count for new day
        data[phone_number] = {"date": today, "count": 1}
        save_rate_limits(data)
        return False
    else:
        if count >= MAX_REQUESTS_PER_DAY:
            return True
        else:
            user_data["count"] = count + 1
            data[phone_number] = user_data
            save_rate_limits(data)
            return False
AVERAGE_PROCESSING_TIME = 10  # seconds, for ETA calculation

# --- Per-user spam throttle ---
USER_THROTTLE_FILE = "user_throttle.json"
THROTTLE_LIMIT = 5  # max messages
THROTTLE_WINDOW = 60  # seconds

def load_user_throttle():
   if not os.path.exists(USER_THROTTLE_FILE):
       return {}
   try:
       with open(USER_THROTTLE_FILE, 'r', encoding='utf-8') as f:
           content = f.read()
           if not content:
               return {}
           return json.loads(content)
   except (json.JSONDecodeError, IOError):
       return {}

def save_user_throttle(data):
   try:
       with open(USER_THROTTLE_FILE, 'w', encoding='utf-8') as f:
           json.dump(data, f, indent=2)
   except IOError:
       pass

def is_user_spamming(phone_number):
   now = time_module.time()
   data = load_user_throttle()
   timestamps = data.get(phone_number, [])

   # Remove timestamps older than window
   timestamps = [ts for ts in timestamps if now - ts < THROTTLE_WINDOW]

   if len(timestamps) >= THROTTLE_LIMIT:
       # User is spamming
       return True
   else:
       # Add current timestamp and save
       timestamps.append(now)
       data[phone_number] = timestamps
       save_user_throttle(data)
       return False

# History Configuration
HISTORY_FILE = "history.json"
MAX_HISTORY_PER_CONTACT = 15

# --- History Management (JSON File) ---
# (Functions load_history, save_history, get_contact_history, add_message_to_history remain the same)
def load_history():
    if not os.path.exists(HISTORY_FILE): return {}
    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            content = f.read();
            if not content: return {}
            return json.loads(content)
    except (json.JSONDecodeError, IOError) as e: print(f"Error loading history file ({HISTORY_FILE}): {e}. Starting fresh."); return {}

def save_history(history_data):
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f: json.dump(history_data, f, indent=2, ensure_ascii=False)
    except IOError as e: print(f"Error saving history file ({HISTORY_FILE}): {e}")

def get_contact_history(contact_number, history_data):
    return history_data.get(contact_number, [])

def add_message_to_history(contact_number, role, content, history_data):
    if contact_number not in history_data: history_data[contact_number] = []
    timestamp = datetime.now(timezone.utc).isoformat()
    history_data[contact_number].append({"role": role, "content": content, "timestamp": timestamp})
    history_data[contact_number] = history_data[contact_number][-MAX_HISTORY_PER_CONTACT:]

# --- WhatsApp API Functions ---
# (Function send_whatsapp_message remains the same)
def send_whatsapp_message(to_number, text):
    if not WHATSAPP_API_KEY or not WHATSAPP_BASE_URL: print("Error: WhatsApp API credentials not found in .env"); return False
    url = f"{WHATSAPP_BASE_URL}/message/send-text"
    headers = {"accept": "application/json", "x-api-key": WHATSAPP_API_KEY, "Content-Type": "application/json"}
    payload = {"to": to_number, "text": text}
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        if data.get("success"): print(f"Successfully sent message to {to_number}"); return True
        else: error_details = data.get('data') or data.get('message', 'Unknown error'); print(f"Error sending message to {to_number}: {error_details}"); return False
    except requests.exceptions.RequestException as e:
        print(f"Error during WhatsApp API request (send_text): {e}")
        if e.response is not None: print(f"Response status: {e.response.status_code}\nResponse text: {e.response.text}")
        return False
    except json.JSONDecodeError: print(f"Error decoding JSON response from WhatsApp API (send_text): {response.text if 'response' in locals() else 'Unknown response'}"); return False

# --- OpenRouter AI Functions ---

# Function for Text-based AI responses
async def get_ai_response(user_message_content, history_list):
    if not OPENROUTER_API_KEY:
        print("Error: OPENROUTER_API_KEY not found in .env")
        return "Error: AI configuration missing."

    formatted_history = [{"role": msg["role"], "content": msg["content"]} for msg in history_list]
    system_prompt = {
        "role": "system",
        "content": "You are a helpful WhatsApp assistant. Respond concisely and format your messages using only WhatsApp's specific formatting: _italic_, *bold*, ~strikethrough~, ```monospace block```, `inline code`, > quote. For lists, use '* item' or '1. item'. Do not use Markdown headers or other complex Markdown."
    }
    final_messages = [system_prompt] + formatted_history

    print("\n--- Sending Text to AI ---")
    print(f"Latest User Message: {user_message_content}")
    print(f"History items sent: {len(history_list)}")
    print("---------------------\n")

    max_retries = 2
    for attempt in range(max_retries):
        try:
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            }
            if YOUR_SITE_URL:
                headers["HTTP-Referer"] = YOUR_SITE_URL
            else:
                print("Warning: YOUR_SITE_URL environment variable is missing or empty. Skipping HTTP-Referer header.")
            if YOUR_SITE_NAME:
                headers["X-Title"] = YOUR_SITE_NAME
            else:
                print("Warning: YOUR_SITE_NAME environment variable is missing or empty. Skipping X-Title header.")

            payload = {
                "model": "openrouter/quasar-alpha",
                "messages": final_messages,
                "temperature": 0.7,
                "max_tokens": 40000
            }

            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload
                )
            response.raise_for_status()
            data = response.json()
            ai_response = data["choices"][0]["message"]["content"].strip()
            print(f"Raw AI Response: {ai_response}")
            return ai_response

        except httpx.HTTPStatusError as e:
            print(f"HTTP error on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2)
            else:
                return f"Sorry, I encountered an API error after {max_retries} attempts."
        except Exception as e:
            print(f"Unexpected error calling OpenRouter (Text) API: {e}")
            return "Sorry, I encountered an unexpected error trying to generate a response."

    return "Sorry, failed to get text response after retries."

# Function for Image-based AI responses (Vision)
def get_ai_vision_response(prompt, image_url):
    if not OPENROUTER_API_KEY: print("Error: OPENROUTER_API_KEY not found in .env"); return "Error: AI configuration missing."
    if not AI_MODEL: print("Error: AI_MODEL not set."); return "Error: Vision model not configured."

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                    },
                },
            ],
        }
    ]
    # Add a system prompt specifically for the vision model
    vision_system_prompt = {
        "role": "system",
        "content": "You are analyzing an image provided by a user on WhatsApp. Describe the image based on the user's prompt. Respond concisely using only WhatsApp formatting: _italic_, *bold*, ~strikethrough~, ```monospace block```, `inline code`, > quote. For lists, use '* item' or '1. item'."
    }
    final_messages = [vision_system_prompt] + messages

    print("\n--- Sending Image to AI ---"); print(f"Prompt: {prompt}"); print(f"Image URL: {image_url}"); print("---------------------\n")

    max_retries = 20
    for attempt in range(max_retries):
        try:
            # Build headers dynamically to avoid NoneType errors
            headers = {}
            if YOUR_SITE_URL:
                headers["HTTP-Referer"] = YOUR_SITE_URL
            else:
                print("Warning: YOUR_SITE_URL environment variable is missing or empty. Skipping HTTP-Referer header.")
            if YOUR_SITE_NAME:
                headers["X-Title"] = YOUR_SITE_NAME
            else:
                print("Warning: YOUR_SITE_NAME environment variable is missing or empty. Skipping X-Title header.")

            completion = client.chat.completions.create(
                model=AI_MODEL,  # This defaults to openrouter/quasar-alpha
                messages=messages,
                max_tokens=30000,  # Adjust as needed for vision descriptions
                extra_headers=headers
            )
            # --- Check for errors within the completion object FIRST ---
            if completion and hasattr(completion, 'error') and completion.error:
                # Use .model_dump() for Pydantic v2 compatibility if needed, or direct access
                error_data = completion.error if isinstance(completion.error, dict) else completion.error.model_dump()
                error_code = error_data.get('code')
                error_message = error_data.get('message', 'Unknown error in completion object')
                print(f"API returned error in completion object (Attempt {attempt + 1}/{max_retries}): Code {error_code}, Message: {error_message}")
                try: print(f"Full error object: {json.dumps(error_data, indent=2)}")
                except: pass # Ignore if cannot serialize

                # Check if it's a retryable error (e.g., 429 Rate Limit/Quota)
                if error_code == 429:
                    if attempt < max_retries - 1:
                        retry_delay = 5 # Default delay
                        try: # Attempt to parse Google's specific retry delay
                            if 'metadata' in error_data and 'raw' in error_data['metadata']:
                                raw_error_details = json.loads(error_data['metadata']['raw'])
                                if 'error' in raw_error_details and 'details' in raw_error_details['error']:
                                    for detail in raw_error_details['error']['details']:
                                        if detail.get('@type') == 'type.googleapis.com/google.rpc.RetryInfo':
                                            delay_str = detail.get('retryDelay', '5s')
                                            retry_delay = int(re.sub(r'\D', '', delay_str)) # Extract digits
                                            break
                        except Exception as parse_e: print(f"Could not parse retry delay, using default: {parse_e}")

                        wait_time = max(retry_delay, (attempt + 1) * 2) # Use provider delay or backoff
                        print(f"Retrying after {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue # Go to the next attempt
                    else:
                        print("Max retries reached after rate limit error.")
                        return f"Sorry, the service is busy or quota exceeded after {max_retries} attempts. Please try again later."
                else: # Non-retryable error in completion object
                    return f"Sorry, the AI service returned an error: {error_message}"

            # --- If no error in completion, check for valid response ---
            elif completion and completion.choices and len(completion.choices) > 0:
                choice = completion.choices[0]
                if choice.message and choice.message.content:
                    ai_response = choice.message.content.strip()
                    print(f"Raw AI Vision Response: {ai_response}")
                    return ai_response # Success!
                else: # Valid structure but missing content
                    print(f"Error: AI response structure missing message content. Finish Reason: {choice.finish_reason}")
                    try: print(f"Problematic choice object: {choice.model_dump_json(indent=2)}")
                    except Exception as log_e: print(f"Could not serialize choice object: {log_e}")
                    # Don't retry if structure is valid but content missing
                    return "Sorry, the AI response was incomplete."
            else: # Unexpected structure (no choices, etc.)
                print("Error: AI response structure missing choices or other issue.")
                if completion:
                    try: print(f"Raw completion object: {completion.model_dump_json(indent=2)}")
                    except Exception as log_e: print(f"Could not serialize completion object: {log_e}")
                # Retry once for unexpected structure, might be transient
                if attempt < max_retries - 1:
                     print("Retrying due to unexpected response structure...")
                     time.sleep(attempt + 2)
                     continue
                else:
                    return "Sorry, the AI response structure was unexpected after retries."

        # --- Catch exceptions raised by the HTTP client/library ---
        except (RateLimitError, APIError, APIConnectionError) as e:
            print(f"Network/API Error calling OpenRouter (Vision) on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                time.sleep(attempt + 2) # Wait longer each retry (2, 3 seconds)
            else:
                print("Max retries reached for vision AI call.")
                return f"Sorry, I encountered an API error trying to analyze the image after {max_retries} attempts."
        except Exception as e: # Catch other unexpected errors
            print(f"Unexpected error calling OpenRouter (Vision) API: {e}")
            # Check if error response has details
            if hasattr(e, 'response') and e.response:
                 try: print(f"API Error Details: {e.response.json()}")
                 except json.JSONDecodeError: print(f"API Error Raw Response: {e.response.text}")
            return "Sorry, I encountered an unexpected error trying to analyze the image."

    return "Sorry, failed to get vision response after retries." # Fallback


# --- Formatting Function ---
# (Function format_for_whatsapp remains the same)
def format_for_whatsapp(text):
    if not isinstance(text, str): return text
    text = re.sub(r'```(\w*)\n(.*?)\n```', r'```\2```', text, flags=re.DOTALL)
    text = re.sub(r'\*\*(.*?)\*\*', r'*\1*', text)
    text = re.sub(r'_(.*?)_', r'_\1_', text)
    text = re.sub(r'~~(.*?)~~', r'~\1~', text)
    text = re.sub(r'^\s*#+\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*>\s+', '> ', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[\*\-]\s+', '* ', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', lambda m: m.group(0).strip() + ' ', text, flags=re.MULTILINE)
    text = re.sub(r'\\\[.*?\\\]', '', text, flags=re.DOTALL)
    text = re.sub(r'\\\(.*?\\\)', '', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = text.strip()
    print(f"Formatted Text: {text}")
    return text

# --- Main Logic ---

# Processes TEXT messages using history
def process_incoming_message(sender_number, message_text):
    print(f"\nProcessing TEXT message from {sender_number}: '{message_text}'")
    history_data = load_history()
    add_message_to_history(sender_number, "user", message_text, history_data)
    current_contact_history = get_contact_history(sender_number, history_data)
    raw_ai_reply = asyncio.run(get_ai_response(message_text, current_contact_history))  # Uses text model
    if raw_ai_reply:
        formatted_reply = format_for_whatsapp(raw_ai_reply)
        success = send_whatsapp_message(sender_number, formatted_reply)
        if success: add_message_to_history(sender_number, "assistant", formatted_reply, history_data)
        else: print(f"Failed to send formatted reply to {sender_number}. Response not added to history.")
    else: print("No AI reply generated.")
    save_history(history_data)

# Processes IMAGE messages (no history persistence for images yet)
def process_incoming_image(sender_number, caption, image_url):
    print(f"\nProcessing IMAGE message from {sender_number}. Caption: '{caption}', URL: {image_url}")

    # Load history
    history_data = load_history()

    # Save the user's image message as multimodal content (list of dicts)
    multimodal_content = [
        {"type": "text", "text": caption},
        {"type": "image_url", "image_url": {"url": image_url}}
    ]
    if sender_number not in history_data:
        history_data[sender_number] = []
    timestamp = datetime.now(timezone.utc).isoformat()
    history_data[sender_number].append({
        "role": "user",
        "content": multimodal_content,
        "timestamp": timestamp
    })
    # Trim history to max length
    history_data[sender_number] = history_data[sender_number][-MAX_HISTORY_PER_CONTACT:]

    # Get response from vision model
    raw_ai_reply = get_ai_vision_response(caption, image_url)  # Uses vision model

    if raw_ai_reply:
        formatted_reply = format_for_whatsapp(raw_ai_reply)
        success = send_whatsapp_message(sender_number, formatted_reply)
        if success:
            # Save assistant's reply as text
            timestamp = datetime.now(timezone.utc).isoformat()
            history_data[sender_number].append({
                "role": "assistant",
                "content": formatted_reply,
                "timestamp": timestamp
            })
            # Trim history
            history_data[sender_number] = history_data[sender_number][-MAX_HISTORY_PER_CONTACT:]
        else:
            print(f"Failed to send vision reply to {sender_number}.")
    else:
        print("No AI vision reply generated or error occurred. Not notifying user.")
        # Do not send any WhatsApp message on error

    # Save updated history
    save_history(history_data)


# --- Flask Webhook Setup ---
app = Flask(__name__)
PORT = 5000

@app.route('/webhook', methods=['POST'])
def whatsapp_webhook():
    global active_requests # Declare global at the top
    print("\n--- Webhook Received ---")
    if not request.is_json:
        print("Error: Request is not JSON")
        return jsonify({"status": "error", "message": "Request must be JSON"}), 400
    data = request.get_json()
    print("Raw Data:", json.dumps(data, indent=2))

    message_text = data.get('message')  # Can be caption for image/video
    sender_number = data.get('from')
    is_from_me = data.get('fromMe', True)
    message_type = data.get('type')
    attachment_url = data.get('attachmentUrl')  # Get image/video URL

    # Ignore outgoing messages
    if is_from_me:
        print("Webhook ignored: Message is from me (outgoing)")
        return jsonify({"status": "ignored", "message": "Outgoing message"}), 200

    # Blocked user check
    blocked_users = load_blocked_users()
    if sender_number in blocked_users:
        print(f"Blocked user {sender_number} attempted to send a message. Ignored.")
        return jsonify({"status": "blocked", "message": "You are blocked from using this service."}), 403

    # Per-user spam throttle check
    if is_user_spamming(sender_number):
        print(f"User {sender_number} is sending messages too quickly.")
        send_whatsapp_message(sender_number, "You are sending messages too quickly. Please slow down.")
        return jsonify({"status": "throttled", "message": "Too many messages in a short time."}), 429

    # Rate limit check
    if is_user_rate_limited(sender_number):
        print(f"User {sender_number} exceeded daily request limit.")
        # Inform user via WhatsApp
        send_whatsapp_message(sender_number, "You have reached your daily limit of 30 messages. Please try again tomorrow.")
        return jsonify({"status": "rate_limited", "message": "Daily request limit reached. Try again tomorrow."}), 429

    # --- Queueing Logic ---
    with queue_lock:
        # global active_requests # Removed from here
        if active_requests < MAX_CONCURRENT_REQUESTS:
            active_requests += 1
            # Pass arguments to the thread target using lambda
            thread_target = lambda: process_request(sender_number, message_type, message_text, attachment_url)
            threading.Thread(target=thread_target).start()
            return jsonify({"status": "accepted", "message": "Your request is being processed"}), 200
        else:
            # Queue the request
            queue_position = len(request_queue) + 1
            estimated_wait = queue_position * AVERAGE_PROCESSING_TIME
            # Store a lambda with arguments in the queue
            queued_target = lambda: process_request(sender_number, message_type, message_text, attachment_url)
            request_queue.append((queued_target, sender_number))
            print(f"Request from {sender_number} queued at position {queue_position}")
            return jsonify({
                "status": "queued",
                "message": f"Server is busy. You are number {queue_position} in the queue. Estimated wait time: {estimated_wait} seconds."
            }), 200
def process_request(sender_number, message_type, message_text, attachment_url):
    global active_requests # Declare global at the top
    try:
        # Enforce rate limit again at processing time
        if is_user_rate_limited(sender_number):
            print(f"User {sender_number} exceeded daily request limit (checked during processing).")
            send_whatsapp_message(sender_number, "You have reached your daily limit of 30 messages. Please try again tomorrow.")
            return

        if sender_number and message_type == 'TEXT' and message_text:
            print(f"Routing to TEXT processing for {sender_number}")
            process_incoming_message(sender_number, message_text)
        elif sender_number and message_type == 'IMAGE' and attachment_url:
            print(f"Routing to IMAGE processing for {sender_number}")
            caption = message_text if message_text else "Describe this image."
            process_incoming_image(sender_number, caption, attachment_url)
        else:
            print(f"Webhook ignored: Type '{message_type}' or missing data.")
    except Exception as e:
        print(f"Error processing message in webhook: {e}")
    finally:
        # Decrement active count and process next in queue
        with queue_lock:
            global active_requests
            active_requests -= 1
            if request_queue:
                next_func, next_sender = request_queue.popleft()
                active_requests += 1
                threading.Thread(target=next_func).start()
                # Optionally notify next_sender their request is now processing

# Removed duplicate/incorrect queuing logic block


# --- Run Flask App with pyngrok ---
# (if __name__ == "__main__": block remains the same)
if __name__ == "__main__":
    if not NGROK_AUTHTOKEN: print("ERROR: NGROK_AUTHTOKEN is missing in your .env file."); exit()
    public_url = None
    try:
        print(f"Starting ngrok tunnel for Flask app on port {PORT}...")
        public_url = ngrok.connect(PORT, "http").public_url
        print("---------------------------------------------------------------------")
        print(f" * ngrok tunnel established: {public_url}")
        print(f" * Use this URL for WhatsApp Callback: {public_url}/webhook")
        print("---------------------------------------------------------------------")
        print(f"Starting Flask server on host 0.0.0.0 port {PORT}...")
        app.run(host='0.0.0.0', port=PORT, debug=False)
    except Exception as e:
         print(f"Error starting ngrok or Flask: {e}")
         if public_url: ngrok.disconnect(public_url)
         ngrok.kill()
    finally:
        print("\nShutting down Flask server and ngrok tunnel...")
        ngrok.kill()
        print("ngrok tunnel process killed.")
