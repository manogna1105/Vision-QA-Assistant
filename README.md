# 🧠 Vision-QA Assistant (Telegram Bot)

Vision-QA Assistant is a multimodal chatbot powered by a vision-language foundation model that answers questions about images via a Telegram interface. Users can send images and ask natural-language questions — either by typing or using voice — and receive intelligent answers or descriptive captions. The bot combines computer vision with language understanding to support real-time visual question answering (VQA).

## 🧠 Problem Statement

In an era where images carry a wealth of information, interpreting them intelligently in real time remains a challenge. Users often need quick insights from visual data — whether it's understanding trends in charts, recognizing objects in photos, or summarizing visual scenes. 

## ✨ Features

- 📸 Send an image and get automatic captions.
- ❓ Ask questions like “What is the person doing?” or “How many objects are there?”
- 🎙 Voice support — ask and receive answers via audio.
- 🔁 Multi-modal: Text + Image + Audio interaction.
- 🤖 Powered by [BLIP (Salesforce)](https://github.com/salesforce/BLIP) for captioning and VQA.

## Create a virtual environment

python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

## Install dependencies

pip install -r requirements.txt
Required libraries include:

* transformers
* torch
* Pillow
* pydub
* gtts
* speechrecognition
* python-telegram-bot
* ffmpeg (must be installed on your system for audio handling)

## Set your Telegram Bot Token

Replace "API_TOKEN" in vqa_telegram_bot.py with your bot's token from BotFather.

BOT_TOKEN = "your_telegram_bot_token"

## ▶️ Running the Bot

python vqa_telegram_bot.py

The bot will start polling Telegram for messages. You can now interact with it via your Telegram app.

## 📸 How to Use

1. Start the bot: /start
2. Send an image (photo or chart).
3. Ask a question about the image:
     * Text: “What is happening here?”
     * Voice: Say your question — it will be transcribed and answered!
4. The bot responds with both text and voice output.

## 🧱 Architecture Overview

* Telegram Bot Interface: Built with python-telegram-bot.
* Vision-Language Model: BLIP (Salesforce) via HuggingFace Transformers.
* Voice Interface: Speech-to-text via SpeechRecognition, text-to-speech via gTTS.
* Audio Handling: pydub and ffmpeg for audio conversion.

## 📌 Limitations

* Works best with general images (e.g., photos, simple graphs).
* Requires an internet connection for HuggingFace model downloads and voice APIs.
* Designed for CPU inference — not optimized for large-scale deployments.

## 🛠 Future Improvements

* 🧠 Swap in more powerful models like LLaVA or MiniGPT-4.
* 🌐 Host model and bot on cloud (Hugging Face Spaces, AWS, etc.).
* 🧩 Add multi-image comparison.
* 🧵 Multi-turn dialogue context support.

## 🤝 Contributions

PRs are welcome! Feel free to fork and enhance the assistant — whether by improving UX, using better models, or adding logging and metrics.








