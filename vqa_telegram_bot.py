import logging
from io import BytesIO
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from telegram import Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
)
import speech_recognition as sr
from gtts import gTTS
import os
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize BLIP captioning model (CPU-friendly)
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Initialize VQA pipeline on CPU
vqa_pipeline = pipeline("visual-question-answering", model="Salesforce/blip-vqa-base", device=-1)

# Speech recognizer instance (global)
recognizer = sr.Recognizer()

# Store user-uploaded images
user_images = {}

# /start command
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Hi! I'm your Vision Assistant Bot (BLIP).\n"
        "Send me an image, then ask a question or describe what you want!\n"
        "You can also send voice messages."
    )

# /help command
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📸 Step 1: Send me an image\n"
        "❓ Step 2: Ask a question or request a description via text or voice message\n"
        "🤖 I'll respond with text and voice!"
    )

# Handle photo
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        photo_file = await update.message.photo[-1].get_file()
        photo_bytes = await photo_file.download_as_bytearray()
        image = Image.open(BytesIO(photo_bytes)).convert("RGB")

        user_images[update.effective_user.id] = image
        await update.message.reply_text("✅ Got the image! Now ask a question or request a description.")
    except Exception as e:
        logger.error(f"Error handling image: {e}")
        await update.message.reply_text("❌ Couldn't process the image. Try again.")

# Convert voice to text
async def voice_to_text(update: Update):
    voice = update.message.voice
    if not voice:
        return None

    file = await voice.get_file()
    ogg_path = "voice.ogg"
    wav_path = "voice.wav"
    await file.download_to_drive(ogg_path)

    try:
        audio = AudioSegment.from_ogg(ogg_path)
        audio.export(wav_path, format="wav")

        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            return recognizer.recognize_google(audio_data)
    except CouldntDecodeError:
        logger.error("❌ Could not decode OGG audio. Is ffmpeg installed?")
        return None
    except sr.UnknownValueError:
        logger.warning("⚠ Could not understand the voice message.")
        return None
    except sr.RequestError as e:
        logger.error(f"❌ Speech Recognition API error: {e}")
        return None
    finally:
        for f in [ogg_path, wav_path]:
            if os.path.exists(f):
                os.remove(f)

# Send TTS voice reply
async def send_voice_reply(update: Update, text: str):
    tts = gTTS(text=text, lang='en')
    tts.save("reply.mp3")
    await update.message.reply_voice(voice=open("reply.mp3", "rb"))
    os.remove("reply.mp3")

# Handle voice messages
async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    if user_id not in user_images:
        await update.message.reply_text("⚠ Please send me an image first.")
        return

    text = await voice_to_text(update)
    if not text:
        await update.message.reply_text("❌ I couldn't understand your voice.")
        return

    image = user_images[user_id]

    try:
        if text.endswith("?"):
            result = vqa_pipeline(image=image, question=text)
            answer = result[0]["answer"]
            await update.message.reply_text(f"🤖 Answer: {answer}")
            await send_voice_reply(update, answer)
        else:
            inputs = caption_processor(images=image, return_tensors="pt")
            out = caption_model.generate(**inputs)
            caption = caption_processor.decode(out[0], skip_special_tokens=True)
            await update.message.reply_text(f"🖼 Description: {caption}")
            await send_voice_reply(update, caption)
    except Exception as e:
        logger.error(f"Error generating voice response: {e}")
        await update.message.reply_text("❌ Couldn't generate a response.")

# Handle text questions
async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    question = update.message.text.strip()

    if user_id not in user_images:
        await update.message.reply_text("⚠ Please send an image first.")
        return

    image = user_images[user_id]
    try:
        if question.endswith("?"):
            result = vqa_pipeline(image=image, question=question)
            answer = result[0]["answer"]
            await update.message.reply_text(f"🤖 Answer: {answer}")
            await send_voice_reply(update, answer)
        else:
            inputs = caption_processor(images=image, return_tensors="pt")
            out = caption_model.generate(**inputs)
            caption = caption_processor.decode(out[0], skip_special_tokens=True)
            await update.message.reply_text(f"🖼 Description: {caption}")
            await send_voice_reply(update, caption)
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        await update.message.reply_text("❌ Couldn't generate a response.")

# Global error handler
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error(msg="Exception occurred:", exc_info=context.error)
    if isinstance(update, Update) and update.message:
        await update.message.reply_text("⚠ Something went wrong. Try again later.")

# Main function
def main():
    # Replace with your actual bot token
    BOT_TOKEN = "API_TOKEN"

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_error_handler(error_handler)

    logger.info("🚀 Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
