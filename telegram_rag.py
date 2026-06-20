from stdio_bot import get_response
import telebot, os

api_key = os.getenv("TELEGRAM_API_KEY")
bot = telebot.TeleBot(api_key)

@bot.message_handler(commands=['start'])
def start(message):
    bot.reply_to(message, "Hello, how can I help you today?")

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    response = get_response(message.text)
    bot.reply_to(message, response)

if __name__ == "__main__":
    bot.infinity_polling()