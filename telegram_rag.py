from query_index import find_related_text_parts
import telebot, os
from openai import OpenAI

api_key = os.getenv("TELEGRAM_API_KEY")
bot = telebot.TeleBot(api_key)

model = "qwen/qwen-3.6-35b-a3b"

client = OpenAI(
    base_url="http://localhost:1234/v1/",  # LM Studio endpoint
    api_key="not-needed",       
)

prompt = """
You are a helpful assistant that firstly thinks about the question and then answer questions about the user's query.
Always write your steps.
Answer in the same language as the user's query.
Answer in concise manner.

Steps example:


Example:
1. First I find out what technology is used in HyperRelay.
2. In the document it is stated that HyperRelay is powered by the VoidCore kernel.
3. Therefore, the answer is VoidCore. 

Answer: VoidCore.

Question: Hello
Steps:
1. It's not a question, it's a greeting.
2. Therefore, I should answer "Please provide a question".

Answer: Please provide a question.

# Context:
Use the following chunks to answer the question:
{chunks}

# Rules:
If you don't know the answer, say "I don't know".
If the question is not related to the chunks, say "I don't know".
If the question is not a question, say "Please provide a question".
"""

def get_response(message: str) -> str:
    text_parts = find_related_text_parts(message)

    parts = ''
    for text_part in text_parts:
        parts += text_part.text + '\n'

    result = prompt.format(chunks=parts)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "developer", "content": result},
            {"role": "user", "content": message}
        ]
    )
    return response.choices[0].message.content

@bot.message_handler(commands=['start'])
def start(message):
    bot.reply_to(message, "Hello, how can I help you today?")

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    response = get_response(message.text)
    bot.reply_to(message, response)

def stdio_handler():
    while True:
        message = input("Enter a message: ")
        response = get_response(message)
        print(response)

# bot.infinity_polling()
stdio_handler()