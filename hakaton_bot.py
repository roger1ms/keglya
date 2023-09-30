import telebot



bot = telebot.TeleBot('6351695121:AAGx8Fj4MJYAzYcDvCoy4NemhG3QkhKIboA')


def text_transmission(message):
    print(message.text)





@bot.message_handler(commands=['start'])
def start(message):
    text = f'Здравствуйте, {message.from_user.first_name}, я чат-бот компании  Smart Consulting,' \
           f' могу помочь с любым вопросом! Что вы хотите узнать?'
    bot.send_message(message.chat.id, text)


@bot.message_handler()
def choice(message):
    text_transmission(message)
    return_message




bot.polling()


