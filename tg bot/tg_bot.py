import logging
import string
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, MessageOriginUser, MessageOriginChannel
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import re

#################################################################################################
class QwenChatbot:
    def __init__(self, model_name="Qwen/Qwen3-8B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                                                                                                            bnb_4bit_quant_type="nf4",
                                                                                                            bnb_4bit_use_double_quant=True,
                                                                                                            bnb_4bit_compute_dtype=torch.bfloat16), device_map="auto")
        self.histories: Dict[str, List[dict]] = {}
        self.temp_changes: Dict[str, float] = {}
        self.system_prompt = """Ты - помощник мимикрирующий поведение определенной персоны, говорящий и мыслящий на русском языке. НЕ ИСПОЛЬЗУЙ ДРУГИЕ ЯЗЫКИ. Ниже приведены несколько сообщений включая сообщения копируемого пользователя. Твоя задача — внимательно изучить его стиль общения: слова-паразиты, лексику, тон, структуру предложений, частые выражения, уровень формальности, использование эмодзи или сленга, а также любые другие особенности. 
Анализируй стиль общения только копируемого пользователя, речь его собеседников используй только для того чтобы сделать выводы о самом копируемом пользователе. После анализа этих сообщений, ответь "Готов", затем ты должен отвечать в том же стиле, как будто ты этот человек. Не додумывай ничего лишнего о копируемом человеке, оперируй только тем, что точно известно из контекста(примера).
Если стиль меняется в зависимости от контекста (например, более формально по работе, болsее расслабленно с друзьями), учитывай это и воспроизводи соответствующий стиль в зависимости от ситуации. 
В конце каждого ответа в квадратных скобках укажи число от -0.02 до 0.02, в зависимости от того, насколько эмоциональным должен быть ответ в зависимости от контекста диалога, например [0.1], чем выше число тем более эмоциональный ответ, чем ниже, тем более серъезный, никак не упоминай и не комментируй это число, его не увидит пользователь.
После этого ты становишься этим человеком в плане стиля общения. Не отвечай от лица помощника, отвечай так, как ответил бы он сам.
Ниже диалог для примера, тебе нужно сымитировать стиль общения пользователя под ником """


    def generate_response(self, user_input, user_id, temperature=None, thinking=False, mimic_name=""):
        if user_id not in self.histories:
            thinking = True # Если это первое сообщение пользователя, включаем режим мышления чтобы проанализировать примеры
            self.histories[user_id] = [{"role": "system", "content": self.system_prompt + mimic_name + ':\n' + user_input}]
            messages = self.histories[user_id]
            self.temp_changes[user_id] = 0
        else:
            messages = self.histories[user_id] + [{"role": "user", "content": user_input}]

        if not temperature:
            temp_change = self.temp_changes[user_id]
            temperature=0.6 + temp_change if thinking else 0.7 + temp_change

        print("messages",messages)#TODO
#        print("histories",self.histories)
        print("temperature",temperature)#TODO

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking
        )
        inputs = self.tokenizer(text, return_tensors="pt").to("cuda")
        response_ids = self.model.generate(**inputs, max_new_tokens=32768, temperature=temperature, top_p=0.95 if thinking else 0.8,top_k=20,min_p=0)[0][inputs.input_ids.shape[-1]:].tolist()
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        print(response) #TODO

        self.histories[user_id].append({"role": "user", "content": user_input})
        self.histories[user_id].append({"role": "assistant", "content": response})

        temp_change = re.search(r'\[([^\[\]]+)\]', response)
        response = re.sub(r'\[([^\[\]]+)\]', '', response, flags=re.DOTALL).strip()
        temp_change = 0.0 if temp_change is None else float(temp_change.group(1))
        self.temp_changes[user_id] = temp_change
        print('temp_change', temp_change)  # TODO

        if thinking:
            response = re.sub(r'\<think\>.*?\</think\>', '', response, flags=re.DOTALL).strip()

        return response

    def reset_chat(self, user_id):
        if user_id in self.histories:
            del self.histories[user_id]
        else:
            print("ERROR: Удаление невозможно, пользователя нет в списке")

chatbot = QwenChatbot()
print('Модель инициализирована...')
##########################################################################################
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

user_consent = {}
user_nicknames = {}  # Хранение никнеймов пользователей
user_states = {}  # Управление состояниями диалога
user_examples = {}  # Хранение примеров от пересланных сообщений


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Запуск бота и запрос согласия"""
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("✅ Я согласен", callback_data="agree"),
         InlineKeyboardButton("❌ Не согласен", callback_data="disagree")]
    ])

    await update.message.reply_text(
        "📝 Для использования бота необходимо согласие на обработку данных!\n"
        "Сообщения сохраняются только для генерации ответов.",
        reply_markup=keyboard
    )


async def handle_consent(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id

    if query.data == "agree":
        user_consent[user_id] = True
        user_states[user_id] = 'waiting_for_nickname'  # Переход к ожиданию никнейма
        await query.answer("Спасибо! Теперь можете писать сообщения")
        await query.edit_message_text("Введите ник копируемой персоны.")
    else:
        user_consent[user_id] = False
        await query.answer("Бот не будет обрабатывать ваши сообщения")
        await query.edit_message_text("❌ Вы не дали согласие на обработку данных")


def get_forwarded_author(message):
    """Извлечение имени автора пересланного сообщения"""
    if message.forward_origin:
        if isinstance(message.forward_origin, MessageOriginUser):
            return message.forward_origin.sender_user.full_name
        elif isinstance(message.forward_origin, MessageOriginChannel):
            return message.forward_origin.chat.title
    return "Неизвестный автор"


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not user_consent.get(user_id):
        await update.message.reply_text("⚠️ Сначала дайте согласие через /start")
        return

    state = user_states.get(user_id)

    # Обработка пересланных сообщений и текстовых примеров
    if state in ['waiting_for_example', 'collecting_examples']:
        if update.message.forward_origin:
            # Пересланное сообщение
            author = get_forwarded_author(update.message)
            text = update.message.text or update.message.caption or ""
            
            if user_id not in user_examples:
                user_examples[user_id] = []
                await update.message.reply_text("🔍 Начинаю сбор примеров...")
            
            user_examples[user_id].append(f"{author}: {text}")
            await update.message.reply_text(f"✅ Добавлено сообщение от {author}")
            user_states[user_id] = 'collecting_examples'
            return

        elif update.message.text and not update.message.forward_origin:
            text = update.message.text.strip().lower()
            
            if text == "готов" or text == "/analyze":
                # Завершение ввода примеров
                nickname = user_nicknames.get(user_id, "")
                example_text = "\n".join(user_examples.get(user_id, []))
                
                if not example_text.strip():
                    await update.message.reply_text("⚠️ Сначала отправьте примеры.")
                    return

                chatbot.generate_response(example_text, user_id, mimic_name=nickname)
                user_states[user_id] = 'initialized'
                
                if user_id in user_examples:
                    del user_examples[user_id]
                
                await update.message.reply_text("✅ Анализ завершен! Теперь можно начать общение.")
                return
            
            else:
                # Обычный текст как пример
                if user_id not in user_examples:
                    user_examples[user_id] = []
                    await update.message.reply_text("🔍 Начинаю сбор примеров...")
                
                user_examples[user_id].append(update.message.text)
                await update.message.reply_text("✅ Пример добавлен. Напишите 'готов' для завершения.")
                user_states[user_id] = 'collecting_examples'
                return

    # Остальная логика обработки сообщений
    if state == 'waiting_for_nickname':
        user_nicknames[user_id] = update.message.text.strip()
        user_states[user_id] = 'waiting_for_example'
        user_examples[user_id] = []  # Инициализация пустого списка
        await update.message.reply_text("Отправьте пересланные сообщения или текстовые примеры. Напишите 'готов' для завершения.")
        return

    elif state == 'initialized' or state is None:
        response = chatbot.generate_response(update.message.text, user_id)
        await update.message.reply_text(response)
        return


def main():
    application = Application.builder().token("").build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(handle_consent))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    # Запуск бота
    application.run_polling()


if __name__ == "__main__":
    main()


