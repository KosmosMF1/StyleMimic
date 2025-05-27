import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

user_consent = {}

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
    """Обработка ответа пользователя"""
    query = update.callback_query
    user_id = query.from_user.id
    
    if query.data == "agree":
        user_consent[user_id] = True
        await query.answer("Спасибо! Теперь можете писать сообщения")
        await query.edit_message_text("✅ Согласие получено! Напишите что-нибудь")
    else:
        user_consent[user_id] = False
        await query.answer("Бот не будет обрабатывать ваши сообщения")
        await query.edit_message_text("❌ Вы не дали согласие на обработку данных")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка сообщений только при наличии согласия"""
    user_id = update.effective_user.id
    
    if not user_consent.get(user_id):
        await update.message.reply_text("⚠ Сначала дайте согласие через /start")
        return
    
    # модель будет здесь
    await update.message.reply_text(f"Вы написали: {update.message.text}")

def main():
    TOKEN = ""
    
    app = Application.builder().token(TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(handle_consent))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    app.run_polling()

if __name__ == "__main__":
    main()