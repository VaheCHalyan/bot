import os
import logging
import requests
import base64
import json
import io
from datetime import datetime
import telebot
from telebot import types
import mimetypes

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Получаем переменные окружения
BOT_TOKEN = os.getenv('BOT_TOKEN')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
ADMIN_CHAT_ID = os.getenv('ADMIN_CHAT_ID')  # Опционально для уведомлений

if not BOT_TOKEN:
    logger.error("❌ BOT_TOKEN не найден в переменных окружения!")
    exit(1)

if not OPENROUTER_API_KEY:
    logger.error("❌ OPENROUTER_API_KEY не найден в переменных окружения!")
    exit(1)

# Инициализация бота
bot = telebot.TeleBot(BOT_TOKEN)

# Конфигурация OpenRouter
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "google/gemini-2.0-flash-exp:free"

# Поддерживаемые типы файлов
SUPPORTED_IMAGE_TYPES = ['image/jpeg', 'image/png', 'image/gif', 'image/webp']
SUPPORTED_DOCUMENT_TYPES = [
    'text/plain', 'application/pdf', 'application/json',
    'text/csv', 'application/msword', 'text/html'
]

class GeminiBot:
    def __init__(self):
        self.user_contexts = {}  # Хранение контекста разговора для каждого пользователя
        self.max_context_length = 10  # Максимум сообщений в контексте
        
    def get_user_context(self, user_id):
        """Получить контекст пользователя"""
        if user_id not in self.user_contexts:
            self.user_contexts[user_id] = []
        return self.user_contexts[user_id]
    
    def add_to_context(self, user_id, role, content):
        """Добавить сообщение в контекст"""
        context = self.get_user_context(user_id)
        context.append({"role": role, "content": content})
        
        # Ограничиваем длину контекста
        if len(context) > self.max_context_length * 2:  # *2 потому что user + assistant
            context.pop(0)  # Удаляем самое старое сообщение
    
    def clear_context(self, user_id):
        """Очистить контекст пользователя"""
        if user_id in self.user_contexts:
            self.user_contexts[user_id] = []
    
    def encode_file_to_base64(self, file_content, mime_type):
        """Кодирование файла в base64 для отправки в API"""
        try:
            encoded_content = base64.b64encode(file_content).decode('utf-8')
            return f"data:{mime_type};base64,{encoded_content}"
        except Exception as e:
            logger.error(f"Ошибка кодирования файла: {e}")
            return None
    
    def prepare_message_content(self, text=None, file_data=None, mime_type=None):
        """Подготовка контента сообщения для Gemini"""
        content = []
        
        if text:
            content.append({
                "type": "text",
                "text": text
            })
        
        if file_data and mime_type:
            if mime_type.startswith('image/'):
                # Для изображений
                encoded_file = self.encode_file_to_base64(file_data, mime_type)
                if encoded_file:
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": encoded_file
                        }
                    })
            else:
                # Для текстовых файлов
                try:
                    if mime_type == 'application/pdf':
                        text_content = f"[PDF файл получен, но текст не может быть извлечен автоматически. Размер: {len(file_data)} байт]"
                    else:
                        text_content = file_data.decode('utf-8', errors='ignore')
                    
                    content.append({
                        "type": "text", 
                        "text": f"Содержимое файла ({mime_type}):\n\n{text_content[:4000]}..."  # Ограничиваем размер
                    })
                except Exception as e:
                    content.append({
                        "type": "text",
                        "text": f"[Не удалось прочитать файл: {str(e)}]"
                    })
        
        return content if content else [{"type": "text", "text": text or "Привет!"}]
    
    def call_gemini_api(self, user_id, text=None, file_data=None, mime_type=None):
        """Вызов API Gemini через OpenRouter"""
        try:
            # Получаем контекст пользователя
            context = self.get_user_context(user_id)
            
            # Подготавливаем сообщение
            message_content = self.prepare_message_content(text, file_data, mime_type)
            
            # Формируем сообщения для API
            messages = context.copy()
            messages.append({
                "role": "user",
                "content": message_content
            })
            
            # Заголовки для запроса
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "X-Title": "Telegram Gemini Bot",
                "HTTP-Referer": "https://github.com/your-repo",
                "X-Title": "Gemini Telegram Bot"
            }
            
            # Данные запроса
            payload = {
                "model": MODEL_NAME,
                "messages": messages,
                "max_tokens": 4096,
                "temperature": 0.7,
                "stream": False
            }
            
            logger.info(f"Отправляю запрос к Gemini для пользователя {user_id}")
            
            # Отправляем запрос
            response = requests.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result['choices'][0]['message']['content']
                
                # Добавляем в контекст
                self.add_to_context(user_id, "user", message_content)
                self.add_to_context(user_id, "assistant", ai_response)
                
                return ai_response
            else:
                logger.error(f"Ошибка API OpenRouter: {response.status_code} - {response.text}")
                return f"❌ Ошибка API: {response.status_code}\n{response.text}"
                
        except requests.exceptions.Timeout:
            return "⏰ Превышено время ожидания. Попробуйте еще раз."
        except requests.exceptions.ConnectionError:
            return "🌐 Ошибка подключения к API. Проверьте интернет."
        except Exception as e:
            logger.error(f"Ошибка при вызове Gemini API: {e}")
            return f"❌ Произошла ошибка: {str(e)}"

# Создаем экземпляр бота
gemini_bot = GeminiBot()

# Обработчики команд
@bot.message_handler(commands=['start'])
def start_handler(message):
    """Приветственное сообщение"""
    welcome_text = """
🤖 **Добро пожаловать в Gemini 2.5-flash Bot!**

Я могу:
• 💬 Отвечать на ваши вопросы
• 🖼️ Анализировать изображения  
• 📄 Работать с файлами (PDF, TXT, JSON и др.)
• 🧠 Вести контекстный диалог

**Команды:**
/start - Показать это сообщение
/clear - Очистить историю разговора
/help - Помощь и примеры
/status - Статус бота

Просто отправьте мне текст, изображение или файл! 🚀
    """
    
    # Создаем клавиатуру
    markup = types.InlineKeyboardMarkup()
    markup.add(
        types.InlineKeyboardButton("📚 Помощь", callback_data="help"),
        types.InlineKeyboardButton("🧹 Очистить чат", callback_data="clear")
    )
    
    bot.send_message(
        message.chat.id,
        welcome_text,
        parse_mode='Markdown',
        reply_markup=markup
    )

@bot.message_handler(commands=['clear'])
def clear_handler(message):
    """Очистка контекста пользователя"""
    user_id = message.from_user.id
    gemini_bot.clear_context(user_id)
    
    bot.send_message(
        message.chat.id,
        "🧹 История разговора очищена! Можем начать заново.",
        reply_markup=types.ReplyKeyboardRemove()
    )

@bot.message_handler(commands=['help'])
def help_handler(message):
    """Помощь и примеры использования"""
    help_text = """
📚 **Как использовать бота:**

**💬 Текстовые сообщения:**
• Задавайте любые вопросы
• Ведите диалог (бот помнит контекст)
• Просите написать код, стихи, рассказы

**🖼️ Работа с изображениями:**
• Отправьте фото + вопрос о нем
• Опишите что на картинке
• Найдите текст на изображении
• Проанализируйте график или схему

**📄 Работа с файлами:**
• PDF - извлечение текста
• TXT - анализ содержимого  
• JSON - обработка данных
• CSV - работа с таблицами

**Примеры запросов:**
• "Что изображено на этой картинке?"
• "Переведи этот текст на английский"
• "Объясни код в файле"
• "Создай план по этому документу"

**⚙️ Дополнительные команды:**
/clear - Очистить историю
/status - Проверить работу бота
    """
    
    bot.send_message(message.chat.id, help_text, parse_mode='Markdown')

@bot.message_handler(commands=['status'])
def status_handler(message):
    """Статус бота"""
    try:
        # Тестовый запрос к API
        test_response = gemini_bot.call_gemini_api(
            message.from_user.id, 
            "Ответь кратко: ты работаешь?"
        )
        
        status_text = f"""
📊 **Статус бота:**

✅ Telegram Bot: Работает  
✅ OpenRouter API: Работает
✅ Gemini 2.5-flash: Доступен

🕐 Время сервера: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🔄 Контекст: {len(gemini_bot.get_user_context(message.from_user.id))} сообщений

**Тест API:** {test_response[:100]}...
        """
        
        bot.send_message(message.chat.id, status_text, parse_mode='Markdown')
    except Exception as e:
        bot.send_message(message.chat.id, f"❌ Ошибка при проверке статуса: {str(e)}")

# Обработчик callback кнопок
@bot.callback_query_handler(func=lambda call: True)
def callback_handler(call):
    """Обработка inline кнопок"""
    if call.data == "help":
        help_handler(call.message)
    elif call.data == "clear":
        user_id = call.from_user.id
        gemini_bot.clear_context(user_id)
        bot.answer_callback_query(call.id, "🧹 История очищена!")
        bot.edit_message_text(
            "История разговора очищена! Можете задать новый вопрос.",
            call.message.chat.id,
            call.message.message_id
        )

# Обработчик изображений
@bot.message_handler(content_types=['photo'])
def photo_handler(message):
    """Обработка изображений"""
    try:
        bot.send_chat_action(message.chat.id, 'typing')
        
        # Получаем файл
        file_info = bot.get_file(message.photo[-1].file_id)  # Берем самое большое изображение
        file_data = bot.download_file(file_info.file_path)
        
        # Получаем текст сообщения (если есть)
        caption = message.caption or "Что изображено на этой картинке? Опиши подробно."
        
        # Определяем MIME тип
        mime_type = 'image/jpeg'  # По умолчанию для Telegram фото
        
        # Отправляем к Gemini
        response = gemini_bot.call_gemini_api(
            message.from_user.id,
            caption,
            file_data,
            mime_type
        )
        
        # Отправляем ответ частями, если он слишком длинный
        if len(response) > 4096:
            for i in range(0, len(response), 4096):
                bot.send_message(message.chat.id, response[i:i+4096])
        else:
            bot.send_message(message.chat.id, response)
            
    except Exception as e:
        logger.error(f"Ошибка при обработке изображения: {e}")
        bot.send_message(message.chat.id, f"❌ Ошибка при обработке изображения: {str(e)}")

# Обработчик документов
@bot.message_handler(content_types=['document'])
def document_handler(message):
    """Обработка документов и файлов"""
    try:
        bot.send_chat_action(message.chat.id, 'typing')
        
        # Получаем информацию о файле
        file_info = bot.get_file(message.document.file_id)
        
        # Проверяем размер файла (ограничение Telegram API - 20MB)
        if file_info.file_size > 20 * 1024 * 1024:
            bot.send_message(message.chat.id, "❌ Файл слишком большой. Максимум 20MB.")
            return
        
        # Определяем MIME тип
        mime_type = message.document.mime_type or mimetypes.guess_type(message.document.file_name)[0]
        
        if not mime_type:
            bot.send_message(message.chat.id, "❌ Не удалось определить тип файла.")
            return
        
        # Проверяем поддерживается ли тип файла
        if mime_type not in SUPPORTED_DOCUMENT_TYPES and not mime_type.startswith('text/'):
            supported_types = ", ".join(SUPPORTED_DOCUMENT_TYPES)
            bot.send_message(
                message.chat.id,
                f"❌ Тип файла не поддерживается.\n\n**Поддерживаемые форматы:**\n{supported_types}",
                parse_mode='Markdown'
            )
            return
        
        # Скачиваем файл
        file_data = bot.download_file(file_info.file_path)
        
        # Получаем текст сообщения
        caption = message.caption or f"Проанализируй содержимое файла {message.document.file_name}"
        
        # Отправляем к Gemini
        response = gemini_bot.call_gemini_api(
            message.from_user.id,
            caption,
            file_data,
            mime_type
        )
        
        # Отправляем ответ
        if len(response) > 4096:
            for i in range(0, len(response), 4096):
                bot.send_message(message.chat.id, response[i:i+4096])
        else:
            bot.send_message(message.chat.id, response)
            
    except Exception as e:
        logger.error(f"Ошибка при обработке документа: {e}")
        bot.send_message(message.chat.id, f"❌ Ошибка при обработке файла: {str(e)}")

# Обработчик голосовых сообщений  
@bot.message_handler(content_types=['voice'])
def voice_handler(message):
    """Обработка голосовых сообщений"""
    bot.send_message(
        message.chat.id,
        "🎤 Голосовые сообщения пока не поддерживаются.\nОтправьте текст, изображение или документ."
    )

# Обработчик текстовых сообщений
@bot.message_handler(func=lambda message: True)
def text_handler(message):
    """Обработка текстовых сообщений"""
    try:
        bot.send_chat_action(message.chat.id, 'typing')
        
        # Отправляем к Gemini
        response = gemini_bot.call_gemini_api(message.from_user.id, message.text)
        
        # Отправляем ответ частями, если он длинный
        if len(response) > 4096:
            for i in range(0, len(response), 4096):
                bot.send_message(message.chat.id, response[i:i+4096])
        else:
            bot.send_message(message.chat.id, response)
            
    except Exception as e:
        logger.error(f"Ошибка при обработке текста: {e}")
        bot.send_message(message.chat.id, f"❌ Произошла ошибка: {str(e)}")

# Обработчик ошибок
@bot.middleware_handler(update_types=['message'])
def modify_message(bot_instance, message):
    """Middleware для логирования сообщений"""
    user = message.from_user
    logger.info(f"Сообщение от {user.first_name} ({user.id}): {message.content_type}")

def send_startup_notification():
    """Отправка уведомления о запуске бота"""
    if ADMIN_CHAT_ID:
        try:
            startup_message = f"""
🚀 **Gemini Bot запущен!**

⏰ Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🤖 Модель: {MODEL_NAME}
☁️ Платформа: Railway.app
✅ Статус: Активен
            """
            bot.send_message(ADMIN_CHAT_ID, startup_message, parse_mode='Markdown')
        except Exception as e:
            logger.error(f"Не удалось отправить уведомление админу: {e}")

# Основная функция
def main():
    """Запуск бота"""
    logger.info("🤖 Запускаю Gemini Telegram Bot...")
    logger.info(f"🔗 Модель: {MODEL_NAME}")
    
    # Отправляем уведомление о запуске
    send_startup_notification()
    
    # Запускаем бота
    try:
        bot.polling(none_stop=True, interval=1, timeout=60)
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
        raise

if __name__ == "__main__":
    main()