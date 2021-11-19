# Основной Discord клиент для работы
# by Sergree
# https://github.com/sergree

from nextcord import Client, Game, MessageType, TextChannel
import collections
import random
from asyncio import sleep as asleep
import config
from core.tokenizer import Tokenizer
from utils.tprint import log
from core import predictor
from core.db import Database

class MainClient(Client):
    def __init__(self, **options):
        super().__init__(**options)
        if config.use_database:
            self.db = Database("bot.db")
        else:
            self.temperature = config.temperature
        self.tokenizer = Tokenizer()
        self.tokenizer.load_vocab_from_file(config.vocab_file)
        self.channel_deques = {}
        self.custom_emoji_collection = []

    @staticmethod
    def decision(probability):
        return random.random() < probability

    def load_custom_emoji_collection(self):
        self.custom_emoji_collection.clear()
        guilds = list(self.guilds)
        for guild in guilds:
            self.custom_emoji_collection.extend(guild.emojis)
        log("Коллекция кастомных emoji обновлена.")

    def random_emoji(self):
        return (str(random.choice(self.custom_emoji_collection)) if self.custom_emoji_collection else "")

    async def on_ready(self):
        log(f"Подключение к Discord успешно под пользователем @{self.user}.")
        self.load_custom_emoji_collection()
        game = Game(config.discord_game_name)
        await self.change_presence(activity=game)

    async def on_guild_join(self, guild):
        await self.wait_until_ready()
        log(f"Зашел на сервер {guild.name}.")
        self.load_custom_emoji_collection()

    async def on_guild_remove(self, guild):
        await self.wait_until_ready()
        log(f"Вышел с сервера {guild.name}.")
        self.load_custom_emoji_collection()

    async def on_guild_emojis_update(self, guild, before, after):
        await self.wait_until_ready()
        log(f"На сервере {guild.name} изменилась коллекция emoji.")
        if len(before) != len(after):
            self.load_custom_emoji_collection()

    async def handle_command(self, message):
        # Команда изменения температуры семплирования
        # Не стали использовать nextcord.ext.commands, т.к. это единственная команда на данный момент
        # Потом добавим, если потребуется
        if message.author.guild_permissions.administrator and message.content.startswith(config.command_temperature_change):
            mc_splitted = message.content.split()
            if len(mc_splitted) > 1:
                try:
                    temp = float(mc_splitted[1])
                except ValueError:
                    return True
                if temp <= 0:
                    return True
                if config.use_database:
                    self.db.set_guild_settings(message.guild.id, temp)
                else:
                    self.temperature = temp
                await message.channel.send(f"`temperature` ➡️ `{temp}`")
                return True
        return False

    async def on_message(self, message):
        await self.wait_until_ready()
        if (not isinstance(message.channel, TextChannel) or message.author.bot or message.type not in [MessageType.default, MessageType.reply]):
            return
        if not message.channel.permissions_for(message.guild.me).send_messages:
            return
        if message.channel.id not in self.channel_deques:
            self.channel_deques[message.channel.id] = collections.deque(maxlen=config.deque_max_len)
        self.channel_deques[message.channel.id].append(message)
        if await self.handle_command(message):
            return
        ref = False
        try:
            ref = await message.channel.fetch_message(message.reference.message_id)
            if ref.author == self.user:
                ref = True
        except:
            pass
        my_mention = self.user in message.mentions or ref
        if self.decision(config.no_mention_prob) or (my_mention and self.decision(config.mention_prob)):
            async with message.channel.typing():
                input_messages = self.channel_deques[message.channel.id]
                input_tensor = self.tokenizer.encode_input(input_messages, self.user)
                output_tensor = predictor.decode_sequence(input_tensor, self.temperature if not config.use_database else self.db.get_guild_settings(message.guild.id))
                output_message, token_count = self.tokenizer.decode_output(self, input_messages, output_tensor)
                if config.use_delay:
                    await asleep(random.uniform(0.1, 0.2) * token_count)
                if output_message:
                    await message.channel.send(output_message[:2000])
