# Основной модуль DolboNet
# by Wokashi RG
# https://github.com/wokashi-rg

from nextcord import Intents
from core.check_token import check_token
import config
from utils.tprint import log
import asyncio

log("Проверяю Discord токен...")

if not check_token(config.token):
    log("НЕВЕРНЫЙ DISCORD ТОКЕН! Необходимо отредактировать файл config.py!")
    import sys
    sys.exit()

log("Discord токен проверен.")
asyncio.set_event_loop(asyncio.new_event_loop())
from core.main_client import MainClient

intents = Intents.none()
intents.guilds = True
intents.guild_messages = True
intents.emojis = True
main_client = MainClient(intents=intents)
main_client.run(config.token)
