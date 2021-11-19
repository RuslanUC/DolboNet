from sqlite3 import connect
import config

class Database:
    def __init__(self, filename):
        self.db = connect(filename, isolation_level=None)
        self.cur = self.db.cursor()
        self.cur.execute(f"""
            CREATE TABLE IF NOT EXISTS `settings` (
                `guild` bigint(20) NOT NULL UNIQUE,
                `temperature` double NOT NULL DEFAULT {config.temperature}
            )
            """)
        self._cache = {}

    def put_guild_settings(self, guild_id, temperature):
        self.cur.execute(f"INSERT OR IGNORE INTO `settings` values ({guild_id}, 0);")
        self.cur.execute(f"UPDATE `settings` SET `temperature`={temperature} WHERE `guild`={guild_id};")
        self._cache[guild_id] = temperature

    def set_guild_settings(self, *args, **kwargs):
        self.put_guild_settings(*args, **kwargs)

    def get_guild_settings(self, guild_id):
        if guild_id in self._cache:
            return self._cache[guild_id]
        temp = self.cur.execute(f"SELECT `temperature` FROM `settings` WHERE `guild`={guild_id}")
        temp = list(temp)
        if temp:
            temp = temp[0]
        else:
            temp = config.temperature
        self._cache[guild_id] = temp
        return temp