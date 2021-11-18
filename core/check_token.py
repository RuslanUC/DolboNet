from requests import get

def check_token(token):
    r = get("https://discord.com/api/v9/users/@me", headers={"Authorization": "Bot "+token})
    return r.status_code == 200