import secrets
import string


def generate_appid(length: int = 16) -> str:
    """
    生成 appid
    """
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length)).upper()


def generate_secret_key(length: int = 32) -> str:
    """
    生成 secret_key
    """
    return secrets.token_urlsafe(length)