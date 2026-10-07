import os
import pymysql
from dotenv import load_dotenv

load_dotenv()


# ==================== 数据库配置 ====================

DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "unity_ai_agent")


# ==================== 获取数据库连接 ====================

def get_db_connection():
    return pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True
    )


# ==================== 测试数据库连接 ====================

def test_db_connection():
    connection = None

    try:
        connection = get_db_connection()

        with connection.cursor() as cursor:
            cursor.execute("SELECT 1 AS result")
            row = cursor.fetchone()

        print("[DB] MySQL 连接成功")
        print(f"[DB] 测试结果：{row}")

        return True

    except Exception as e:
        print(f"[DB] MySQL 连接失败：{e}")
        return False

    finally:
        if connection:
            connection.close()


# ==================== 用户相关操作 ====================
# 创建用户
def create_user(user_id: str, password_hash: str) -> bool:
    connection = None

    try:
        connection = get_db_connection()

        with connection.cursor() as cursor:
            sql = """
                INSERT INTO users (username, password_hash)
                VALUES (%s, %s)
            """
            cursor.execute(sql, (user_id, password_hash))

        return True

    except pymysql.err.IntegrityError:
        # username 唯一约束冲突
        return False

    finally:
        if connection:
            connection.close()


# 根据用户名查询密码哈希。
# 用户不存在时返回 None。
def get_user_password_hash(user_id: str):
    connection = None

    try:
        connection = get_db_connection()

        with connection.cursor() as cursor:
            sql = """
                SELECT password_hash
                FROM users
                WHERE username = %s
            """
            cursor.execute(sql, (user_id,))
            row = cursor.fetchone()

        if row is None:
            return None

        return row["password_hash"]

    finally:
        if connection:
            connection.close()


# 判断用户是否存在。
def user_exists(user_id: str) -> bool:
    connection = None

    try:
        connection = get_db_connection()

        with connection.cursor() as cursor:
            sql = """
                SELECT id
                FROM users
                WHERE username = %s
                LIMIT 1
            """
            cursor.execute(sql, (user_id,))
            return cursor.fetchone() is not None

    finally:
        if connection:
            connection.close()


# 判断用户是否是管理员
def is_user_admin(user_id: str) -> bool:
    connection = None
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            sql = "SELECT is_admin FROM users WHERE username = %s LIMIT 1"
            cursor.execute(sql, (user_id,))
            row = cursor.fetchone()
        if row is None:
            return False
        return bool(row["is_admin"])
    except Exception as e:
        print(f"[db] 查询管理员失败：{e}")
        return False
    finally:
        if connection:
            connection.close()


if __name__ == "__main__":
    test_db_connection()