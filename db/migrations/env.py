from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from db.session import db_url
from db.tables import Base

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

safe_db_url = db_url.replace('%', '%%')
config.set_main_option("sqlalchemy.url", db_url)

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
target_metadata = Base.metadata


# Only include tables that are in the target_metadata
# See: https://alembic.sqlalchemy.org/en/latest/autogenerate.html#omitting-table-names-from-the-autogenerate-process
def include_name(name, type_, parent_names):
    if type_ == "table":
        return name in target_metadata.tables
    else:
        return True


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        include_name=include_name,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_schemas=True,                  # ← 추가(권장)
        version_table_schema="ai",             # ← target_metadata.schema 대신 명시
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    connection.exec_driver_sql("CREATE SCHEMA IF NOT EXISTS ai")
    connection.exec_driver_sql("SET search_path TO ai, public")
    with connectable.connect() as connection:
        row = connection.exec_driver_sql(
            "select current_database(), current_user, current_schema(), current_setting('search_path')"
        ).first()
        print("[ALEMBIC-CONN]",
            "db=", row[0],
            "user=", row[1],
            "schema=", row[2],
            "search_path=", row[3],
        )
        # ai 스키마에 보이는 테이블 개수도 확인
        cnt = connection.exec_driver_sql(
            "select count(*) from information_schema.tables where table_schema='ai'"
        ).scalar_one()
        print("[ALEMBIC-CONN] ai.tables.count =", cnt)


        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            #include_name=include_name,
            include_schemas=True,               # ✅ 스키마를 반영/비교
            version_table_schema='ai',
            compare_type=True,
            compare_server_default=True,
            include_object=lambda obj, name, type_, reflected, compare_to: (
                getattr(obj, "schema", None) == "ai"
                if type_ in {"table","column","index","unique_constraint","foreign_key_constraint","primary_key"}
                else True
            ),
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
