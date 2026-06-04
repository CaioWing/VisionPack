from visionpack.index.json_index import JsonIndex
from visionpack.index.sqlite_index import SqliteIndex

# SqliteIndex is the default backend; JsonIndex remains for legacy migration.
Index = SqliteIndex

__all__ = ["Index", "JsonIndex", "SqliteIndex"]
