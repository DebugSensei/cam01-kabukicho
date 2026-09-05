"""looq — оффлайн-пайплайн CAM-01 Kabukicho.

Правило 2 CLAUDE.md: этапы изолированы и общаются только через файлы.
Схемы этих файлов описаны в docs/CONTRACTS.md и версионируются SCHEMA_VERSION.

SCHEMA_VERSION меняется ТОЛЬКО явным решением с записью в docs/DECISIONS.md.
Каждый артефакт несёт свою версию внутри себя:
  * json     -> ключ "schema_version" верхнего уровня
  * geojson  -> properties.schema_version в FeatureCollection
  * parquet  -> file-level metadata, ключ b"schema_version" + колонка schema_version
"""

__version__ = "0.1.0"

#: Версия контрактов артефактов. См. таблицу версий в docs/CONTRACTS.md.
SCHEMA_VERSION = "1"

#: Значение поля status в артефакте, записанном каркасом без реальной логики.
#: Любой verify обязан отвергнуть артефакт с этим статусом (правило 8).
STATUS_SKELETON = "skeleton"

#: Значение поля status в артефакте, записанном полноценным прогоном этапа.
STATUS_OK = "ok"

__all__ = ["__version__", "SCHEMA_VERSION", "STATUS_SKELETON", "STATUS_OK"]
