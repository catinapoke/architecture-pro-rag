# Update Index Process

Ниже диаграмма процесса инкрементального обновления индекса с `IndexIDMap2` и файловыми метаданными чанков.

```plantuml
@startuml
start

:Cron triggers scripts/update_index_cron.sh;
:Run python3 update_index.py;

:Load chunks_data.json (schema v2) or create empty metadata;
:Load index.faiss as IndexIDMap2 or create empty index;
:Scan knowledge_base and compute file sha256;
:Detect new / changed / removed files;

if (Any changes?) then (no)
  :Log no-op result;
  stop
else (yes)
  :Remove old chunk ids for changed and removed files;
  :Process changed and new files;
  :Split text into chunks;
  :Run safety check for each chunk;
  :Generate embeddings and deterministic chunk ids;
  :Add vectors with add_with_ids(...);
  :Update files[file].chunks metadata;
  :Write index.faiss.tmp and chunks_data.json.tmp;
  :Atomically replace final files with os.replace(...);
  :Log added and removed chunk ids;
  stop
endif

@enduml
```

## Cron usage

- Скрипт запуска: `scripts/update_index_cron.sh`
- Лог: `logs/update_index.log`
- Пример cron-правила (ежедневно в 06:00):
  - `0 6 * * * /absolute/path/to/architecture-pro-rag/scripts/update_index_cron.sh`
