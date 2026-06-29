# Суммаризация — быстрая эволюция (полный пример для TUI)

> **Важно:** цепочки из Chat/CARL с `run_python`, host-путями или зашитым текстом одного примера **автоматически адаптируются** перед submit (`platform_chain_adapter`). Для суммаризации Platform подаёт каждую строку CSV как `outer_context`. Явный Platform-seed — `examples/summarizer/chain.json`.

Готовый seed уже лежит в Memory:

| | |
|---|---|
| **Имя (Chat, устаревший для Platform)** | `easy-summarizer-seed` |
| **Platform seed (рекомендуется)** | `examples/summarizer/chain.json` — один LLM-шаг, `$outer_context` |
| **chain_id (Memory, может быть Chat-версия)** | `a3d8fee6-ac11-4b38-abcd-d8737e5e9bf7` |
| **Задача** | Суммаризовать короткий абзац в одно предложение |
| **Датасет** | `examples/summarizer/eval.jsonl` (8 кейсов) |

### Platform-совместимый seed (импорт в Memory)

```bash
# из каталога care — сохранить chain.json в Memory и получить chain_id
uv run care validate examples/summarizer/chain.json
# затем через TUI: /library → Import или сохраните chain.json как новую цепочку
```

Минимальная схема: **один `llm`-шаг**, поля `aim` + `stage_action`, без `run_python` и без хостовых путей. Platform подаёт каждую строку CSV как `outer_context` (колонки кроме `expected`).

---

## Шаг 0 — проверить, что сервисы живы

```bash
uv run care doctor
```

Нужны ✓ **memory** и ✓ **mage**. Platform: укажите `CARE_PLATFORM__BASE_URL=http://localhost:8000` (master-api). Если в конфиге остался старый `:8001` (runner-api), CARE сам перенаправит upload/create на master — но лучше поправить URL. Проверка: `curl -s http://localhost:8000/health`.

При **первом** построении Platform-фасада MAESTRO автоматически:

- пишет `../gigaevo-platform/llm_models.yml` из вашего `config.toml`;
- копирует `tools.comparison` / `tools.redis2pd` в runner (если Docker-контейнер запущен).

Ручной чеклист (`sync_platform_llm_models.py`, `sync_runner_gigaevo_tools.sh`) нужен только если Platform на **удалённом** хосте или вы отключили авто-bootstrap: `CARE_PLATFORM__AUTO_BOOTSTRAP=0`.

При **Launch** эволюции MAESTRO автоматически накладывает актуальные `helper.py` / `validate.py` на папку `exp_*` в runner.

---

## Шаг 1 — запустить TUI

```bash
make run
# или: uv run care
```

---

## Шаг 2 — (опционально) прогнать seed вручную

**Вариант A — новая задача с текстом в чате** (режим «Генерация агентной цепочки»):

```
Суммаризируй в одно предложение: Климатические изменения ускоряют таяние ледников. Это повышает уровень моря и угрожает прибрежным городам.
```

**Вариант B — повторный прогон сохранённой цепочки:**

1. `/library` → выделите `easy-summarizer-seed` (или id `a3d8fee6-…`)
2. Нажмите **Run** (или `R` на экране просмотра)
3. Введите задачу в поле «Описание задачи» и подтвердите

> `/run <chain_id>` только **открывает просмотр** цепочки — для выполнения используйте Run из библиотеки.

Ожидайте одно предложение-суммари в ответе.

---

## Шаг 3 — загрузить датасет в Memory (для /dataset run)

Скопируйте команды **по одной** в чат (режим Interactive подходит):

```
/dataset add a3d8fee6-ac11-4b38-abcd-d8737e5e9bf7 "Климатические изменения ускоряют таяние ледников. Это повышает уровень моря и угрожает прибрежным городам." --expected "Таяние ледников из-за изменения климата повышает уровень моря и угрожает прибрежным городам."
```

```
/dataset add a3d8fee6-ac11-4b38-abcd-d8737e5e9bf7 "Компания выпустила новый смартфон с улучшенной камерой и батареей на два дня." --expected "Компания представила смартфон с лучшей камерой и двухдневной батареей."
```

```
/dataset add a3d8fee6-ac11-4b38-abcd-d8737e5e9bf7 "Исследование показало, что тридцать минут ходьбы в день снижают риск сердечных заболеваний." --expected "Ежедневная тридцатиминутная ходьба снижает риск сердечных заболеваний."
```

```
/dataset add a3d8fee6-ac11-4b38-abcd-d8737e5e9bf7 "Город открыл новую велосипедную дорожку вдоль реки. Власти надеются, что это уменьшит пробки в центре." --expected "Новая велосипедная дорожка у реки должна снизить пробки в центре города."
```

```
/dataset add a3d8fee6-ac11-4b38-abcd-d8737e5e9bf7 "Учёные обнаружили новый вид жуков в тропическом лесу. Находка поможет лучше понять биоразнообразие региона." --expected "Обнаружение нового вида жуков в тропическом лесу расширяет знания о биоразнообразии региона."
```

```
/dataset add a3d8fee6-ac11-4b38-abcd-d8737e5e9bf7 "Школа ввела обязательные уроки программирования для пятиклассников. Родители в целом поддержали инициативу." --expected "Школа сделала программирование обязательным для пятиклассников, и родители в основном одобрили это."
```

```
/dataset add a3d8fee6-ac11-4b38-abcd-d8737e5e9bf7 "Местная библиотека продлила часы работы по выходным. Посетителей стало заметно больше." --expected "Продление работы библиотеки по выходным привело к росту числа посетителей."
```

```
/dataset add a3d8fee6-ac11-4b38-abcd-d8737e5e9bf7 "Фермеры сообщают о рекордном урожае яблок после дождливого лета. Цены на фрукты могут немного снизиться." --expected "Рекордный урожай яблок после дождливого лета может немного снизить цены на фрукты."
```

Проверка:

```
/dataset list a3d8fee6-ac11-4b38-abcd-d8737e5e9bf7
```

Оценка baseline на всех кейсах:

```
/dataset run a3d8fee6-ac11-4b38-abcd-d8737e5e9bf7
```

Экспорт (если нужен отдельный файл):

```
/dataset export a3d8fee6-ac11-4b38-abcd-d8737e5e9bf7 /home/volkova/care/examples/summarizer/eval-from-memory.jsonl
```

---

## Шаг 4 — запустить эволюцию из TUI (основной путь)

### 4a. Открыть библиотеку

```
/library
```

Найдите строку **`easy-summarizer-seed`**, выделите её.

### 4b. Открыть форму эволюции

Нажмите **`E`** (*Evolve with my data*).

### 4c. Заполнить форму

| Поле | Что ввести |
|------|------------|
| **Базовая цепочка** | `a3d8fee6-ac11-4b38-abcd-d8737e5e9bf7` (обычно уже подставлен) |
| **Датасет (путь к JSONL)** | `/home/volkova/care/examples/summarizer/eval.jsonl` |
| **Критерии** | `Ответ — одно предложение; смысл совпадает с expected; без лишних деталей.` |
| **Максимум итераций** | `3` |
| **Размер популяции** | `4` |
| **Максимум времени** | *(оставить пустым)* |

Нажмите **Launch** / **Запуск**.

### 4d. Следить за прогоном

На экране эволюции — вкладки Fitness / Pareto / Events.

Или из чата (после старта CARE напишет `evolution_id`):

```
/evolution watch <run_id>
```

Принять победителя: **`a`** на экране эволюции.

---

## Альтернатива — эволюция из чата (кнопка Evolve)

1. `/run a3d8fee6-ac11-4b38-abcd-d8737e5e9bf7 <любой абзац>`
2. После прогона нажмите кнопку **«Эволюционировать»** / **Evolve** под ответом.
3. В модалке укажите тот же путь к JSONL и параметры, что в шаге 4c.

---

## Альтернатива — Production с нуля

```
/mode production
```

Затем отправьте задачу (Enter):

```
Суммаризуй входной абзац (2–3 предложения) в одно грамматически корректное предложение, сохранив главную мысль. Вход: Климатические изменения ускоряют таяние ледников. Это повышает уровень моря и угрожает прибрежным городам.
```

CARE автоматически: сгенерирует цепочку → сохранит → baseline → (если Platform жива) эволюцию.

После save скопируйте новый `chain_id` и добавьте датасет командами из шага 3, подставив свой id.

---

## Устранение неполадок

### `404 Not Found` для `/api/v1/evolutions`

Текущая Platform принимает эволюцию только через **`POST /api/v1/experiments/chains`**, куда нужно передать **JSON цепочки** (`base_chain_content`). Старый маршрут `/api/v1/evolutions` больше не существует.

CARE перед submit автоматически подтягивает цепочку из Memory. **Перезапустите TUI** после обновления CARE.

### `gen: 0` / `failed` / `Permission denied: '/llm_models.yml'`

**Монтирование в compose уже есть** — см. `gigaevo-platform/docker-compose.runner-pool.generated.yml`:

```yaml
- ./llm_models.yml:/llm_models.yml:ro
```

Типичные причины сбоев:

1. **Устаревший `llm_models.yml`** (не совпадает с `[platform]` в CARE).
2. **Runner пересоздан вручную** без compose-mount (старый `recreate-care-runner.sh`).
3. **Эксперимент отменён** — `Esc` на экране эволюции вызывает `stop` на Platform.

**Починка (из каталога care):**

```bash
# 1. Синхронизировать llm_models.yml из ~/.config/care/config.toml
uv run python scripts/sync_platform_llm_models.py \
  --platform-dir ../gigaevo-platform

# 2. Пересоздать runner с правильными mount'ами
../gigaevo-platform/scripts/recreate-care-runner.sh

# 3. Проверка
curl -s http://localhost:8001/health
docker exec gigaevo-platform-runner-api-1-1 \
  /app/.venv/bin/python3 -c \
  "from common.llm_registry import load_llm_registry as l; print([m['id'] for m in l()['models']])"
# ожидайте: ['care-mutation', 'care-validation']
```

Затем запустите **новый** прогон эволюции (старый `exp_*` в статусе `cancelled` / `failed` не продолжить).

### `best fitness: 0.000` на seed при ROUGE-L

Три типичные причины (исправлены в свежих шаблонах Platform + MAESTRO):

1. **Старый `helper.py` в папке эксперимента** — цепочка не вызывает LLM (`get_response_with_retries` / пустой `prediction`). MAESTRO после Launch автоматически накладывает актуальные шаблоны на runner; вручную: `./scripts/sync_experiment_chain_templates.sh exp_<uuid>`.
2. **ROUGE HuggingFace на кириллице всегда ≈0** — даже при хорошем ответе. В `validate.py` включён word-level ROUGE для non-Latin текста.
3. **Ответ на английском при русском `expected`** — в промпте теперь «ответ на том же языке, что вход».

После фикса seed на `eval.jsonl` должен давать **~0.55–0.65** ROUGE-L. Если всё ещё 0 — запустите **новый** эксперимент (старый seed уже оценён с битыми шаблонами).


Runner передаёт в Hydra неверный ключ. В gigaevo-core лимит поколений — `max_mutants`, не `max_generations`.

**Быстрый фикс (из каталога care):**

```bash
./scripts/patch_platform_max_mutants.sh
```

После патча снова запустите эволюцию. Постоянно: попросите владельца checkout `gigaevo-platform` применить тот же патч в `runner_api/src/services/gigavolve_service.py` и пересобрать образ (`./scripts/patch_platform_max_mutants.sh --rebuild`).

---

## CLI (без TUI)

```bash
uv run care evolve a3d8fee6-ac11-4b38-abcd-d8737e5e9bf7 \
  --test-data-path /home/volkova/care/examples/summarizer/eval.jsonl \
  --iterations 3 \
  --population 4 \
  --validation-criteria "Ответ — одно предложение; смысл совпадает с expected." \
  --wait
```

---

## Формат eval.jsonl

Каждая строка — JSON-объект:

- **`input`** — вход для Platform при эволюции
- **`task`** — то же самое (для совместимости с `/dataset` в CARE)
- **`expected`** — эталонный ответ (колонка для судьи, метрика ROUGE-L)

Файл: [`eval.jsonl`](eval.jsonl)
