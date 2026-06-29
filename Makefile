.PHONY: run sync help test lint

LOG_DIR := logs
TIMESTAMP := $(shell date +%Y%m%d-%H%M%S)
UI_LOG := $(LOG_DIR)/care-ui-$(TIMESTAMP).log
APP_LOG := $(LOG_DIR)/care-app-$(TIMESTAMP).log

# `make run` launches the TUI. Pass `LOG=1` to also capture
# debug logs to `./logs`:
#
#   * care-ui-<timestamp>.log   — Textual UI events (compose,
#                                 mount, dispatch, render)
#                                 driven by TEXTUAL_LOG.
#   * care-app-<timestamp>.log  — Python app/client log
#                                 (care.* modules, httpx Memory
#                                 / Platform calls, MAGE / CARL
#                                 workers) driven by CARE_LOG_FILE
#                                 / CARE_LOG_LEVEL.
#
# Pass `LOG_LEVEL=DEBUG` to widen the app-log channel (default INFO).
# `PYTHONIOENCODING=utf-8` keeps non-ASCII output safe when LC_*
# is unset or set to `C`.
LOG_LEVEL ?= INFO

run: sync
ifeq ($(LOG),1)
	@mkdir -p $(LOG_DIR)
	@echo "UI  log -> $(UI_LOG)"
	@echo "App log -> $(APP_LOG) (level=$(LOG_LEVEL))"
	PYTHONIOENCODING=utf-8 \
	TEXTUAL_LOG=$(UI_LOG) \
	CARE_LOG_FILE=$(APP_LOG) \
	CARE_LOG_LEVEL=$(LOG_LEVEL) \
	uv run care
else
	PYTHONIOENCODING=utf-8 uv run care
endif

sync:
	uv sync

help:
	uv run care --help

test:
	uv run pytest

lint:
	uv run ruff check .
