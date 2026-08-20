.PHONY: report verify test lint all clean-report

report:            ## Собрать фигуры, таблицы и PDF из замороженных артефактов
	uv run python scripts/build_report_figures.py
	cd report && xelatex -interaction=nonstopmode report.tex >/dev/null
	cd report && bibtex report >/dev/null 2>&1 || true
	cd report && xelatex -interaction=nonstopmode report.tex >/dev/null
	cd report && xelatex -interaction=nonstopmode report.tex >/dev/null
	@echo "report/report.pdf готов"

verify:            ## Проверить, что числа в отчёте совпадают с артефактами
	uv run python scripts/verify_report_numbers.py

test:              ## Полный набор тестов
	uv run pytest -q

lint:              ## Ruff
	uv run ruff check src/ scripts/ tests/

all: report verify test lint
