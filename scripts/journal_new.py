"""Create journal/experiments/<E0NN>.qmd from the template if it does not exist (never overwrites).

    python scripts/journal_new.py E020 [--title "Короткое имя"]     (page title "E020 — Короткое имя")

The page includes the generated blocks journal/_auto/<E0NN>_{rule,results,checks}.md and has one
hand-written section, "Выводы", that no script ever touches. Run journal_build_auto.py afterwards.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGES = ROOT / "journal" / "experiments"

TEMPLATE = """---
title: "{title}"
---

{{{{< include ../_auto/{exp}_header.md >}}}}

## Вопрос и правило до запуска

{{{{< include ../_auto/{exp}_rule.md >}}}}

## Результаты

{{{{< include ../_auto/{exp}_results.md >}}}}

## Контроль и аномалии

{{{{< include ../_auto/{exp}_checks.md >}}}}

## Выводы

<!-- Ручной раздел: только дописывать (дата, что понял, ссылка на файл с числом). Генератор его не трогает. -->

Не заполнено.
"""


def main(argv: list[str]) -> int:
    short = None
    if len(argv) == 3 and argv[1] == "--title":
        short, argv = argv[2], argv[:1]
    if len(argv) != 1 or not re.fullmatch(r"E0\d\d", argv[0]):
        print("использование: python scripts/journal_new.py E0NN [--title \"имя\"]", file=sys.stderr)
        return 2
    exp = argv[0]
    page = PAGES / f"{exp}.qmd"
    if page.exists():
        print(f"{page.relative_to(ROOT)} уже есть: не перезаписываю")
        return 0
    folders = sorted(p for p in (ROOT / "experiments").glob(f"{exp}_*") if p.is_dir())
    if not folders:
        print(f"нет папки experiments/{exp}_*: сначала создайте эксперимент", file=sys.stderr)
        return 1
    readme = folders[0] / "README.md"
    title = readme.read_text(encoding="utf-8").splitlines()[0].lstrip("# ").strip() if readme.is_file() else exp
    if short:
        title = f"{exp} — {short}"
    PAGES.mkdir(parents=True, exist_ok=True)
    page.write_text(TEMPLATE.format(title=title.replace('"', "'"), exp=exp,
                                    folder=folders[0].relative_to(ROOT).as_posix()), encoding="utf-8")
    config = ROOT / "journal" / "_quarto.yml"
    marker = "      # journal_new.py appends new experiment pages above this line"
    text = config.read_text(encoding="utf-8")
    entry = f"      - experiments/{exp}.qmd\n"
    if entry not in text and marker in text:   # flat sidebar: add the page to the menu
        config.write_text(text.replace(marker, entry + marker), encoding="utf-8")
    print(f"создана {page.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
