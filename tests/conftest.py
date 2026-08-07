"""
Общая инфраструктура для юнит-тестов blender headless-пайплайна.

bpy недоступен из системного python (мы ставили полноценный
Blender.app, не pip-пакет bpy) — поэтому каждый тест шеллится в
Blender headless как сабпроцесс, запуская соответствующий прототип-
скрипт из blender/, и затем проверяет числа из его *_result.json
СВОИМИ assert'ами (а не просто доверяя "PASS" из самого скрипта —
иначе баг в PASS-логике скрипта останется незамеченным).
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
BLENDER_DIR = PROJECT_ROOT / "blender"
BLENDER_BIN = "/Applications/Blender.app/Contents/MacOS/Blender"


def run_blender_script(script_name: str, timeout: int = 120) -> dict:
    """Запустить blender/<script_name> headless и вернуть его *_result.json.

    Падает с понятной ошибкой, если бинарник Blender недоступен, если
    процесс завершился с ненулевым кодом (сам скрипт уже делает
    sys.exit(1) при провале своей внутренней проверки), или если
    результирующий JSON не нашёлся/не распарсился.
    """
    script_path = BLENDER_DIR / script_name
    assert script_path.exists(), f"prototype script not found: {script_path}"

    result_path = BLENDER_DIR / (script_path.stem + "_result.json")
    if result_path.exists():
        result_path.unlink()  # чтобы не подхватить старый результат при падении рендера

    proc = subprocess.run(
        [BLENDER_BIN, "--background", "--python", str(script_path)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    assert proc.returncode == 0, (
        f"{script_name} exited with code {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout[-4000:]}\n"
        f"--- stderr ---\n{proc.stderr[-4000:]}"
    )
    assert result_path.exists(), f"{script_name} did not produce {result_path.name}"

    return json.loads(result_path.read_text())


@pytest.fixture(scope="session", autouse=True)
def _check_blender_available():
    if not Path(BLENDER_BIN).exists():
        pytest.exit(f"Blender not found at {BLENDER_BIN} — is it installed?", returncode=1)
