#!/bin/bash
# ──────────────────────────────────────────────────────────────
# download_bvi_rlv.sh — скачать 20 сцен BVI-RLV через AWS CLI
#
# Инструкция:
#   1. Залогиниться на ieee-dataport.org
#   2. My Account → скопировать AWS Access Key и Secret Key
#   3. Со страницы датасета скопировать S3 URI (кнопка "AWS")
#   4. Запустить:
#      IEEE_AWS_ACCESS_KEY=xxx IEEE_AWS_SECRET_KEY=yyy \
#      BVI_S3_URI=s3://... bash scripts/download_bvi_rlv.sh
#
# Установить AWS CLI если нет:
#   brew install awscli
# ──────────────────────────────────────────────────────────────

set -e

AWS_ACCESS_KEY="${IEEE_AWS_ACCESS_KEY:-ЗАМЕНИ_НА_СВОЙ}"
AWS_SECRET_KEY="${IEEE_AWS_SECRET_KEY:-ЗАМЕНИ_НА_СВОЙ}"
S3_BASE_URI="${BVI_S3_URI:-ЗАМЕНИ_НА_S3_URI}"
OUTPUT_DIR="$(dirname "$0")/../data/BVI-RLV"

# 20 сцен — при желании список можно менять
SCENES=(
  "S02_animals1"   "S03_animals2"   "S04_colour_sticks"
  "S05_bunnies"    "S06_lego"       "S07_hats"
  "S08_soft_toys"  "S09_kitchen"    "S10_messy_toy"
  "S11_gift_wrap"  "S12_toys"       "S13_books"
  "S14_plants"     "S15_desk"       "S16_food"
  "S17_sport"      "S18_outdoor1"   "S19_outdoor2"
  "S20_street"     "S21_nature"
)

# ── Проверки ───────────────────────────────────────────────────
if ! command -v aws &> /dev/null; then
  echo "❌ AWS CLI не установлен. Установи: brew install awscli"
  exit 1
fi
if [[ "$AWS_ACCESS_KEY" == "ЗАМЕНИ_НА_СВОЙ" ]]; then
  echo "❌ Укажи AWS ключи через env:"
  echo "   IEEE_AWS_ACCESS_KEY=... IEEE_AWS_SECRET_KEY=... bash scripts/download_bvi_rlv.sh"
  exit 1
fi
if [[ "$S3_BASE_URI" == "ЗАМЕНИ_НА_S3_URI" ]]; then
  echo "❌ Укажи S3 URI: BVI_S3_URI=s3://... bash scripts/download_bvi_rlv.sh"
  exit 1
fi

# ── AWS config ────────────────────────────────────────────────
aws configure set aws_access_key_id "$AWS_ACCESS_KEY"
aws configure set aws_secret_access_key "$AWS_SECRET_KEY"
aws configure set region us-east-1
mkdir -p "$OUTPUT_DIR"

echo ""
echo "📥 BVI-RLV Download"
echo "   Output : $OUTPUT_DIR"
echo "   Scenes : ${#SCENES[@]}"
echo ""

TOTAL=${#SCENES[@]}; DONE=0; FAILED=()

for SCENE in "${SCENES[@]}"; do
  DONE=$((DONE+1))
  echo "[$DONE/$TOTAL] $SCENE ..."
  S3_PATH="${S3_BASE_URI%/}/${SCENE}/"
  if aws s3 cp "$S3_PATH" "$OUTPUT_DIR/$SCENE" --recursive; then
    echo "  ✓ $SCENE"
  else
    echo "  ✗ $SCENE — ошибка"
    FAILED+=("$SCENE")
  fi
done

echo ""
echo "──────────────────────────────────────"
echo "Готово: $((TOTAL-${#FAILED[@]}))/$TOTAL сцен"
echo "Размер: $(du -sh "$OUTPUT_DIR" 2>/dev/null | cut -f1)"
[ ${#FAILED[@]} -gt 0 ] && echo "Не скачались: ${FAILED[*]}"
echo "──────────────────────────────────────"
