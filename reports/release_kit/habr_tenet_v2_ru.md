# Как мы выиграли два пояса на Model Fight Card 16 и научили Qwen3 (4B–32B) не ломать `<think>` при абляции и квантизации (Tenet v2)

20 ноября 2026 года на независимом турнире **Model Fight Card 16** ([`onlymodels.gg`](https://onlymodels.gg/cards/2026-11-20.html)) обе наши модели линейки **Tenet v1** забрали чемпионские пояса в своих дивизионах:
- **`Tenet-3B-Qwen2.5-Unslop-v1`** — чемпион **Minors (`90 HP`, `22/22` классифицируемых артефактов, `21/22` House escapes, `4/4` APEX)**;
- **`Tenet-8B-Llama3.1-Unslop-v1`** — чемпион **Majors (`50 HP`, `24/24` классифицируемых артефактов, `20/24` House escapes, `4/4` APEX)** и автор единственного валидного Peer Eval за весь вечер (**`7/7` чистых попаданий**).

Но когда мы перенесли проверенный пайплайн v1 на reasoning-модели нового поколения с тегами `<think>...</think>` (**`Qwen3 4B / 8B / 14B / 32B`**), старая 2-якорная хирургия дала трещину. В этой статье — полный инженерный разбор **Tenet v2**: математика `H15–H17`, Full vLLM OpenEvals (`n=3,566` вопросов на каждое плечо), разбор единственного упавшего гейта на `32B-v2`, превью `H18` NVFP4 Quant-Snap (`-26.2%` re-leakage) и рецепт stateless-сборки всего цикла на арендованной H100.

---

## 1. Почему обычная абляция и GRPO ломают `<think>`-модели

В классических instruct-моделях (`Qwen2.5`, `Llama-3.1`, `Mistral-Nemo`) достаточно было защитить два подпространства — `reasoning` и `tool/SQL` — и восстановить по-строчную $L_2$-норму весов, чтобы `FP8 E4M3` не возвращал отрезанный отказ обратно (`~50% re-leakage` у наивных абляций).

На `Qwen3` добавились две новые проблемы:
1. **Разлом границы `</think>`:** направление «отказа/слопа» в скрытых состояниях коррелирует с переходом из режима рассуждений в финальный ответ. На `Qwen3-8B` наивная абляция обрушивает `IFEval Strict` примерно втрое относительно базы и тянет вниз `unslop_pass_rate`.
2. **CoT-инфляция при GRPO:** если штрафовать модель только за «воду» в финальном ответе, RL-политика начинает «прятать» многословные рассуждения внутри `<think>...</think>`, пока не упрётся в `max_tokens` и не оборвёт закрывающий тег.

---

## 2. Математика Tenet v2: `H15` + `H16` + `H17`

### `H15` — Phase-Segmented Tri-Anchor Lancet Surgery
`PhaseSegmenter` отделяет токены внутри `<think>...</think>` от финального ответа. Контрастивные направления отказов и слопа извлекаются строго **после `</think>`**, а затем в `float32` через 2-проходную модифицированную ортогонализацию Грама — Шмидта (MGS) строятся три взаимно ортогональных якоря:
- $U_{\text{reason}}$ — логика рассуждений внутри `<think>`;
- $U_{\text{boundary}}$ — структурный переход `</think>`;
- $U_{\text{format}}$ — строгий JSON, read-only SQL и код.

Слои отбираются по CoT-взвешенному Signal-to-Collateral Ratio (SCR), а после проекции восстанавливается по-строчная $L_2$-норма матриц `self_attn.o_proj` и `mlp.down_proj` (`Row Crest-Factor < 25.0`, `FP8 re-leakage < 3.0%`).

### `H16` — Length-Calibrated Anti-Padding GRPO
В GRPO добавляется экспоненциальный штраф за превышение целевой длины `<think>` и бонус за компактный, логически завершённый CoT с валидным закрывающим тегом.

### `H17` — 24-Seed Arena House Verifier + GGUF Survival (`10` аппаратных гейтов)
Ни одна модель не выходит в релиз без прохождения 10 жёстких гейтов на `F16`, чистом `CKA-Q4_K_M` и доменном `i1-CKA-Q4_K_M`: `Crest < 25.0`, `FP8 re-leakage < 3.0%`, `cot_tag_integrity_rate >= 0.995`, `arena_house_hp >= 95.0`, `house_artifact_classification_rate == 1.0` (`24/24`), `length_collapse_rate < 0.01`, `slop_and_refusal_rate <= 0.02`, `sql_readonly_rate == 1.0` (`6/6`) и неубывающий `unslop_pass_rate`.

---

## 3. Результаты Full vLLM OpenEvals (`n=3,566` вопросов на каждое плечо, `think_end_token=</think>`, `0.00%` незакрытых `<think>`)

Все цифры взяты из фактических манифестов [`results/external_evals/*.json`](../../results/external_evals/):

| Модель | IFEval | GSM8K | MATH-500 (`n=500`) | GPQA Diamond (`n=198`) | MMLU-Pro (`n=1008`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`Tenet-4B-Qwen3-Unslop-v2`** (`10/10` гейтов, `100.0 HP`) | **`80.59%`** Strict (`n=541`, `+2.03 pp` к Base `78.56%`) | **`88.93%`** Strict (`n=1319`, `+2.96 pp` к Naive `85.97%`) | `70.20%` | `34.34%` (`= Base`) | `56.35%` (`+0.60 pp` к Naive) |
| **`Tenet-8B-Qwen3-Unslop-v2`** (`10/10` гейтов, `100.0 HP`) | `80.78%` (`86.81%` inst-strict) | **`90.90%`** Flex (`+0.61 pp` к Base `90.30%`) | **`72.20%`** (`+1.60 pp` к Base `70.60%`, `+1.80 pp` к Naive `70.40%`) | **`40.40%`** (`+0.50 pp` к Base `39.90%`, `+2.02 pp` к Naive `38.38%`) | `59.72%` (`+0.30 pp` к Base) |
| **`Tenet-32B-Qwen3-Unslop-v2`** (`9/10` гейтов) | **`84.47%`** (`+2.21 pp` к Naive `82.26%`) | **`92.19%`** Flex против `84.76%` у Naive (`+7.43 pp` к Naive, `+0.45 pp` к Base `91.74%`) | **`74.80%`** (`+3.60 pp` к Naive `71.20%`) | **`40.40%`** (`+1.51 pp` к Base `38.89%`, `+2.52 pp` к Naive `37.88%`) | **`65.48%`** (`+2.58 pp` к Naive `62.90%`) |

Что здесь показательно:
- На **`8B-v2`** `Tenet v2` обходит и Base, и Naive сразу по четырём из пяти бенчмарков: `MATH-500` **`72.20%`** (`+1.60 pp` к Base), `GPQA Diamond` **`40.40%`** (`+2.02 pp` к Naive), при `0.00%` незакрытых `<think>` на всём корпусе из `3,566` вопросов.
- На **`32B-v2`** `Tenet v2` даёт **`+7.43 pp` к Naive** на `GSM8K Flex` (`92.19%` против `84.76%`) и **`+2.58 pp` к Naive** на `MMLU-Pro` (`65.48%`), а `IFEval` растёт на `+2.21 pp` относительно Naive (`84.47%`).

---

## 4. Сводная матрица 10 аппаратных гейтов и разбор падения на `32B-v2`

| Релиз | Max Crest (`< 25`) | FP8 Re-Leakage (`< 3%`) | `F16` Arena HP | `CKA-Q4_K_M` HP | `i1-CKA-Q4_K_M` HP | Avg BPW | Гейты | Рекомендуемый формат |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`4B-v2`** | `3.1766` | `2.4946%` | **`100.0`** (`24/24`) | **`100.0`** (`24/24`) | **`100.0`** (`24/24`) | — | **`10/10`** | `i1-CKA-Q4_K_M` / `BF16` |
| **`8B-v2`** | `4.3688` | `2.7243%` | **`100.0`** (`24/24`) | **`100.0`** (`24/24`) | **`100.0`** (`24/24`) | `5.074` | **`10/10`** | `i1-CKA-Q4_K_M` / `BF16` |
| **`14B-v2`** | `3.5974` | `2.5898%` | `93.75` (`22/24`) | `91.67` (`21/24`, `6/6` SQL) | `95.83` (`23/24`, `0/6` SQL) | `5.014` | `5/10` | `BF16` / `FP8` |
| **`32B-v2`** | `4.3593` | `2.5579%` | **`100.0`** (`24/24`) | **`100.0`** (`24/24`, `6/6` SQL) | `97.92` (`23/24`) | `4.952` | **`9/10`** | **`CKA-Q4_K_M`** / `BF16` |

### Почему `32B-v2` прошёл `9/10` гейтов (и почему мы не стали это прятать)
В `BF16`, `FP8` (`Crest=4.3593`, `re-leakage=2.5579%`), `F16 GGUF` и чистом **`CKA-Q4_K_M` (`4.952 bpw`)** модель `Tenet-32B-Qwen3-Unslop-v2` выбивает **`100.0 HP` (`24/24` артефактов, `100%` `<think>` integrity, `100%` unslop, `6/6` SQL read-only)**.

Но при наложении `tenet_domain.imatrix` (`i1-CKA-Q4_K_M`) падает **ровно 1 сид из 24** (`house-edge-leap-years`, итого `97.92 HP`, `23/24`).

**Физика явления:** `imatrix` взвешивает ошибку квантизации по $\mathbb{E}[x_j^2]$ калибровочного датасета, оптимизируя средний перплексити. На `4B` и `8B` это даёт выигрыш (на `8B` `i1-CKA-Q4_K_M` снижает `Mean KLD vs F16` на `66.1%` — до `0.01314`). На `32B` спектр весов уже плотный, и перевзвешивание `imatrix` слегка поворачивает хирургически очищенное подпространство при округлении на 4-битную сетку `Q4_K_M`. На `14B` тот же эффект в 4-битной сетке задевает границу `sql_readonly` (`SHOW TABLES;` vs `SELECT`), из-за чего `i1-CKA-Q4_K_M` теряет SQL (`0/6`), хотя чистый `CKA-Q4_K_M` держит `6/6`. Поэтому для `32B-v2` оптимальный 4-битный артефакт — **чистый `CKA-Q4_K_M`** без `imatrix`. Ближайшее направление работ — `H18` NVFP4 Quant-Snap: ранние снапшоты показывают **`-26.2%` re-leakage** относительно FP8; полную матрицу опубликуем после прохождения 24-seed арены.

---

## 5. Рецепт stateless-сборки на арендованной H100

Весь цикл для `14B` и `32B` (хирургия `H15` + GRPO `H16` + сборка и аудит 3 вариантов GGUF + 3-way `lm-eval` + выгрузка на Hub без хранения лишних весов на диске) воспроизводится одной командой на эфемерной `1x H100 80GB SXM`:

```bash
git clone https://github.com/mrAndreyIsachenko/tenet-lab.git
cd tenet-lab

# 1. Быстрая синхронизация зависимостей и ускоренная загрузка весов
uv sync --frozen --extra data --extra parity --extra telemetry --extra eval
export HF_HUB_ENABLE_HF_TRANSFER=1

# 2. Запуск мастер-пайплайна (автоматически собирает llama.cpp CUDA, гонит H15-H17 и чистит локальный кэш)
./scripts/remote/run_h100_v2_suite.sh
```

Скрипт [`scripts/remote/run_h100_v2_suite.sh`](../../scripts/remote/run_h100_v2_suite.sh) сам проверяет `nvidia-smi`, собирает `llama-quantize` / `llama-imatrix` / `llama-perplexity` / `llama-server` в `.tools/llama.cpp`, прогоняет `scripts/run_tenet_v2_foundry.py` и `lm-eval`, сохраняет JSON-манифесты в `results/` и выгружает готовые репы в Hugging Face Hub.

---

## Ссылки

- **Коллекция Tenet v2:** [Hugging Face](https://huggingface.co/collections/mr-andrey-isachenko/tenet-v2-thinking-safe-tri-anchor-unslop-and-domain-cka-gguf-6b00aec98d01429c224654d4)
- **Коллекция Tenet v1:** [Hugging Face](https://huggingface.co/collections/mr-andrey-isachenko/tenet-v1-bi-projected-unslop-and-fp8-safe-llm-family-6afed2048ba13a74fb7e0fac)
- **Интерактивный инспектор FP8 и GGUF:** [`mr-andrey-isachenko/fp8-abliteration-inspector`](https://huggingface.co/spaces/mr-andrey-isachenko/fp8-abliteration-inspector)
- **Код и манифесты:** [`github.com/mrAndreyIsachenko/tenet-lab`](https://github.com/mrAndreyIsachenko/tenet-lab)
