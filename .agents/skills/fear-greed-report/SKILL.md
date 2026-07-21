---
name: fear-greed-report
description: Inspect a CNN Fear & Greed market-sentiment report through the local fng-chatbot MCP, verify its seven indicators and Python-selected factors, review the optional DeepSeek explanation in the Telegram preview, and send only after explicit approval. Use for requests to check current market fear/greed, explain the main fear and buffer factors, prepare a Telegram summary, or deliver an already reviewed report.
---

# Fear & Greed Report

Use the MCP tools in this exact safety order. Treat the result as informational market sentiment, never as trading advice.

## 1. Retrieve the report

Call `get_fear_greed_report` first.

- Keep `force_refresh=false` unless the user explicitly needs a new CNN fetch.
- Stop and report the retrieval error if CNN data is unavailable. Do not invent missing values.

## 2. Verify quality and evidence

Before interpreting the report:

1. Confirm that `indicators` contains exactly seven distinct IDs.
2. Check `data_quality.complete`, `missing_fields`, `cached`, and `stale`.
3. Confirm `summary_source=rules`.
4. For every `fear_drivers` item and `buffer`, match its ID, name, score, and rating against `indicators`.
5. If data is stale or incomplete, say so before the interpretation.

Explain only the supplied score, rating, factors, and data-quality facts. Do not add news, forecasts, buy/sell language, security recommendations, or return predictions.

## 3. Preview before delivery

Call `preview_telegram_report` before any send. Show the returned `telegram.text` to the user and retain both `telegram.text` and `telegram.preview_hash` unchanged.

Previewing is read-only but may call NVIDIA-hosted DeepSeek when `NVIDIA_API_KEY` is configured. Do not call DeepSeek separately.

Inspect `interpretation` before presenting the preview:

- For `status=applied`, require `source=deepseek`, one or two explanation lines, and `referenced_indicator_ids` exactly matching the Python-selected fear drivers and buffer in order. Treat `model` as metadata; do not add it to the Telegram text.
- For `status=rules_fallback`, state that the preview uses only the Python rules and briefly report the supplied non-secret `reason`. Do not imply that DeepSeek produced an explanation.

A preview does not authorize sending.

## 4. Send only with authority

Call `send_telegram_report` only when one of these is true:

- The user explicitly approves sending the displayed preview in the current conversation.

Pass:

- `preview_text`: the exact text returned by the preview tool
- `preview_hash`: the matching hash returned by the preview tool
- `idempotency_key`: `daily-report:YYYY-MM-DD` using the report's intended delivery date
- `confirm_send`: `true`

Never edit the text between preview and send. If the text must change, create a new preview and request approval again. Do not retry a timeout automatically because delivery status may be ambiguous.

## 5. Report the outcome

- On success, report `message_id` and `idempotency_key` without exposing tokens or chat IDs.
- On Telegram failure, state that data retrieval and analysis succeeded but delivery failed.
- Never expose `NVIDIA_API_KEY`, Telegram credentials, authorization headers, or raw provider errors.
- Do not claim a message was sent when only a preview was generated.
