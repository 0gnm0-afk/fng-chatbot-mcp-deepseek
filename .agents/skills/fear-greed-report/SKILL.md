---
name: fear-greed-report
description: Inspect a CNN Fear & Greed market-sentiment report through the local fng-chatbot MCP, verify its seven indicators and Python-selected factors, and send the generated Telegram report by default without a separate approval turn. Use preview-only when the user explicitly requests no delivery. Use for requests to check current market fear/greed, explain the main fear and buffer factors, preview a Telegram summary, or deliver the current report.
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

## 3. Deliver by default or preview only

Unless the user explicitly says not to send, call `send_telegram_report` after the quality checks. Pass:

- `idempotency_key`: `daily-report:YYYY-MM-DD` using the report's intended delivery date
- `force_refresh`: `false` unless the user explicitly requested a new CNN fetch

The send tool builds one current Telegram preview internally, optionally calls NVIDIA-hosted DeepSeek, binds the exact text to SHA-256, and immediately sends that same text. Do not call `preview_telegram_report` first in the default delivery path because prices or model wording can change between two renders. No separate user approval turn is required.

If the user explicitly requests preview-only, no delivery, or analysis without sending, call `preview_telegram_report` instead and do not call the send tool. Previewing is read-only but may call DeepSeek when `NVIDIA_API_KEY` is configured. Do not call DeepSeek separately.

Inspect the returned `interpretation` in either path:

- For `status=applied`, require `source=deepseek`, one or two explanation lines, and `referenced_indicator_ids` exactly matching the Python-selected fear drivers and buffer in order. Treat `model` as metadata; do not add it to the Telegram text.
- For `status=rules_fallback`, state that only the Python rules were used and briefly report the supplied non-secret `reason`. The default delivery path still sends the rules-based report.

## 4. Report the outcome

- On delivery success, require `telegram.sent=true` and report `message_id`, `idempotency_key`, and `preview_hash` without exposing tokens or chat IDs.
- If `MCP_ALLOW_TELEGRAM_SEND=false`, state that the server kill switch blocked the default delivery.
- On Telegram failure, state that data retrieval and analysis succeeded but delivery failed.
- Do not retry a send timeout automatically because delivery status may be ambiguous.
- Never expose `NVIDIA_API_KEY`, Telegram credentials, authorization headers, or raw provider errors.
- Do not claim a message was sent when only a preview was generated.
