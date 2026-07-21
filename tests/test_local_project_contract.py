import json
import sys
import tomllib
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TOOLS = {
    "get_fear_greed_report",
    "preview_telegram_report",
    "send_telegram_report",
}
EXPECTED_ENVIRONMENT_VARIABLES = {
    "NVIDIA_API_KEY",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
    "MCP_ALLOW_TELEGRAM_SEND",
    "CNN_FNG_CACHE_TTL_SECONDS",
}


def test_remote_runtime_artifacts_are_absent() -> None:
    removed_paths = (
        PROJECT_ROOT / "Dockerfile",
        PROJECT_ROOT / ".dockerignore",
        PROJECT_ROOT / "deploy",
        PROJECT_ROOT / "src" / "fng_chatbot" / "app.py",
        PROJECT_ROOT / "tests" / "test_http_app.py",
        PROJECT_ROOT / "tests" / "test_container_contract.py",
        PROJECT_ROOT / "tests" / "test_cloud_run_source.py",
    )

    assert all(not path.exists() for path in removed_paths)


def test_package_exposes_only_the_local_mcp_entrypoint() -> None:
    config = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert config["project"]["name"] == "fng-chatbot"
    assert config["project"]["scripts"] == {"fng-chatbot-mcp": "fng_chatbot.mcp_server:main"}
    assert config["project"]["dependencies"] == ["httpx>=0.28,<1", "mcp>=1.27,<2"]


def test_codex_and_claude_start_the_same_stdio_module() -> None:
    codex = tomllib.loads((PROJECT_ROOT / ".codex" / "config.toml").read_text(encoding="utf-8"))[
        "mcp_servers"
    ]["fng_chatbot"]
    claude = json.loads((PROJECT_ROOT / ".mcp.json").read_text(encoding="utf-8"))["mcpServers"][
        "fng-chatbot"
    ]

    assert codex["args"] == claude["args"] == ["-m", "fng_chatbot.mcp_server"]
    assert codex["command"] == "../.venv/Scripts/python.exe"
    assert codex["cwd"] == ".."
    assert claude["command"] == "${CLAUDE_PROJECT_DIR:-.}/.venv/Scripts/python.exe"
    assert claude["type"] == "stdio"
    assert codex["default_tools_approval_mode"] == "writes"
    assert codex["tools"]["send_telegram_report"]["approval_mode"] == "prompt"
    assert set(codex["env_vars"]) == EXPECTED_ENVIRONMENT_VARIABLES
    assert claude["env"] == {
        "NVIDIA_API_KEY": "${NVIDIA_API_KEY:-}",
        "TELEGRAM_BOT_TOKEN": "${TELEGRAM_BOT_TOKEN:-}",
        "TELEGRAM_CHAT_ID": "${TELEGRAM_CHAT_ID:-}",
        "MCP_ALLOW_TELEGRAM_SEND": "${MCP_ALLOW_TELEGRAM_SEND:-false}",
        "CNN_FNG_CACHE_TTL_SECONDS": "${CNN_FNG_CACHE_TTL_SECONDS:-300}",
    }


def test_codex_and_claude_skills_share_one_contract() -> None:
    codex_skill = (
        PROJECT_ROOT / ".agents" / "skills" / "fear-greed-report" / "SKILL.md"
    ).read_text(encoding="utf-8")
    claude_skill = (
        PROJECT_ROOT / ".claude" / "skills" / "fear-greed-report" / "SKILL.md"
    ).read_text(encoding="utf-8")

    assert codex_skill == claude_skill
    assert EXPECTED_TOOLS <= {name for name in EXPECTED_TOOLS if name in codex_skill}
    assert "explicitly approves" in codex_skill
    assert "status=applied" in codex_skill
    assert "status=rules_fallback" in codex_skill
    assert "Do not call DeepSeek separately" in codex_skill
    assert "Do not claim a message was sent" in codex_skill


@pytest.mark.anyio
async def test_real_stdio_process_initializes_and_lists_three_tools() -> None:
    config_dir = PROJECT_ROOT / ".codex"
    codex = tomllib.loads((config_dir / "config.toml").read_text(encoding="utf-8"))["mcp_servers"][
        "fng_chatbot"
    ]
    server = StdioServerParameters(
        command=sys.executable,
        args=codex["args"],
        cwd=(config_dir / codex["cwd"]).resolve(),
        env={
            "MCP_ALLOW_TELEGRAM_SEND": "false",
        },
    )

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()

    assert {tool.name for tool in result.tools} == EXPECTED_TOOLS


@pytest.mark.anyio
async def test_claude_project_config_initializes_and_lists_three_tools() -> None:
    claude = json.loads((PROJECT_ROOT / ".mcp.json").read_text(encoding="utf-8"))["mcpServers"][
        "fng-chatbot"
    ]
    server = StdioServerParameters(
        command=sys.executable,
        args=claude["args"],
        cwd=PROJECT_ROOT,
        env={"MCP_ALLOW_TELEGRAM_SEND": "false"},
    )

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()

    assert {tool.name for tool in result.tools} == EXPECTED_TOOLS
