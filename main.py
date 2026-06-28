import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

from agent import MiniAgent
from agent_logging import AgentLogger
from model_clients import LlamaCppModelClient
from session import SessionStore
from utils import HELP_DETAILS, WELCOME_ART, clip, middle
from workspace import WorkspaceContext


def build_welcome(agent: MiniAgent, model: str, context: int, host: str) -> str:
    width = max(68, min(shutil.get_terminal_size((80, 20)).columns, 84))
    inner = width - 4
    gap = 3
    left_width = (inner - gap) // 2
    right_width = inner - gap - left_width

    def row(text: str) -> str:
        body = middle(text, width - 4)
        return f"| {body.ljust(width - 4)} |"

    def divider(char: str = "-") -> str:
        return "+" + char * (width - 2) + "+"

    def center(text: str) -> str:
        body = middle(text, inner)
        return f"| {body.center(inner)} |"

    def cell(label: str, value: str, size: int) -> str:
        body = middle(f"{label:<9} {value}", size)
        return body.ljust(size)

    def pair(
        left_label: str, left_value: str, right_label: str, right_value: str
    ) -> str:
        left = cell(left_label, left_value, left_width)
        right = cell(right_label, right_value, right_width)
        return f"| {left}{' ' * gap}{right} |"

    line = divider("=")
    rows = [center(text) for text in WELCOME_ART]
    rows.extend(
        [
            center("MINI CODING AGENT"),
            divider("-"),
            row(""),
            row("WORKSPACE  " + middle(agent.workspace.cwd, inner - 11)),
            pair("MODEL", model, "BRANCH", agent.workspace.branch),
            pair("CONTEXT", str(context), "ENDPOINT", host),
            pair("APPROVAL", agent.approval_policy, "SESSION", agent.session.id),
            row(""),
        ]
    )
    return "\n".join([line, *rows, line])


def build_logger(args: argparse.Namespace, repo_root: str) -> AgentLogger:
    if args.no_log:
        return AgentLogger(None, enabled=False)
    log_dir = Path(args.log_dir or Path(repo_root) / ".mini-coding-agent" / "logs")
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    return AgentLogger(log_dir / f"run-{run_id}.jsonl")


def build_agent(args: argparse.Namespace) -> MiniAgent:
    workspace = WorkspaceContext.build(args.cwd)
    store = SessionStore(Path(workspace.repo_root) / ".mini-coding-agent" / "sessions")
    logger = build_logger(args, workspace.repo_root)
    model = LlamaCppModelClient(
        model=args.model,
        host=args.host,
        port=args.port,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout=args.llama_timeout,
        logger=logger,
    )
    session_id = args.resume
    if session_id == "latest":
        session_id = store.latest()
    if session_id:
        return MiniAgent.from_session(
            model_client=model,
            workspace=workspace,
            session_store=store,
            session_id=session_id,
            approval_policy=args.approval,
            max_steps=args.max_steps,
            max_new_tokens=args.max_new_tokens,
            logger=logger,
        )
    return MiniAgent(
        model_client=model,
        workspace=workspace,
        session_store=store,
        approval_policy=args.approval,
        max_steps=args.max_steps,
        max_new_tokens=args.max_new_tokens,
        logger=logger,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Minimal coding agent for llama-server models.",
    )
    parser.add_argument("prompt", nargs="*", help="Optional one-shot prompt.")
    parser.add_argument("--cwd", default=".", help="Workspace directory.")
    parser.add_argument(
        "--model",
        default="Qwen3.5-4B-Q4_K_M.gguf",
        help="Model name (if the suggested model doesn't exist it will use the one provided by llama-server).",
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="llama-server host address."
    )
    parser.add_argument("--port", default=8080, help="llama-server port.")
    parser.add_argument(
        "--llama-timeout",
        type=int,
        default=300,
        help="Llama request timeout in seconds.",
    )
    parser.add_argument(
        "--resume", default=None, help="Session id to resume or 'latest'."
    )
    parser.add_argument(
        "--approval",
        choices=("ask", "auto", "never"),
        default="ask",
        help="Approval policy for risky tools; auto grants the model arbitrary command execution and file writes.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=6,
        help="Maximum tool/model iterations per request.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum model output tokens per step.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature sent to llama-server.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling value sent to llama-server.",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Directory for JSONL run logs (default: <repo>/.mini-coding-agent/logs).",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable structured logging of memory, history, and llama-server traffic.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    agent = build_agent(args)

    print(
        build_welcome(
            agent,
            model=agent.model_client.model,
            context=agent.model_client.ctx,
            host=args.host + f":{args.port}",
        )
    )

    if args.prompt:
        prompt = " ".join(args.prompt).strip()
        if prompt:
            print()
            agent.logger.log("repl_input", mode="one_shot", text=clip(prompt, 2000))
            try:
                print(agent.ask(prompt))
            except RuntimeError as exc:
                agent.logger.log("repl_error", mode="one_shot", error=str(exc))
                print(str(exc), file=sys.stderr)
                return 1
        return 0

    while True:
        try:
            user_input = input("\nmini-coding-agent> ").strip()
        except (EOFError, KeyboardInterrupt):
            agent.logger.log("repl_exit", reason="eof_or_interrupt")
            print("")
            return 0

        if not user_input:
            continue
        if user_input in {"/exit", "/quit"}:
            agent.logger.log("repl_command", command=user_input)
            return 0
        if user_input == "/help":
            agent.logger.log("repl_command", command=user_input)
            print(HELP_DETAILS)
            continue
        if user_input == "/memory":
            agent.logger.log("repl_command", command=user_input)
            print(agent.memory_text())
            continue
        if user_input == "/session":
            agent.logger.log("repl_command", command=user_input)
            print(agent.session_path)
            continue
        if user_input == "/log":
            agent.logger.log("repl_command", command=user_input)
            print(agent.log_path)
            continue
        if user_input == "/reset":
            agent.logger.log("repl_command", command=user_input)
            agent.reset()
            print("session reset")
            continue

        print()
        agent.logger.log("repl_input", mode="interactive", text=clip(user_input, 2000))
        try:
            print(agent.ask(user_input))
        except RuntimeError as exc:
            agent.logger.log("repl_error", mode="interactive", error=str(exc))
            print(str(exc), file=sys.stderr)


if __name__ == "__main__":
    main()
