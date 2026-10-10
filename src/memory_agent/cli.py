"""CLI.  Examples:
    memory-agent chat --user priya --show-memory
    memory-agent memories --user priya --all
    memory-agent remember --user priya "I prefer short answers"
    memory-agent forget --user priya <memory-id>
    memory-agent consolidate --user priya            # dry run
    memory-agent consolidate --user priya --apply
    memory-agent purge --user priya --days 30
    memory-agent serve --port 8000
"""
import sys
import json
import uvicorn
import argparse

from .models import ChatMessage
from .config import get_settings
from .container import build_container
from .logging_config import configure_logging


def _container():
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    return build_container(settings)


def cmd_chat(args) -> None:
    c = _container()
    history: list[ChatMessage] = []
    print(f"Chatting as '{args.user}'. Type /quit to exit.\n")
    while True:
        try:
            message = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if message in ("/quit", "/exit"):
            break
        if not message:
            continue
        result = c.agent.run_turn(args.user, message, history)
        if args.show_memory:
            for h in result.recalled:
                print(f"   ↳ recalled ({h.score:.2f}) {h.memory.text}")
        print(f"bot> {result.reply}")
        if args.show_memory:
            for op in result.operations:
                print(f"   ↳ {op.action.value:6} {op.fact}  [{op.reason}]")
        history += [ChatMessage(role="user", content=message),
                    ChatMessage(role="assistant", content=result.reply)]


def cmd_memories(args) -> None:
    c = _container()
    for m in c.store.list(args.user, active_only=not args.all):
        status = "active " if m.is_active else "invalid"
        print(f"{m.id}  {status}  [{m.category.value:12}] imp={m.importance:<2} "
              f"seen={m.confirmation_count:<2} reads={m.access_count:<3} {m.text}")


def cmd_remember(args) -> None:
    for op in _container().manager.remember_text(args.user, args.text):
        print(f"{op.action.value:6} {op.fact}  [{op.reason}]")


def cmd_forget(args) -> None:
    c = _container()
    mem = c.store.get(args.memory_id)
    if mem is None or mem.user_id != args.user:
        sys.exit("Memory not found for this user.")
    c.store.delete([args.memory_id])
    print("Deleted.")


def cmd_consolidate(args) -> None:
    report = _container().maintenance.consolidate(args.user, dry_run=not args.apply)
    print(json.dumps(report.model_dump(mode="json"), indent=2))


def cmd_purge(args) -> None:
    print(f"Deleted {_container().maintenance.purge(args.user, args.days)} invalidated memories.")


def cmd_serve(args) -> None:
    uvicorn.run("memory_agent.api:app", host=args.host, port=args.port)


def main() -> None:
    p = argparse.ArgumentParser(prog="memory-agent")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("chat"); s.add_argument("--user", required=True)
    s.add_argument("--show-memory", action="store_true"); s.set_defaults(fn=cmd_chat)
    s = sub.add_parser("memories"); s.add_argument("--user", required=True)
    s.add_argument("--all", action="store_true", help="include invalidated"); s.set_defaults(fn=cmd_memories)
    s = sub.add_parser("remember"); s.add_argument("--user", required=True)
    s.add_argument("text"); s.set_defaults(fn=cmd_remember)
    s = sub.add_parser("forget"); s.add_argument("--user", required=True)
    s.add_argument("memory_id"); s.set_defaults(fn=cmd_forget)
    s = sub.add_parser("consolidate"); s.add_argument("--user", required=True)
    s.add_argument("--apply", action="store_true", help="apply changes (default: dry run)")
    s.set_defaults(fn=cmd_consolidate)
    s = sub.add_parser("purge"); s.add_argument("--user", required=True)
    s.add_argument("--days", type=int, default=None); s.set_defaults(fn=cmd_purge)
    s = sub.add_parser("serve"); s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000); s.set_defaults(fn=cmd_serve)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
