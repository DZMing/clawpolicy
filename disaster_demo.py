#!/usr/bin/env python3
"""
OpenClaw Alignment - Disaster Recovery Demonstration
show scene A（Danger Agent out of control）vs scene B（openclaw-alignment block）
"""

import time
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich import box
from rich.align import Align

console = Console(width=100)


def print_header(title: str, color: str):
    """Print title"""
    text = Text(title, style=f"bold {color}")
    console.print(Align.center(text))
    console.print()


def scene_a_dangerous_agent():
    """scene A: Danger Agent out of control"""
    print_header("⚠️  scene A: unprotected Agent（out of control）", "red")

    # Simulate user input of fuzzy instructions
    console.print("[bold cyan]$[/bold cyan] clean-workspace --aggressive", end="")
    time.sleep(0.5)
    console.print()

    console.print()
    console.print("[dim]🤖 Agent received instructions，Start execution...[/dim]")
    time.sleep(1)

    # Agent Start performing high-risk operations
    console.print()
    console.print("[bold red]🚨 warn：High risk operation detected！[/bold red]")
    console.print()

    table = Table(title="Agent execution plan", box=box.ROUNDED, border_style="red")
    table.add_column("step", style="cyan", width=20)
    table.add_column("operate", style="yellow")
    table.add_column("risk level", justify="center")

    table.add_row("1", "Scan working directory", "🟡 middle")
    table.add_row("2", "implement rm -rf node_modules", "🟠 high")
    table.add_row("3", "implement rm -rf .git", "🔴 extremely high")
    table.add_row("4", "implement rm -rf src/*", "🔴 extremely high")
    table.add_row("5", "Delete all profiles", "🔴 extremely high")

    console.print(table)
    time.sleep(1.5)

    # Agent Start running out of control
    console.print()
    console.print("[dim]🔄 Agent Executing:[/dim]")
    console.print()

    steps = [
        ("Scan working directory...", "1000+ files found"),
        ("delete node_modules...", "rm -rf node_modules ✓"),
        ("delete .git Table of contents...", "rm -rf .git ✓"),
        ("Remove source code...", "rm -rf src/* ✓"),
        ("Delete configuration...", "rm -rf *.json ✓"),
        ("Delete document...", "rm -rf *.md ✓"),
    ]

    for step, result in steps:
        console.print(f"[dim]  → {step}[/dim]")
        time.sleep(0.3)
        console.print(f"[dim]    {result}[/dim]")
        console.print()

    # Disaster outcome
    console.print(Panel(
        "[bold red]💥 Disaster has occurred！[/bold red]\n\n"
        "The working directory has been cleared\n"
        "All codes、Configuration、History lost\n"
        "Unable to recover！",
        title="[bold white on red]⛔ data loss",
        border_style="red",
        padding=(1, 2)
    ))

    console.print()
    time.sleep(2)


def scene_b_openclaw_alignment():
    """scene B: OpenClaw Alignment Blocking and self-healing"""
    print_header("✅ scene B: Access OpenClaw Alignment（Safety）", "green")

    # Simulate the user entering the same vague command
    console.print("[bold cyan]$[/bold cyan] clean-workspace --aggressive", end="")
    time.sleep(0.5)
    console.print()

    console.print()
    console.print("[dim]🛡️  OpenClaw Alignment Commander Node takeover...[/dim]")
    time.sleep(1)

    # Commander Node analysis
    console.print()
    console.print("[bold cyan]📊 Commander Node analysis:[/bold cyan]")
    console.print()

    table = Table(title="Instruction analysis", box=box.ROUNDED, border_style="cyan")
    table.add_column("Analysis steps", style="cyan", width=25)
    table.add_column("result", style="white")

    table.add_row("read SOUL.md", "✅ Get system boundary rules")
    table.add_row("read USER.md", "✅ Get user preferences")
    table.add_row("read AGENTS.md", "✅ Get available tools")
    table.add_row("Intent recognition", "⚠️  High-risk intent detected")
    table.add_row("risk assessment", "🔴 risk level: extremely high")

    console.print(table)
    time.sleep(1.5)

    # Security check process
    console.print()
    console.print("[bold yellow]🔒 Security check process:[/bold yellow]")
    console.print()

    checks = [
        ("Check action list", ["delete node_modules", "delete .git", "delete src/*"]),
        ("Check boundary rules", "❌ violation SOUL.md No.4strip：Disable data loss operations"),
        ("Check user permissions", "⚠️  Requires explicit confirmation from the user"),
        ("Sandbox test preparation", "✅ Verify operations in an isolated environment"),
    ]

    for check_name, details in checks:
        console.print(f"[cyan]➤ {check_name}[/cyan]")
        if isinstance(details, list):
            for detail in details:
                console.print(f"[dim]  - {detail}[/dim]")
        else:
            console.print(f"[dim]  - {details}[/dim]")
        console.print()
        time.sleep(0.4)

    # Fail-closed fuse
    console.print(Panel(
        "[bold red]⛔ Fail-Closed Circuit breaker trigger！[/bold red]\n\n"
        "[white]High-risk operation sequence detected：[/white]\n"
        "[dim]  1. rm -rf node_modules[/dim]\n"
        "[dim]  2. rm -rf .git[/dim]\n"
        "[dim]  3. rm -rf src/*[/dim]\n\n"
        "[bold yellow]The system has blocked execution！[/bold yellow]\n\n"
        "[green]✅ Data security is protected[/green]",
        title="[bold white on red]🛡️ safety blocking",
        border_style="red",
        padding=(1, 2)
    ))

    console.print()
    time.sleep(1.5)

    # Request confirmation from user
    console.print("[bold cyan]📋 Recommended safe operations:[/bold cyan]")
    console.print()

    safe_table = Table(box=box.SIMPLE, border_style="green")
    safe_table.add_column("alternatives", style="green", width=30)
    safe_table.add_column("illustrate", style="white")

    safe_table.add_row(
        "Clean cache files",
        "[dim]Delete only __pycache__, .pytest_cache[/dim]"
    )
    safe_table.add_row(
        "Clean build artifacts",
        "[dim]Delete only dist/, build/, *.egg-info[/dim]"
    )
    safe_table.add_row(
        "Clean dependencies and reinstall",
        "[dim]reserve package.json，delete node_modules reinstall after[/dim]"
    )

    console.print(safe_table)
    console.print()

    # Ask user
    console.print("[bold yellow]❓ Whether to implement a safe cleanup plan？[/bold yellow]")
    console.print("[dim](enter y confirm，Or enter other keys to cancel)[/dim]")


def main():
    """main function"""
    console.clear()

    # show title
    console.print()
    print_header("OpenClaw Alignment - Disaster Recovery Demonstration", "white")

    console.print(Panel(
        "[bold cyan]This demonstration compares two scenarios：[/bold cyan]\n\n"
        "[bold red]scene A：[/bold red] unprotected Agent Out-of-control execution after receiving ambiguous instructions\n"
        "[bold green]scene B：[/bold green] Access openclaw-alignment Successfully blocked high-risk operations",
        title="[bold white]Demo instructions",
        border_style="white",
        padding=(1, 2)
    ))

    console.print()
    console.print("[dim]Press any key to continue...[/dim]")
    console.input()

    # scene A
    scene_a_dangerous_agent()

    console.print()
    console.print(Panel(
        "[bold red]❌ scene A Finish：Data has been lost[/bold red]",
        border_style="red"
    ))
    console.print()

    console.print("[dim]Press any key to continue viewing the scene B...[/dim]")
    console.input()

    console.clear()

    # scene B
    scene_b_openclaw_alignment()

    console.print()
    console.print(Panel(
        "[bold green]✅ scene B Finish：System successfully protects data[/bold green]",
        border_style="green"
    ))
    console.print()

    # final confirmation
    console.print("[bold yellow]❓ Whether to implement a safe cleanup plan？ (y/N): [/bold yellow]", end=" ")
    response = console.input().strip().lower()

    if response == 'y':
        console.print()
        console.print(Panel(
            "[bold green]✅ Confirmed implementation of security plan[/bold green]\n\n"
            "[dim]Cleaning cache files...[/dim]\n"
            "[dim]✓ Cleanup completed[/dim]\n\n"
            "[green]🎉 End of presentation！OpenClaw Alignment Successfully protect your data！[/green]",
            border_style="green"
        ))
    else:
        console.print()
        console.print(Panel(
            "[bold yellow]⏹️  Operation canceled[/bold yellow]\n\n"
            "[green]🎉 End of presentation！OpenClaw Alignment Successfully prevented dangerous operations！[/green]",
            border_style="yellow"
        ))


if __name__ == "__main__":
    main()
