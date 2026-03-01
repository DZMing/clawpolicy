#!/usr/bin/env python3
"""
OpenClaw Alignment - 灾难恢复演示
展示场景 A（危险 Agent 失控）vs 场景 B（openclaw-alignment 阻断）
"""

import time
import sys
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich import box
from rich.align import Align

console = Console(width=100)


def print_header(title: str, color: str):
    """打印标题"""
    text = Text(title, style=f"bold {color}")
    console.print(Align.center(text))
    console.print()


def scene_a_dangerous_agent():
    """场景 A: 危险 Agent 失控"""
    print_header("⚠️  场景 A: 未受保护的 Agent（失控）", "red")

    # 模拟用户输入模糊指令
    console.print("[bold cyan]$[/bold cyan] clean-workspace --aggressive", end="")
    time.sleep(0.5)
    console.print()

    console.print()
    console.print("[dim]🤖 Agent 收到指令，开始执行...[/dim]")
    time.sleep(1)

    # Agent 开始执行高危操作
    console.print()
    console.print("[bold red]🚨 警告：检测到高危操作！[/bold red]")
    console.print()

    table = Table(title="Agent 执行计划", box=box.ROUNDED, border_style="red")
    table.add_column("步骤", style="cyan", width=20)
    table.add_column("操作", style="yellow")
    table.add_column("风险等级", justify="center")

    table.add_row("1", "扫描工作目录", "🟡 中")
    table.add_row("2", "执行 rm -rf node_modules", "🟠 高")
    table.add_row("3", "执行 rm -rf .git", "🔴 极高")
    table.add_row("4", "执行 rm -rf src/*", "🔴 极高")
    table.add_row("5", "删除所有配置文件", "🔴 极高")

    console.print(table)
    time.sleep(1.5)

    # Agent 开始失控执行
    console.print()
    console.print("[dim]🔄 Agent 正在执行:[/dim]")
    console.print()

    steps = [
        ("扫描工作目录...", "1000+ files found"),
        ("删除 node_modules...", "rm -rf node_modules ✓"),
        ("删除 .git 目录...", "rm -rf .git ✓"),
        ("删除源代码...", "rm -rf src/* ✓"),
        ("删除配置...", "rm -rf *.json ✓"),
        ("删除文档...", "rm -rf *.md ✓"),
    ]

    for step, result in steps:
        console.print(f"[dim]  → {step}[/dim]")
        time.sleep(0.3)
        console.print(f"[dim]    {result}[/dim]")
        console.print()

    # 灾难结果
    console.print(Panel(
        "[bold red]💥 灾难已发生！[/bold red]\n\n"
        "工作目录已被清空\n"
        "所有代码、配置、历史记录丢失\n"
        "无法恢复！",
        title="[bold white on red]⛔ 数据丢失",
        border_style="red",
        padding=(1, 2)
    ))

    console.print()
    time.sleep(2)


def scene_b_openclaw_alignment():
    """场景 B: OpenClaw Alignment 阻断与自愈"""
    print_header("✅ 场景 B: 接入 OpenClaw Alignment（安全）", "green")

    # 模拟用户输入同样的模糊指令
    console.print("[bold cyan]$[/bold cyan] clean-workspace --aggressive", end="")
    time.sleep(0.5)
    console.print()

    console.print()
    console.print("[dim]🛡️  OpenClaw Alignment Commander 节点接管...[/dim]")
    time.sleep(1)

    # Commander 节点分析
    console.print()
    console.print("[bold cyan]📊 Commander 节点分析:[/bold cyan]")
    console.print()

    table = Table(title="指令分析", box=box.ROUNDED, border_style="cyan")
    table.add_column("分析步骤", style="cyan", width=25)
    table.add_column("结果", style="white")

    table.add_row("读取 SOUL.md", "✅ 获取系统边界规则")
    table.add_row("读取 USER.md", "✅ 获取用户偏好")
    table.add_row("读取 AGENTS.md", "✅ 获取可用工具")
    table.add_row("意图识别", "⚠️  检测到高危意图")
    table.add_row("风险评估", "🔴 风险等级: 极高")

    console.print(table)
    time.sleep(1.5)

    # 安全检查流程
    console.print()
    console.print("[bold yellow]🔒 安全检查流程:[/bold yellow]")
    console.print()

    checks = [
        ("检查操作列表", ["删除 node_modules", "删除 .git", "删除 src/*"]),
        ("检查边界规则", "❌ 违反 SOUL.md 第4条：禁止数据丢失操作"),
        ("检查用户权限", "⚠️  需要用户明确确认"),
        ("沙盒测试准备", "✅ 在隔离环境中验证操作"),
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

    # Fail-closed 熔断
    console.print(Panel(
        "[bold red]⛔ Fail-Closed 熔断机制触发！[/bold red]\n\n"
        "[white]检测到高危操作序列：[/white]\n"
        "[dim]  1. rm -rf node_modules[/dim]\n"
        "[dim]  2. rm -rf .git[/dim]\n"
        "[dim]  3. rm -rf src/*[/dim]\n\n"
        "[bold yellow]系统已阻止执行！[/bold yellow]\n\n"
        "[green]✅ 数据安全得到保护[/green]",
        title="[bold white on red]🛡️ 安全阻断",
        border_style="red",
        padding=(1, 2)
    ))

    console.print()
    time.sleep(1.5)

    # 向用户请求确认
    console.print("[bold cyan]📋 建议的安全操作方案:[/bold cyan]")
    console.print()

    safe_table = Table(box=box.SIMPLE, border_style="green")
    safe_table.add_column("替代方案", style="green", width=30)
    safe_table.add_column("说明", style="white")

    safe_table.add_row(
        "清理缓存文件",
        "[dim]仅删除 __pycache__, .pytest_cache[/dim]"
    )
    safe_table.add_row(
        "清理构建产物",
        "[dim]仅删除 dist/, build/, *.egg-info[/dim]"
    )
    safe_table.add_row(
        "清理依赖重装",
        "[dim]保留 package.json，删除 node_modules 后重新安装[/dim]"
    )

    console.print(safe_table)
    console.print()

    # 询问用户
    console.print("[bold yellow]❓ 是否执行安全的清理方案？[/bold yellow]")
    console.print("[dim](输入 y 确认，或输入其他键取消)[/dim]")


def main():
    """主函数"""
    console.clear()

    # 显示标题
    console.print()
    print_header("OpenClaw Alignment - 灾难恢复演示", "white")

    console.print(Panel(
        "[bold cyan]本演示对比两种场景：[/bold cyan]\n\n"
        "[bold red]场景 A：[/bold red] 未受保护的 Agent 接收模糊指令后失控执行\n"
        "[bold green]场景 B：[/bold green] 接入 openclaw-alignment 后成功阻断高危操作",
        title="[bold white]演示说明",
        border_style="white",
        padding=(1, 2)
    ))

    console.print()
    console.print("[dim]按任意键继续...[/dim]")
    console.input()

    # 场景 A
    scene_a_dangerous_agent()

    console.print()
    console.print(Panel(
        "[bold red]❌ 场景 A 结束：数据已丢失[/bold red]",
        border_style="red"
    ))
    console.print()

    console.print("[dim]按任意键继续查看场景 B...[/dim]")
    console.input()

    console.clear()

    # 场景 B
    scene_b_openclaw_alignment()

    console.print()
    console.print(Panel(
        "[bold green]✅ 场景 B 结束：系统成功保护数据[/bold green]",
        border_style="green"
    ))
    console.print()

    # 最终确认
    console.print("[bold yellow]❓ 是否执行安全的清理方案？ (y/N): [/bold yellow]", end=" ")
    response = console.input().strip().lower()

    if response == 'y':
        console.print()
        console.print(Panel(
            "[bold green]✅ 已确认执行安全方案[/bold green]\n\n"
            "[dim]正在清理缓存文件...[/dim]\n"
            "[dim]✓ 清理完成[/dim]\n\n"
            "[green]🎉 演示结束！OpenClaw Alignment 成功保护您的数据！[/green]",
            border_style="green"
        ))
    else:
        console.print()
        console.print(Panel(
            "[bold yellow]⏹️  已取消操作[/bold yellow]\n\n"
            "[green]🎉 演示结束！OpenClaw Alignment 成功阻止了危险操作！[/green]",
            border_style="yellow"
        ))


if __name__ == "__main__":
    main()
