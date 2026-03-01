#!/usr/bin/env python3
"""
OpenClaw Alignment CLI 命令行接口

提供一键初始化和配置管理功能
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Optional


class OpenClawAlignmentCLI:
    """OpenClaw Alignment CLI 主类"""

    def __init__(self):
        self.memory_dir_name = ".openclaw_memory"
        self.config_file_name = "config.json"
        self.templates = {
            "USER.md": "USER_template.md",
            "SOUL.md": "SOUL_template.md",
            "AGENTS.md": "AGENTS_template.md",
        }

    def get_template_dir(self) -> Path:
        """获取模板文件目录"""
        # 从包内获取模板目录
        package_dir = Path(__file__).parent
        template_dir = package_dir.parent / "templates"
        return template_dir

    def get_memory_dir(self, cwd: Optional[Path] = None) -> Path:
        """获取记忆库目录"""
        if cwd is None:
            cwd = Path.cwd()
        return cwd / self.memory_dir_name

    def init(self, target_dir: Optional[str] = None, force: bool = False) -> bool:
        """
        初始化 OpenClaw Alignment 记忆库

        Args:
            target_dir: 目标目录，默认为当前工作目录
            force: 强制覆盖已存在的文件

        Returns:
            是否成功初始化
        """
        if target_dir:
            cwd = Path(target_dir).resolve()
        else:
            cwd = Path.cwd()

        memory_dir = self.get_memory_dir(cwd)
        template_dir = self.get_template_dir()

        # 检查模板目录是否存在
        if not template_dir.exists():
            print(f"❌ 错误：模板目录不存在: {template_dir}")
            print(f"   请确保 openclaw-alignment 已正确安装")
            return False

        # 检查是否已初始化
        if memory_dir.exists():
            if not force:
                print(f"⚠️  记忆库已存在: {memory_dir}")
                print(f"   如需重新初始化，请使用 --force 参数")
                return False
            print(f"🔄 强制重新初始化...")
        else:
            print(f"🚀 初始化 OpenClaw Alignment 记忆库...")

        # 创建记忆库目录
        memory_dir.mkdir(parents=True, exist_ok=True)

        # 复制模板文件
        success_count = 0
        for target_name, template_name in self.templates.items():
            template_file = template_dir / template_name
            target_file = memory_dir / target_name

            if not template_file.exists():
                print(f"⚠️  模板文件不存在: {template_name}")
                continue

            if target_file.exists() and not force:
                print(f"⏭️  跳过已存在: {target_name}")
                continue

            shutil.copy2(template_file, target_file)
            success_count += 1
            print(f"✅ 创建: {target_file}")

        # 创建配置文件
        config_file = memory_dir / self.config_file_name
        if not config_file.exists() or force:
            config = {
                "version": "1.0.0",
                "initialized_at": str(Path.cwd()),
                "memory_path": str(memory_dir),
                "features": {
                    "rl_enabled": True,
                    "auto_learning": True,
                    "safety_checks": True,
                },
            }
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print(f"✅ 创建: {config_file}")

        # 创建 .gitignore
        gitignore_file = memory_dir / ".gitignore"
        if not gitignore_file.exists() or force:
            with open(gitignore_file, "w", encoding="utf-8") as f:
                f.write("# OpenClaw Alignment 本地配置\n")
                f.write("# 不要提交到版本控制\n")
                f.write("config.json\n")
                f.write("*.backup\n")
                f.write("*.cache\n")
            print(f"✅ 创建: {gitignore_file}")

        # 显示成功信息
        print("")
        print("=" * 60)
        print("✨ 初始化完成！")
        print("=" * 60)
        print(f"📂 记忆库位置: {memory_dir}")
        print(f"📄 已创建文件:")
        for target_name in self.templates.keys():
            print(f"   - {target_name}")
        print(f"   - {self.config_file_name}")
        print(f"   - .gitignore")
        print("")
        print("📝 下一步:")
        print("   1. 编辑 USER.md，配置你的个人偏好")
        print("   2. 检查 SOUL.md，了解系统原则")
        print("   3. 查看 AGENTS.md，了解可用的工具")
        print("   4. 运行: openclaw-align analyze")
        print("")

        return True

    def status(self) -> None:
        """显示当前状态"""
        cwd = Path.cwd()
        memory_dir = self.get_memory_dir(cwd)
        config_file = memory_dir / self.config_file_name

        print(f"📊 OpenClaw Alignment 状态")
        print(f"")
        print(f"📂 记忆库位置: {memory_dir}")
        print(f"   状态: {'✅ 存在' if memory_dir.exists() else '❌ 不存在'}")
        print(f"")

        if memory_dir.exists():
            print(f"📄 配置文件:")

            # 检查配置文件
            if config_file.exists():
                with open(config_file, "r", encoding="utf-8") as f:
                    config = json.load(f)
                print(f"   ✅ {self.config_file_name}")
                print(f"      版本: {config.get('version', 'unknown')}")
                print(f"      RL启用: {config.get('features', {}).get('rl_enabled', False)}")
            else:
                print(f"   ❌ {self.config_file_name} (不存在)")

            # 检查模板文件
            for target_name in self.templates.keys():
                target_file = memory_dir / target_name
                status = "✅" if target_file.exists() else "❌"
                print(f"   {status} {target_name}")
        else:
            print(f"💡 提示: 运行 'openclaw-align init' 初始化记忆库")
        print("")

    def version(self) -> None:
        """显示版本信息"""
        from . import __version__
        print(f"OpenClaw Alignment CLI v{__version__}")
        print(f"")
        print(f"Python: {__import__('sys').version}")
        print(f"安装路径: {Path(__file__).parent.parent}")


def main():
    """CLI 主入口"""
    parser = argparse.ArgumentParser(
        description="OpenClaw Alignment - 强化学习驱动的对齐系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  openclaw-align init              初始化记忆库
  openclaw-align init --force       强制重新初始化
  openclaw-align init ~/projects    在指定目录初始化
  openclaw-align status             查看状态
  openclaw-align version            显示版本
        """
    )

    parser.add_argument(
        "--version",
        action="store_true",
        help="显示版本信息"
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # init 命令
    init_parser = subparsers.add_parser(
        "init",
        help="初始化记忆库"
    )
    init_parser.add_argument(
        "target_dir",
        nargs="?",
        help="目标目录（默认为当前目录）"
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="强制覆盖已存在的文件"
    )

    # status 命令
    subparsers.add_parser(
        "status",
        help="显示当前状态"
    )

    # 分析命令（保留原有功能）
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="分析 Git 历史学习偏好"
    )
    analyze_parser.add_argument(
        "--repo",
        default=".",
        help="Git 仓库路径"
    )
    analyze_parser.add_argument(
        "--commits",
        type=int,
        default=100,
        help="分析的提交数量"
    )

    args = parser.parse_args()
    cli = OpenClawAlignmentCLI()

    # 显示版本
    if args.version:
        cli.version()
        return

    # 执行命令
    if args.command == "init":
        success = cli.init(
            target_dir=args.target_dir,
            force=args.force
        )
        exit(0 if success else 1)

    elif args.command == "status":
        cli.status()

    elif args.command == "analyze":
        # 调用原有的分析功能
        from .integration import IntentAlignmentEngine
        engine = IntentAlignmentEngine(args.repo)
        engine.run_analysis(args.commits)

    else:
        # 默认显示状态
        parser.print_help()


if __name__ == "__main__":
    main()
