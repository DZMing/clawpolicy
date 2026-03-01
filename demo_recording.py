#!/usr/bin/env python3
"""
生成 OpenClaw Alignment 演示 GIF
模拟终端运行 openclaw-align init 的过程
"""

from PIL import Image, ImageDraw, ImageFont
import os

# 配置
WIDTH, HEIGHT = 900, 500
FONT_SIZE = 16
BG_COLOR = (30, 30, 30)  # 深灰色终端背景
TEXT_COLOR = (200, 200, 200)  # 浅灰色文本
SUCCESS_COLOR = (100, 255, 100)  # 绿色
CMD_COLOR = (100, 200, 255)  # 蓝色命令
PROMPT_COLOR = (255, 200, 100)  # 黄色提示符

# 命令序列
COMMANDS = [
    ("$ pip install openclaw-alignment", CMD_COLOR),
    ("", TEXT_COLOR),
    ("$ openclaw-align init", CMD_COLOR),
    ("", TEXT_COLOR),
    ("🚀 初始化 OpenClaw Alignment 记忆库...", TEXT_COLOR),
    ("✅ 创建: .openclaw_memory/USER.md", SUCCESS_COLOR),
    ("✅ 创建: .openclaw_memory/SOUL.md", SUCCESS_COLOR),
    ("✅ 创建: .openclaw_memory/AGENTS.md", SUCCESS_COLOR),
    ("✅ 创建: .openclaw_memory/config.json", SUCCESS_COLOR),
    ("✅ 创建: .openclaw_memory/.gitignore", SUCCESS_COLOR),
    ("", TEXT_COLOR),
    ("=" * 60, TEXT_COLOR),
    ("✨ 初始化完成！", SUCCESS_COLOR),
    ("=" * 60, TEXT_COLOR),
    ("📂 记忆库位置: .openclaw_memory", TEXT_COLOR),
    ("📄 已创建文件:", TEXT_COLOR),
    ("   - USER.md", TEXT_COLOR),
    ("   - SOUL.md", TEXT_COLOR),
    ("   - AGENTS.md", TEXT_COLOR),
    ("   - config.json", TEXT_COLOR),
    ("   - .gitignore", TEXT_COLOR),
    ("", TEXT_COLOR),
    ("📝 下一步:", TEXT_COLOR),
    ("   1. 编辑 USER.md，配置你的个人偏好", TEXT_COLOR),
    ("   2. 检查 SOUL.md，了解系统原则", TEXT_COLOR),
    ("   3. 查看 AGENTS.md，了解可用的工具", TEXT_COLOR),
    ("", TEXT_COLOR),
]

def create_terminal_frame(lines, cursor_position=None):
    """创建一帧终端画面"""
    img = Image.new('RGB', (WIDTH, HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)

    # 尝试使用等宽字体
    try:
        # macOS/Linux 上的等宽字体
        font = ImageFont.truetype('/System/Library/Fonts/Menlo.ttc', FONT_SIZE)
    except:
        try:
            font = ImageFont.truetype('DejaVuSansMono.ttf', FONT_SIZE)
        except:
            font = ImageFont.load_default()

    y_offset = 30
    x_offset = 30

    # 绘制标题栏
    draw.rectangle([(0, 0), (WIDTH, 25)], fill=(50, 50, 50))
    draw.text((10, 5), "OpenClaw Alignment Demo", fill=(255, 255, 255), font=font)

    # 绘制命令历史
    for line, color in lines:
        # 添加提示符
        if line.startswith("$"):
            display_line = line
        else:
            display_line = line

        draw.text((x_offset, y_offset), display_line, fill=color, font=font)
        y_offset += FONT_SIZE + 8

    # 绘制光标
    if cursor_position is not None:
        cursor_x, cursor_y = cursor_position
        draw.rectangle([(cursor_x, cursor_y), (cursor_x + 10, cursor_y + FONT_SIZE)],
                      fill=(200, 200, 200))

    return img

def generate_gif(output_path='demo.gif', duration=80):
    """生成 GIF 动画"""
    frames = []
    current_lines = []

    print("🎬 生成演示 GIF...")

    # 逐行添加命令
    for i, (line, color) in enumerate(COMMANDS):
        current_lines.append((line, color))

        # 为每一行生成多帧，模拟打字效果
        if line and not line.startswith("="):
            # 打字效果（减少帧数）
            for j in range(1, len(line) + 1, 2):  # 每 2 个字符一帧
                partial_line = line[:j]
                temp_lines = current_lines[:-1] + [(partial_line, color)]
                frame = create_terminal_frame(temp_lines)
                frames.append(frame)
        else:
            # 非打字行（空行、分隔线）
            frame = create_terminal_frame(current_lines)
            frames.append(frame)

        # 在每个命令后停顿（减少停顿帧）
        for _ in range(1):
            frame = create_terminal_frame(current_lines)
            frames.append(frame)

    # 最后一帧停留更久
    for _ in range(5):
        frame = create_terminal_frame(current_lines)
        frames.append(frame)

    # 保存 GIF
    print(f"💾 保存 GIF 到 {output_path}...")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        optimize=True,
        duration=duration,
        loop=0
    )

    print(f"✅ GIF 生成成功！")
    print(f"📊 文件大小: {os.path.getsize(output_path) / 1024:.1f} KB")
    print(f"📝 总帧数: {len(frames)}")

if __name__ == "__main__":
    generate_gif()
