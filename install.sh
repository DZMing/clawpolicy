#!/bin/bash

# OpenClaw 意图对齐 Skill - 一键安装脚本
# 适用于 macOS 和 Linux

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# 配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENCLAW_DIR="$HOME/.openclaw"
EXTENSION_DIR="$OPENCLAW_DIR/extensions/intent-alignment"

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${MAGENTA}  $1${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

# 显示欢迎信息
show_welcome() {
    cat <<EOF

╔════════════════════════════════════════════════════════════╗
║                                                              ║
║     OpenClaw 意图对齐 Skill - 一键安装                      ║
║                                                              ║
║     简简单单，让OpenClaw更懂你                                ║
║     自动学习你的偏好并自适应优化                              ║
║                                                              ║
╚════════════════════════════════════════════════════════════╝

本安装脚本将会：
  1. 创建扩展目录
  2. 安装意图对齐Skill
  3. 安装默认配置
  4. 验证安装

预计时间：30 秒

按任意键继续...
EOF

    read -n 1 -s
    echo ""
}

# 检查系统要求
check_requirements() {
    log_step "检查系统要求"

    # 检查操作系统
    local os=$(uname -s)
    log_info "操作系统: $os"

    if [ "$os" != "Darwin" ] && [ "$os" != "Linux" ]; then
        log_error "不支持的操作系统: $os"
        exit 1
    fi

    # 检查OpenClaw目录
    if [ ! -d "$OPENCLAW_DIR" ]; then
        log_warning "未检测到OpenClaw目录"
        log_info "创建OpenClaw目录: $OPENCLAW_DIR"
        mkdir -p "$OPENCLAW_DIR"
    fi

    log_success "✅ 系统要求检查通过"
}

# 创建目录结构
create_directories() {
    log_step "创建目录结构"

    local dirs=(
        "$EXTENSION_DIR"
        "$EXTENSION_DIR/skills"
        "$EXTENSION_DIR/config"
    )

    for dir in "${dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            log_info "创建目录: $dir"
        fi
    done

    log_success "✅ 目录结构创建完成"
}

# 安装Skill
install_skill() {
    log_step "安装意图对齐Skill"

    # 复制Skill文件
    if [ -f "$SCRIPT_DIR/skills/意图对齐.md" ]; then
        cp "$SCRIPT_DIR/skills/意图对齐.md" "$EXTENSION_DIR/skills/"
        log_success "✅ Skill文件安装完成"
    else
        log_error "未找到Skill文件: $SCRIPT_DIR/skills/意图对齐.md"
        exit 1
    fi
}

# 安装配置文件
install_config() {
    log_step "安装配置文件"

    # 复制配置文件（如果存在）
    if [ -f "$SCRIPT_DIR/config/config.json" ]; then
        cp "$SCRIPT_DIR/config/config.json" "$EXTENSION_DIR/config/"
        log_success "✅ 配置文件安装完成"
    else
        log_warning "未找到配置文件，创建默认配置..."
        mkdir -p "$EXTENSION_DIR/config"
        cat > "$EXTENSION_DIR/config/config.json" <<EOF
{
  "version": "1.0.0",
  "last_updated": "2026-02-28T18:00:00Z",
  "preferences": {
    "automation_level": "balanced",
    "communication_style": "direct"
  }
}
EOF
        log_success "✅ 默认配置已创建"
    fi
}

# 创建README
create_readme() {
    log_step "创建使用说明"

    cat > "$EXTENSION_DIR/README.md" <<EOF
# 意图对齐 Skill

OpenClaw意图对齐扩展，帮助你更好地使用OpenClaw。

## 使用方法

在OpenClaw中调用：
- "优化我的工作流"
- "意图对齐"
- "分析我的偏好"

## 配置文件

配置位置：~/.openclaw/extensions/intent-alignment/config/config.json

## 卸载

删除目录：rm -rf ~/.openclaw/extensions/intent-alignment
EOF

    log_success "✅ 使用说明已创建"
}

# 验证安装
verify_installation() {
    log_step "验证安装"

    local all_good=true

    # 检查文件是否存在
    if [ -f "$EXTENSION_DIR/skills/意图对齐.md" ]; then
        log_success "✅ Skill文件已安装"
    else
        log_error "❌ Skill文件缺失"
        all_good=false
    fi

    if [ -f "$EXTENSION_DIR/config/config.json" ]; then
        log_success "✅ 配置文件已安装"
    else
        log_error "❌ 配置文件缺失"
        all_good=false
    fi

    if [ "$all_good" = true ]; then
        return 0
    else
        return 1
    fi
}

# 显示完成信息
show_completion() {
    cat <<EOF

╔════════════════════════════════════════════════════════════╗
║          安装完成！                                         ║
╚════════════════════════════════════════════════════════════╝

✅ 意图对齐Skill已安装到: $EXTENSION_DIR
✅ 配置文件已就绪

📚 使用方法:
  在OpenClaw中调用：
  - "优化我的工作流"
  - "意图对齐"
  - "分析我的偏好"

📂 安装位置:
  - Skill: $EXTENSION_DIR/skills/意图对齐.md
  - 配置: $EXTENSION_DIR/config/config.json

🔄 配置调整:
  编辑配置文件：vim $EXTENSION_DIR/config/config.json

📖 完整文档:
  cat $SCRIPT_DIR/README.md

感谢使用意图对齐Skill！

EOF
}

# 主函数
main() {
    show_welcome
    check_requirements
    create_directories
    install_skill
    install_config
    create_readme

    if verify_installation; then
        show_completion
        log_success "🎉 安装完成！"
    else
        log_error "❌ 安装验证失败，请检查错误信息"
        exit 1
    fi
}

# 执行
main "$@"
