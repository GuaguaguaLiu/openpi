#!/bin/bash

# 单帧预测测试示例脚本
# 使用方法：bash scripts/example_test_prediction.sh

echo "🚀 单帧预测测试示例"
echo "===================="

# 设置变量（请根据你的实际情况修改）
CONFIG_NAME="agi_debug"
CHECKPOINT_PATH="./checkpoints/agi_debug/my_experiment"
OUTPUT_DIR="./test_results"

echo "📋 测试配置："
echo "  配置名称: $CONFIG_NAME"
echo "  检查点路径: $CHECKPOINT_PATH"
echo "  输出目录: $OUTPUT_DIR"
echo ""

# 检查检查点路径是否存在
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "❌ 错误：检查点路径不存在: $CHECKPOINT_PATH"
    echo "请修改脚本中的 CHECKPOINT_PATH 变量"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "🔍 开始测试..."
echo ""

# 运行测试脚本
python scripts/test_single_prediction.py \
    "$CONFIG_NAME" \
    "$CHECKPOINT_PATH" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "✅ 测试完成！"
echo "📁 结果保存在: $OUTPUT_DIR/prediction_results.json"
echo ""

# 显示结果文件内容（如果存在）
if [ -f "$OUTPUT_DIR/prediction_results.json" ]; then
    echo "📊 结果摘要："
    echo "============="
    python -c "
import json
with open('$OUTPUT_DIR/prediction_results.json', 'r') as f:
    data = json.load(f)
print(f'Loss值: {data[\"loss\"]:.6f}')
print(f'MSE: {data[\"comparison_stats\"][\"mse\"]:.6f}')
print(f'MAE: {data[\"comparison_stats\"][\"mae\"]:.6f}')
print(f'最大误差: {data[\"comparison_stats\"][\"max_error\"]:.6f}')
"
fi

echo ""
echo "🎉 测试脚本执行完毕！"
