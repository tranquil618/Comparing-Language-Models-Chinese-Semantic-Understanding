
from huggingface_hub import snapshot_download
#DeepSeek-R1-Distill-Qwen-1.5B/7B 或 DeepSeek-V2-Lite
#model_id ="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
#model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B-GPTQ"
#model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B-AWQ"
#model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B-int4"

# 下载模型到指定目录
snapshot_download(
  repo_id=model_id,
  local_dir="./DeepSeek-Model", # 本地保存路径
  resume_download=True,         # 启用断点续传
  local_dir_use_symlinks=False  # 不使用符号链接，复制实际文件
)

