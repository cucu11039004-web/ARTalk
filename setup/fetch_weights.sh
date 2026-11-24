#!/usr/bin/env bash
# 下载推理所需权重并校验 SHA-256，哈希来自 artalk/assets_manifest.json（固定到上游 revision 6fbead66）。
# 校验失败就删除文件并报错。不下载 FLAME_with_eye.pt（FLAME 许可需本人确认；推理系数用不到它）和 GAGAvatar 相关文件。
#   bash setup/fetch_weights.sh
# HF_ENDPOINT 可改镜像：HF_ENDPOINT=https://hf-mirror.com bash setup/fetch_weights.sh
set -euo pipefail

cd "$(dirname "$0")/.."
HF="${HF_ENDPOINT:-https://huggingface.co}"
REV=6fbead6606ecd147865a9f7e1bc3f02940fdd78c
BASE="$HF/xg-chu/ARTalk/resolve/$REV"

fetch() {  # 目标路径  相对路径  SHA-256
    local dest="assets/$1" url="$BASE/$2" sha="$3"
    if [[ -f "$dest" ]] && [[ "$(sha256sum "$dest" | cut -d' ' -f1)" == "$sha" ]]; then
        echo "已存在且校验通过：$dest"
        return
    fi
    mkdir -p "$(dirname "$dest")"
    echo "下载 $dest"
    curl -fL --retry 5 --retry-delay 3 -C - -o "$dest" "$url"
    local got
    got="$(sha256sum "$dest" | cut -d' ' -f1)"
    if [[ "$got" != "$sha" ]]; then
        echo "校验失败：$dest（期望 $sha，实际 $got）" >&2
        rm -f "$dest"
        exit 1
    fi
    echo "校验通过：$dest"
}

fetch config.json config.json 7546248a7eace5184a3b072de6fac80f22109e32f073ba4f616a7971a6f190fe
fetch ARTalk_wav2vec.pt ARTalk_wav2vec.pt d89615a32e2a6b4e35cf5494c1e400b2acedc43812f63423b9f5d705046d67c2  # 1.96 GB

# 风格片段（50 帧 × 106 维，2 秒），共约 21 MB
while read -r name sha; do
    fetch "style_motion/$name.pt" "style_motion/$name.pt" "$sha"
done <<'EOF'
angry_0 4f17f930661c1e42306ecf6e222b370ca2c4f775dbc63bde6402ca2f8135c556
curious_0 f0db8a645684583af9694d72924cad9b4ac4d16d1a3e7c26c979fa9a7a51220d
doubtful_0 65863da3c4cc9fc11604f21c660684864ddfd135f88a84c0f25d0adba25bfc36
doubtful_1 2ea3241f8a9a3bd3f428ba77d76720c32ef8696042d05329af4f7e95a1baecce
happy_0 fdba588c443f1cfb17a11781b37515ababd933f65564a32f6859622ff62d3412
happy_1 3ed4aa1ee540c136e88546a63ae324ac4f677a72a3bd41423fab1a5eea7c30ca
happy_2 0b39a71e934c3bca01dce558e08b1953935392f98f1334e39759e0c9023cbd0f
natural_0 5c929ca5a4f91ef8c4921964648cd2724f9e10d0ca520e0207009a2ed9b2bdcd
natural_1 775273821b842701d41acc17ed8bc022c806c9a44cc77b860213c6e56fe36612
natural_2 7cebd082ba8fa4bb91c1d51c6a7a7c9e26c8892f648c9942f49344d42264de70
natural_3 ae85ce6fbeaa3af633ee1022a6f96a543a88f40becd61a4c5f0bc997ab074078
natural_4 cc76e17d724001d3d047c6d40c9c86a50550e7fb734a11bf043cc1eaad376005
natural_5 838dc72fccb22908dfc61b8965225295a7ae0c84a57f0672da31a48327b09fc0
natural_6 b32a9d6ed07978d375e51f0f8da9ece1b3e4bc8dad64a281f990fe712c54d7eb
natural_7 7ee71908c968e3a02f7d00c4079cbae0227551251da7a08ad2ed6ee074cd42f6
EOF

# wav2vec2 XLS-R 300M 的 config.json：BitwiseARModel 构建时用 from_pretrained 取，只要配置不要权重，缓存到 ~/.cache/huggingface
if command -v python >/dev/null 2>&1 && python -c "import huggingface_hub" 2>/dev/null; then
    echo "缓存 facebook/wav2vec2-xls-r-300m 的 config.json"
    HF_ENDPOINT="$HF" python -c "
from huggingface_hub import hf_hub_download
print(hf_hub_download('facebook/wav2vec2-xls-r-300m', 'config.json'))
"
else
    echo "未检测到装有 huggingface_hub 的 python：先 conda activate artalk 再重跑本脚本，以缓存 wav2vec2 配置" >&2
fi
