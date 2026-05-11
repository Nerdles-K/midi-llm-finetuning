# 部署到 Hugging Face Spaces 步骤

## 0. 一次性准备

```powershell
# 装 git-lfs（用来传 2.5GB 模型权重）
winget install Git.LFS
git lfs install
```

## 1. 在 HF 网页上创建 Space

1. 登录 https://huggingface.co
2. 右上角头像 → New Space
3. 填：
   - Space name: `midi-llm-demo`（自己取，将出现在 URL 里）
   - License: `apache-2.0`
   - SDK: **Gradio**
   - Hardware: **CPU basic** (free)
   - Visibility: **Public**
4. 创建后会得到一个空仓库，URL 类似：
   `https://huggingface.co/spaces/<你的用户名>/midi-llm-demo`

## 2. clone 这个空 Space

```powershell
cd d:\University\course_materials\dda4220\MIDI-LLM\webapp
git clone https://huggingface.co/spaces/<你的用户名>/midi-llm-demo
cd midi-llm-demo
```

首次会要 HF 的 access token：到 https://huggingface.co/settings/tokens 创建一个 write 权限的 token，密码处粘贴。

## 3. 拷贝部署文件 + 模型权重

```powershell
# 拷贝 hf_deploy 下的所有部署文件
Copy-Item ..\hf_deploy\app.py .
Copy-Item ..\hf_deploy\requirements.txt .
Copy-Item ..\hf_deploy\README.md . -Force
Copy-Item ..\hf_deploy\.gitattributes .

# 拷贝整个 merged_model 文件夹
Copy-Item ..\merged_model . -Recurse
```

## 4. push 到 HF（会触发自动构建）

```powershell
git add .gitattributes
git add app.py requirements.txt README.md
git add merged_model/
git commit -m "Initial deploy: MIDI-LLM demo"
git push
```

push 时 git-lfs 会自动处理大文件，可能需要几分钟（看你网速，2.5GB 上传）。

## 5. 等构建完成

push 完成后到你的 Space 页面，会看到 "Building" 状态。
首次构建约 **15-25 分钟**（装依赖 + 加载模型）。
完成后状态变成 "Running"，URL 就生效了。

## 6. 公网 URL

```
https://huggingface.co/spaces/<你的用户名>/midi-llm-demo
```

把这个 URL 放简历里、发给别人，谁都能打开使用。

---

## 后续升级（如果 CPU 太慢）

到 Space 的 Settings → Hardware：
- 升级到 **CPU upgrade** ($0.03/小时) - 8 vCPU，速度提升 2-3 倍
- 升级到 **T4 small** ($0.40/小时) - 真正的 GPU，10x 加速
- 申请 **ZeroGPU** (需 PRO 订阅 $9/月) - 共享 A100，按需调用

升级硬件不需要改代码，HF 会自动重新部署。
