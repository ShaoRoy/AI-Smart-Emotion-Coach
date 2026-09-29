# AI-Smart-Emotion-Coach

> 本地安装RAG感情问答助手、帮助解决感情疑问。
> 可以自由提问或选热点问题。
> (2.5TB原始数据2100小时视频、5000个语言资料训练出来的数据库)
> 后续会开发接入各APP智能对答功能

---

## ⚠️ 重要提示

**普通用户请勿 clone 源码运行，本仓库仅供代码参考。**

完整索引文件（约 1.7GB）和 AI 模型（约 6GB）**不在仓库中**，
请直接从 [Releases](../../releases) 下载完整版，按下方步骤操作。

---

## 📸 界面预览
<img width="1193" height="858" alt="screenshot" src="https://github.com/user-attachments/assets/cc99d51a-a9c9-4f78-a7bf-b63a39223492" />
<img width="1573" height="840" alt="screenshot1" src="https://github.com/user-attachments/assets/af02f152-8e0b-428c-91bf-e9a72c0c271d" />
<img width="502" height="552" alt="screenshot2" src="https://github.com/user-attachments/assets/0ba287ef-03bb-41bc-85d8-361a39b40fc9" />
<img width="2048" height="2048" alt="screenshot3" src="https://github.com/user-attachments/assets/c4ce1c48-6e6a-4ff7-a83a-00dbeb166a69" />

---

## ✨ 功能特性

- **本地运行**：所有推理在本地完成，数据不上传云端
- **混合检索**：向量语义检索 + BM25 关键词检索
- **重排序增强**：BGE-Reranker 精细重排，回答更精准
- **多轮对话**：记住最近 5 轮对话上下文
- **阶段推荐**：按感情阶段智能推荐热点问题
- **桌面 GUI**：深色 / 浅色主题一键切换
- **完全免费**：无任何收费功能

---

## 🚀 快速开始

### 方式一：AI 引导安装（推荐小白用户）

不会装？把下面**整段提示词**复制给 **豆包 / Kimi / ChatGPT**，
AI 会一步步引导你完成安装。

---

**请复制以下内容给 AI：**

```
我要在你的指导下安装一个本地 AI 问答软件"情感助手"。
请按照下面的安装步骤，一步步引导我操作，每完成一步
让我回复"下一步"。如果遇到报错，我会把错误发给你，
请根据错误给出解决方案。

【软件信息】
- 名称：情感助手（本地 RAG 智能问答系统）
- 平台：Windows 10/11 64位
- 分发形式：7z 分卷压缩包 + 独立索引数据
- 不需要 Python 环境，解压即用

【依赖环境】
1. Ollama（本地大模型运行时）
   - 官网：https://ollama.com/download
   - 安装后自动后台运行

2. 大语言模型 qwen2.5:7b（约 4.7GB）
   - 命令行执行：ollama pull qwen2.5:7b
   - 用途：回答用户问题

3. 嵌入模型 bge-m3（约 1.2GB）
   - 命令行执行：ollama pull bge-m3
   - 用途：把用户问题转成向量用于检索

4. 重排序模型 BAAI_bge-reranker-base（约 1.1GB）
   - 模型文件会随主程序包一起提供，无需单独下载
   - 位置：解压后放在"情感助手/models/BAAI_bge-reranker-base/"目录下

【硬件要求】
- 操作系统：Windows 10 / 11（64位）
- 显卡：建议 8GB 显存以上（如 RTX 3060 / 4060 / 5060）
- 内存：16GB 以上
- 硬盘：至少 15GB 可用空间

【安装步骤】
第一步：从 GitHub Release 页面下载
        - 情感助手.7z.001
        - 情感助手.7z.002
        - 情感助手.7z.003 
        - （所有分卷必须全部下载）

第二步：用 7-Zip 解压
        - 官网：https://www.7-zip.org/
        - 右键 .7z.001 文件 → 7-Zip → 解压到当前文件夹
        - 会自动合并所有分卷

第三步：安装 Ollama
        - 从 https://ollama.com/download 下载 Windows 版
        - 双击安装，一路默认即可
        - 安装后任务栏会出现 Ollama 图标，说明已启动

第四步：下载 AI 模型（打开命令行操作）
        - 按 Win + R 输入 cmd 回车，打开命令行
        - 依次执行两条命令（每条约需 5~20 分钟）：
          ollama pull qwen2.5:7b
          ollama pull bge-m3
        - 下载完成后执行 ollama list 确认两个模型都在

第五步：运行程序
        - 双击解压后的"情感助手.exe"
        - 首次启动会解密索引并加载模型（约 10~30 秒）
        - 看到主界面即安装成功

【常见问题】
Q1：双击 exe 闪退怎么办？
A：先确认 Ollama 是否在后台运行；再确认两个模型是否已下载。
   打开命令行执行 ollama list 查看。

Q2：提示"未找到嵌入模型"怎么办？
A：执行 ollama pull bge-m3 重新下载。

Q3：杀毒软件报毒怎么办？
A：PyInstaller 打包程序可能误报，添加信任即可。

Q4：回答速度很慢怎么办？
A：检查显卡驱动是否安装、显存是否足够。
   任务管理器看 GPU 占用，如果 Ollama 没走 GPU 会慢。

Q5：解压时报错"分卷不完整"？
A：所有 .7z.00X 文件必须全部下载到同一个文件夹里，
   只解压第一个（.001）即可。

【开始】
请先告诉我第一步要做什么，我会执行后回复你。
```

**复制完上面这段提示词后，AI 会自动开始引导你。**

---

### 方式二：手动安装

#### 第 1 步：下载程序

进入 [Releases](../../releases) 页面，下载最新的：

- `情感助手.7z.001`
- `情感助手.7z.002`
- `情感助手.7z.003`

（所有分卷必须全部下载）

#### 第 2 步：解压

用 [7-Zip](https://www.7-zip.org/) 或 WinRAR：

- 右键 `情感助手.7z.001` → 解压到当前文件夹
- 会自动合并所有分卷

解压后得到 `情感助手` 文件夹。

#### 第 3 步：安装 Ollama

访问 https://ollama.com/download 下载并安装。

安装后启动 Ollama（一般在后台自动运行）。

#### 第 4 步：下载 AI 模型

打开命令行（Win + R，输入 `cmd`，回车），依次运行：

```cmd
ollama pull qwen2.5:7b
ollama pull bge-m3
```

- 第一个约 4.7GB，第二个约 1.2GB
- 下载时间取决于网速，请耐心等待

#### 第 5 步：运行程序

双击 `情感助手.exe` 即可。

首次启动会解密索引并加载模型（约 10~30 秒），之后启动就快了。

---

## 📋 环境要求


| 项目     | 要求                                      |
| -------- | ----------------------------------------- |
| 操作系统 | Windows 10 / 11（64位）                   |
| 显卡     | 8GB 显存以上（如 RTX 3060 / 4060 / 5060） |
| 内存     | 16GB 以上                                 |
| 硬盘     | 至少 15GB 可用空间                        |
| Ollama   | 必须安装                                  |

**没有独显可以用吗？** 可以，但速度会慢很多，建议至少 8GB 显存。

---

## ❓ 常见问题

### Q1：双击 exe 没反应 / 闪退

- 确认已安装 Ollama，并且它在运行
- 确认下载了 `qwen2.5:7b` 和 `bge-m3` 两个模型
- 确认 `index_data` 文件夹和 exe 在同一个目录

### Q2：提示"找不到模型"

在命令行运行：

```cmd
ollama list
```

看是否包含 `qwen2.5:7b` 和 `bge-m3`。如果没有，运行：

```cmd
ollama pull qwen2.5:7b
ollama pull bge-m3
```

### Q3：杀毒软件报毒

PyInstaller 打包的 exe 容易被误报。请添加信任，或临时关闭杀毒软件。

### Q4：回答速度很慢

- 检查是否使用了 GPU（任务管理器看 GPU 占用）
- 显存不足会自动降级到 CPU，速度会慢很多
- 关闭其他占用显存的程序

### Q5：首次启动很慢

首次启动需要解密索引并加载模型，约 10~30 秒。这是正常现象。

### Q6：怎么卸载

直接删除 `情感助手` 文件夹即可。程序不会写注册表，不残留任何数据。

---

## 📁 目录结构

```
情感助手/
├── 情感助手.exe              主程序
├── index_data/               加密索引数据
│   ├── integrated_index.faiss.enc
│   ├── integrated_chunks.pkl.enc
│   └── bm25_cache.pkl.enc
├── models/                   重排序模型
│   └── BAAI_bge-reranker-base/
├── themes/                   主题文件
├── stage_data.json           阶段数据
├── quick_phrases.json        快捷短语
├── wechat.png                
├── alipay.png                
└── _internal/                运行依赖
```

---

## 🛠️ 技术栈

Python · FAISS · BGE-M3 · BGE-Reranker · Qwen2.5 · Ollama · CustomTkinter · PyTorch · Transformers

---

## 📖 开发者：源码运行

本项目源码公开，可自行研究学习。

### 环境要求

- Python 3.11（推荐）
- Ollama 已安装并运行
- 已 pull `qwen2.5:7b` 和 `bge-m3`

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行

```bash
python 情感导师GUI.py
```

**注意**：源码运行需要自己准备 `index_data/` 索引文件，
仓库中不包含。索引文件可通过 [Releases](../../releases) 下载。

---

## ☕ 支持作者

如果这个工具帮到了你，欢迎请我喝杯奶茶：


|        微信        |        支付宝        |
| :-----------------: | :-------------------: |
| ![微信](wechat.png) | ![支付宝](alipay.png) |

**感谢支持，你的支持是我研发的动力。**

---

## 📮 合作 / 反馈

- 邮箱：yingshao113113@163.com
- GitHub：https://github.com/ShaoRoy

有问题欢迎提 [Issue](../../issues)，或发邮件联系。

---

## 📄 许可证

本项目采用 **CC BY-NC-ND 4.0** 许可证。

- ✅ 允许个人免费使用
- ✅ 允许分享给他人
- ❌ 禁止商业用途
- ❌ 禁止修改后分发
- ✅ 必须保留原作者署名

详见 [LICENSE](LICENSE) 文件。

---

## 👤 作者

**RoyShao**

- GitHub：[@ShaoRoy](https://github.com/ShaoRoy)
- 邮箱：yingshao113113@163.com

---

## ⭐ 支持项目

如果觉得这个项目对你有帮助，欢迎点个 **Star** ⭐

你的 Star 是我更新的最大动力。
