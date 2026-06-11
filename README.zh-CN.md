# learn-llm-by-doing（中文）

[English →](README.md)

一套自学课程，通过从零搭建核心部件来真正理解大语言模型是怎么跑起来的，而不是停留在读文档。

涵盖三个方向：

- **训练** —— 从零写一个 GPT，再把它扩展到多张 GPU 上。
- **推理** —— 手写一个精简版的推理引擎（类似 vLLM），并和真实系统对比。
- **Agent** —— multi-agent 系统背后的设计取舍，配一个对照实验。

每个方向拆成若干小模块。每个模块给你：要读什么、动手做什么、以及一组简短的自查题来确认是否真的学懂了。

## 目录结构

```
courses/        课程内容（markdown）：每个模块一个文件夹
build_site.py   把课程内容生成一个可浏览的网站
site/           生成的网站 —— 打开 site/index.html 开始
```

## 怎么用

```bash
open site/index.html        # 浏览课程
```

要修改某个模块，编辑 `courses/` 下的 markdown 再重新生成：

```bash
python3 build_site.py
```

---

> 个人学习项目。目标是理解机制，不是刷 benchmark 数字。
