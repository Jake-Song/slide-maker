# Slide Maker
- LLM Slide Maker inspired by notebookLM
- Given text, make a slide show with LLM  

## demo
- Text: Pick Karpathy's blog post. It shows retrospective view of introducing OpenAI

    [Original Link](https://karpathy.ai/hncapsule/2015-12-11/index.html#article-10720176)

- Slide: 

| | |
| --- | --- |
| ![](/demo/output_0.png) | ![](/demo/output_1.png)
| ![](/demo/output_2.png) | ![](/demo/output_3.png)
| ![](/demo/output_4.png) | ![](/demo/output_5.png)
| ![](/demo/output_6.png) | ![](/demo/output_7.png)
| ![](/demo/output_8.png) | 

## Installation
### install uv

macOS/Linux:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows:
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### install packages

```bash
# install packages
uv sync

# including dev packages
uv sync --dev
```