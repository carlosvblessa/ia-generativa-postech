# POSTECH MLE - Módulo 4: IA Generativa

Repositório com os notebooks, roteiros e apresentações utilizados nas aulas de IA Generativa do curso de Machine Learning Engineering (MLE) da POSTECH. Os materiais conectam fundamentos de attention, GANs, difusão e large language models com implementações práticas em NumPy, TensorFlow, PyTorch, Diffusers e Transformers.

## Estrutura
- `Aula01-AttentionMechanism.pdf` a `Aula05-MidJourney_Dall-E_GPT 4.pdf`: slides oficiais das cinco aulas (Attention, Arquiteturas Generativas, Diffusion/Stable Diffusion, LLMs e ferramentas multimodais).
- `Aula1/`: notebooks introdutórios sobre self-attention, cross-attention e mecanismos de atenção clássicos implementados em NumPy (`Attention_Mechanisms.ipynb`, `Self_attention.ipynb`, `Cross_Attention.ipynb`).
- `Aula2/`: `Redes_Generativas_Adversariais_.ipynb` traz o passo a passo de uma DCGAN em TensorFlow/Keras treinada com MNIST.
- `Aula3/`: dois fluxos práticos com difusão (`stable_diffusion_2.ipynb` para inferência com Diffusers e `stable_diffusion_CIFAR10.ipynb` para treino de um gerador simples sobre CIFAR-10 usando PyTorch).
- `Aula4/`: notebooks focados em NLP/LLMs (`fine_tuning.ipynb` mostra fine-tuning do BERT em IMDB com Hugging Face datasets/Trainer, enquanto `llm.ipynb` demonstra geração com `pipeline('text-generation')`).

## Conteúdo das aulas
- **Aula 01 – Attention Mechanisms:** revisa o conceito de pesos de atenção, softmax e operações de self vs. cross-attention, com exemplos vetoriais simples para entendimento matemático.
- **Aula 02 – Arquiteturas Generativas:** constrói uma DCGAN (gerador + discriminador + loop de treino) em TensorFlow, normaliza MNIST em [-1, 1], monitora perdas e gera grids de dígitos sintéticos.
- **Aula 03 – Modelos de Difusão:** cobre tanto o uso de modelos pré-treinados (Stable Diffusion v1-4 via `diffusers`) quanto a implementação de um gerador treinado em CIFAR-10 com PyTorch/Torchvision e training loop instrumentado com `tqdm`.
- **Aula 04 – LLMs e Fine-tuning:** utiliza a API `pipeline` para geração com GPT-2, discute hiperparâmetros (temperatura/top-k/top-p) e mostra preparação de datasets HF, tokenização, Trainer/TrainingArguments e avaliação (`load_metric`) para ajustar um BERT em sentimentos.
- **Aula 05 – Plataformas multimodais:** detalhado nos slides (`Aula05-MidJourney_Dall-E_GPT 4.pdf`) com foco em ferramentas SaaS (Midjourney, DALL·E, GPT-4) e boas práticas operacionais.

## Requisitos e preparação
- Python 3.10+ com `pip` recente.
- GPU recomendada para notebooks de GANs, difusão e fine-tuning (CUDA disponível para PyTorch/TensorFlow ou acesso ao Colab).
- Dependências listadas em `requirements.txt`:
  - Núcleo científico: `numpy`, `matplotlib`.
  - IA generativa: `tensorflow`, `torch`, `torchvision`, `diffusers`, `transformers`, `datasets`, `accelerate`, `safetensors`.
  - Utilidades: `tqdm`, `google-colab`, `jupyter`.

### Setup (venv + pip)
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
> Caso utilize GPU, instale a variante correta de PyTorch/TensorFlow seguindo as instruções oficiais antes de rodar `pip install -r requirements.txt`.

### Tokens e downloads
- Para `StableDiffusionPipeline`, configure o token da Hugging Face (`huggingface-cli login`) e aceite os termos do modelo `CompVis/stable-diffusion-v1-4`.
- O notebook de fine-tuning baixa automaticamente o dataset `imdb`; redirecione `HF_HOME` se precisar controlar o cache.
- O exemplo de GAN usa MNIST via `tf.keras.datasets`, baixado na primeira execução.

## Como executar os notebooks
1. Ative o ambiente (ou abra no Google Colab) e rode `jupyter lab` para abrir os arquivos desejados.
2. **Aula1:** execute sequencialmente para visualizar operações matriciais de atenção; nenhum dataset externo é necessário.
3. **Aula2:** confirme acesso à GPU, rode as células de setup (instalação do TensorFlow), carregue MNIST e acompanhe o treinamento da DCGAN. Ajuste `epochs`/`BUFFER_SIZE` para execuções mais rápidas.
4. **Aula3:** 
   - `stable_diffusion_2.ipynb`: após autenticar na Hugging Face, substitua o `prompt` e execute para gerar/baixar imagens.
   - `stable_diffusion_CIFAR10.ipynb`: instala PyTorch/Torchvision, prepara CIFAR-10, define gerador/discriminador e realiza o treino com progress bar do `tqdm`.
5. **Aula4:** 
   - `llm.ipynb`: instala `transformers`, monta o `pipeline('text-generation')` e experimenta variações de hiperparâmetros.
   - `fine_tuning.ipynb`: baixa IMDB, cria tokenizador, define `Trainer` e avalia métricas (accuracy). Ajuste `num_train_epochs`/`per_device_train_batch_size` conforme o hardware.

## Boas práticas
- Prefira ambientes isolados (venv ou Conda) para separar dependências pesadas (TensorFlow, PyTorch, Diffusers).
- Use GPU sempre que possível: `torch.cuda.is_available()` define automaticamente `device` nos notebooks; em CPU, aumente `gradient_accumulation_steps` e reduza `batch_size`.
- Monitore uso de VRAM com modelos de difusão; desative `safetensors`/`fp16` apenas se os drivers suportarem.
- Para execuções em Colab, mantenha as células `!pip install` no topo e considere salvar pesos/outputs no Google Drive.
- Atualize os tokens/API keys da Hugging Face em variáveis de ambiente, evitando hardcode nos notebooks.

## Licença
Material educacional destinado às turmas de Machine Learning Engineering da POSTECH. Consulte os termos do curso antes de redistribuir.
