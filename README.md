
<br><br>
<h1 align="center">Chain-of-Exemplar: Enhancing Distractor Generation for Multimodal
Educational Question Generation</h1>
<p align="center">
<br>
<a href="assets/wechat.jpg">WeChat</a>&nbsp&nbsp | &nbsp&nbsp<a href="https://huggingface.co/datasets/Lhh123/CoE_ScienceQA">Dataset</a>&nbsp&nbsp | &nbsp&nbsp<a href="https://huggingface.co/Lhh123/coe_multitask_blip2xl_angle_2ep">Checkpoints</a>&nbsp ｜ &nbsp<a href="https://aclanthology.org/2024.acl-long.432.pdf">Paper</a>&nbsp&nbsp
</p>
<br><br>

**Chain-of-Exemplar** is a multimodal educational multi-choice question generation framework, which utilizes multimodal large language models (MLLMs) with Chain-of-Thought reasoning to improve the generation of challenging distractors. Furthermore, CoE leverages three-stage contextualized exemplar retrieval to retrieve exemplary questions as guides for generating more subject-specific educational questions. Experimental results on the ScienceQA benchmark demonstrate the superiority of CoE in both question generation and distractor generation over existing methods across various subjects and educational levels.

<br>
<p align="center">
    <img src="assets/method_00.png"/>
<p>
<br>


## News and Updates
* ```2024.8.16``` **Chain-of-Exemplar** has been accepted to ACL'24!

## Requirements

* python 3.8 and above
* pytorch 1.12 and above, 2.0 and above are recommended
* CUDA 11.4 and above are recommended (this is for GPU users)
  <br>

## Quickstart

Below, we provide simple examples to show how to use Chain-of-Exemplar with 🤖 ModelScope and 🤗 Transformers.

Before running the code, make sure you have setup the environment and installed the required packages. Make sure you meet the above requirements, and then install the dependent libraries.

```bash
pip install -r requirements.txt
```

Now you can start with Transformers or ModelScope(updating...).

#### 🤗 Transformers

To use Chain-of-Exemplar for the inference, all you need to do is to input a few lines of codes as demonstrated below. However, **please make sure that you are using the latest code.**

```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('Lhh123/coe_multitask_blip2xl_angle_2ep', trust_remote_code=True)
model = AutoPeftModelForCausalLM.from_pretrained(
    'Lhh123/coe_multitask_blip2xl_angle_2ep', # path to the output directory
    device_map="cuda",
    trust_remote_code=True
).eval()

query = "Picture: <img>YOUR_IMAGE_PATH</img>\nGenerate a question based on the picture."
response, history = model.chat(tokenizer, query=query, history=None)
print(response)
```

#### 🤖 ModelScope(updating...)

ModelScope is an opensource platform for Model-as-a-Service (MaaS), which provides flexible and cost-effective model service to AI developers. Similarly, you can run the models with ModelScope as shown below:

```python
from modelscope import (
    snapshot_download, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
)
model_id = 'Lhh123/coe_multitask_blip2xl_angle_2ep'

model_dir = snapshot_download(model_id, revision=revision)


```

## Finetuning

Now we provide the official training script, `finetune.py`, for users to finetune the pretrained model for downstream applications in a simple fashion. Additionally, we provide shell scripts to launch finetuning with no worries. This script supports the training with DeepSpeed and FSDP. The shell scripts that we provide use DeepSpeed, and thus we advise you to install DeepSpeed before you start:

```bash
pip install deepspeed
```

### Data preparation
To prepare your training data, you need to put all the samples into a list and save it to a json file. We provide our train&val datasets (with our own root) [here](https://huggingface.co/datasets/Lhh123/CoE_ScienceQA). You need to change the image path to your own path instead of directly using it. Alternatively, we also provide the original dataset [ScienceQA](https://scienceqa.github.io/) with [train](https://huggingface.co/datasets/Lhh123/CoE_ScienceQA/tree/main/train) & [val](https://huggingface.co/datasets/Lhh123/CoE_ScienceQA/tree/main/val) & [test](https://huggingface.co/datasets/Lhh123/CoE_ScienceQA/tree/main/test) split. You can download and convert them into the following format by running:
```python
# Question generation data pre-process
python prepare_qg.py

# Rationale generation data pre-process
python prepare_rg.py

# Distractor generation data pre-process
python prepare_dg.py
```
Subsequently, in **Contextualized Exemplar Retrieval** module, you can retrieve the top-k similar samples to serve as exemplars by running:
```python
python retrieve.py
```

Each sample is a dictionary consisting of an id and a list for conversation. Below is a simple example list with 1 sample:
```json
[
  {
    "id": "identity_0",
    "conversations": [
      {
        "from": "user",
        "value": "Answer: apostrophe\nPlease generate a question from the corresponding answer."
      },
      {
        "from": "assistant",
        "value": "Which figure of speech is used in this text?\nSing, O goddess, the anger of Achilles son of Peleus"
      }
    ]
  },
  {
    "id": "identity_1",
    "conversations": [
      {
        "from": "user",
        "value": "Picture: <img>/data/luohh/mm-cot/data/scienceqa/test/11/image.png</img>\nAnswer: New Hampshire\nPlease generate a question from this picture and the corresponding answer."
      },
      {
        "from": "assistant",
        "value": "What is the name of the colony shown?"
      },
    ]
  },
```
For the VL tasks, there are special tokens that are used, including `<img> </img>`.

The picture is represented as `Picture id: <img>img_path</img>\n{your prompt}`, where `id` indicates the position of the image in the conversation, starting from 1. The "img_path" can be a local file path or a web link. 


After data preparation, you can use the provided shell scripts to run finetuning. Remember to specify the path to the data file, `$DATA`.

The finetuning scripts allow you to perform:
- Full-parameter finetuning
- LoRA
- Q-LoRA

In addition to our checkpoints we provided, you can also conduct finetuning on [Qwen-VL-Chat-Int4](https://huggingface.co/Qwen/Qwen-VL-Chat)(our base model).

### Full-parameter finetuning
Full-parameter parameter finetuning requires updating all parameters of LLM in the whole training process. In our experiments, frozening the parameters of ViT during the fine-tuning phase achieves better performance. To launch your training, run the following script:

```bash
sh finetune/finetune_ds.sh
```

Remember to specify the correct model name or path, the data path, as well as the output directory in the shell scripts. If you want to make changes, just remove the argument `--deepspeed` or make changes in the DeepSpeed configuration json file based on your requirements. Additionally, this script supports mixed-precision training, and thus you can use `--bf16 True` or `--fp16 True`. Empirically we advise you to use bf16 to make your training consistent with our pretraining and alignment if your machine supports bf16, and thus we use it by default.

### LoRA
Similarly, to run LoRA, use another script to run as shown below. Before you start, make sure that you have installed `peft`. Also, you need to specify your paths to your model, data, and output. We advise you to use absolute path for your pretrained model. This is because LoRA only saves the adapter and the absolute path in the adapter configuration json file is used for finding out the pretrained model to load.

```bash
# Single GPU training
sh finetune/finetune_lora_single_gpu.sh
# Distributed training
sh finetune/finetune_lora_ds.sh
```

In comparison with full-parameter finetuning, LoRA ([paper](https://arxiv.org/abs/2106.09685)) only updates the parameters of adapter layers but keeps the original large language model layers frozen. This allows much fewer memory costs and thus fewer computation costs. 

Note that if you use LoRA to finetune the base language model, e.g., Qwen-VL, instead of chat models, e.g., Qwen-VL-Chat, the script automatically switches the embedding and output layer as trainable parameters. This is because the base language model has no knowledge of special tokens brought by ChatML format. Thus these layers should be updated for the model to understand and predict the tokens. Or in another word, if your training brings in special tokens in LoRA, you should set the layers to trainable parameters by setting `modules_to_save` inside the code. Additionally, we find that there is a significant gap between the memory footprint of LoRA with and without these trainable parameters. Therefore, if you have trouble with memory, we advise you to LoRA finetune the chat models. Check the profile below for more information.

### Q-LoRA
However, if you still suffer from insufficient memory, you can consider Q-LoRA ([paper](https://arxiv.org/abs/2305.14314)), which uses the quantized large language model and other techniques such as paged attention to allow even fewer memory costs. To run Q-LoRA, directly run the following script:

```bash
# Single GPU training
sh finetune/finetune_qlora_single_gpu.sh
# Distributed training
sh finetune/finetune_qlora_ds.sh
```

For Q-LoRA, we advise you to load our provided quantized model, e.g., Qwen-VL-Chat-Int4. 
You **SHOULD NOT** use the bf16 models. Different from full-parameter finetuning and LoRA, only fp16 is supported for Q-LoRA. Besides, for Q-LoRA, the troubles with the special tokens in LoRA still exist. However, as we only provide the Int4 models for chat models, which means the language model has learned the special tokens of ChatML format, you have no worry about the layers. Note that the layers of the Int4 model should not be trainable, and thus if you introduce special tokens in your training, Q-LoRA might not work.



Different from full-parameter finetuning, the training of both LoRA and Q-LoRA only saves the adapter parameters. You can load the finetuned model for inference as shown below:

```python
from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained(
    path_to_adapter, # path to the output directory
    device_map="auto",
    trust_remote_code=True
).eval()
```

If you want to merge the adapters and save the finetuned model as a standalone model (you can only do this with LoRA, and you CANNOT merge the parameters from Q-LoRA), you can run the following codes:

```python
from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained(
    path_to_adapter, # path to the output directory
    device_map="auto",
    trust_remote_code=True
).eval()

merged_model = model.merge_and_unload()
# max_shard_size and safe serialization are not necessary. 
# They respectively work for sharding checkpoint and save the model to safetensors
merged_model.save_pretrained(new_model_directory, max_shard_size="2048MB", safe_serialization=True)
```

Note: For multi-GPU training, you need to specify the proper hyperparameters for distributed training based on your machine. Besides, we advise you to specify your maximum sequence length with the argument --model_max_length, based on your consideration of data, memory footprint, and training speed.


### Profiling of Memory and Speed
We profile the GPU memory and training speed of both LoRA (Base) refers to training the embedding and output layer, while LoRA (Chat) has no trainable embedding and output layer) and Q-LoRA in the setup of single-GPU training. In this test, we experiment on a single A100-SXM4-80G GPU, and we use CUDA 11.8 and Pytorch 2.0. We uniformly use a batch size of 1 and gradient accumulation of 8. Each sample contains an image. We profile the memory (GB) and speed (s/iter) of inputs of different lengths, namely 384, 512, 1024, and 2048. The statistics are listed below:


<table>
    <tr>
 <th rowspan="2">Method</th><th colspan="4" align="center">Sequence Length</th>
    </tr>
    <tr>
        <th align="center">384</th><th align="center">512</th><th align="center">1024</th><th align="center">2048</th>
    </tr>
    <tr>
      <td>LoRA (Base)</td><td align="center">37.1G / 2.3s/it</td><td align="center">37.3G / 2.4s/it</td><td align="center">38.7G / 3.6s/it</td><td align="center">38.7G / 6.1s/it</td>
    </tr>
    <tr>
      <td>LoRA (Chat)</td><td align="center">23.3G / 2.2s/it</td><td align="center">23.6G / 2.3s/it</td><td align="center">25.1G / 3.5s/it</td><td align="center">27.3G / 5.9s/it</td>
    </tr>
    <tr>
        <td>Q-LoRA</td><td align="center">17.0G / 4.2s/it</td><td align="center">17.2G / 4.5s/it</td><td align="center">18.2G / 5.5s/it</td><td align="center">19.3G / 7.9s/it</td>
    </tr>

</table>

<br>

## Evaluation

Evaluate the quality of generated questions and distractors by running:
```python
python scoring.py
```

## Citation

If you find our paper and code useful in your research, please consider giving a star :star: and citation :pencil: :)

```BibTeX
@inproceedings{luo2024chain,
  title={Chain-of-Exemplar: Enhancing Distractor Generation for Multimodal Educational Question Generation},
  author={Luo, Haohao and Deng, Yang and Shen, Ying and Ng, See Kiong and Chua, Tat-Seng},
  booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={7978--7993},
  year={2024}
}
```

<br>

## Contact Us

If you are interested to leave a message to either our research team or product team, feel free to send an email to luohh5@mail2.sysu.edu.cn.

