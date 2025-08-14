import torch
from transformers import AutoTokenizer
from transformers.models.qwen3 import modeling_qwen3
from openlm_hub import repo_download
from lxt.efficient import monkey_patch
from lxt.utils import latex_heatmap, clean_tokens
import datetime
import os

# modify the Qwen3 module to compute LRP in the backward pass
monkey_patch(modeling_qwen3, verbose=True)

model_name = 'Qwen/Qwen3-1.7B'
path = repo_download(model_name)
# Load model without quantization
model = modeling_qwen3.Qwen3ForCausalLM.from_pretrained(path, device_map='cuda', torch_dtype=torch.bfloat16)

# optional gradient checkpointing to save memory (2x forward pass)
model.train()
model.gradient_checkpointing_enable()

# deactive gradients on parameters to save memory
for param in model.parameters():
    param.requires_grad = False

tokenizer = AutoTokenizer.from_pretrained(path)

prompt = """背景：珠穆朗玛峰吸引了许多登山者，包括经验丰富的登山者。有两条主要的攀登路线，一条从尼泊尔东南部（称为标准路线）接近峰顶，另一条从西藏北部接近峰顶。虽然在标准路线上没有构成实质性的技术攀登挑战，但珠穆朗玛峰存在高原反应、天气和风等危险，以及雪崩和昆布冰川的危险。截至2022年11月，已有310人在珠穆朗玛峰上丧生。200多具尸体仍留在山上，由于情况危险，尚未被移走。英国登山者首次登上珠穆朗玛峰。由于尼泊尔当时不允许外国人进入该国，英国人从西藏一侧在北岭路线上进行了几次尝试。1921年，英国首次侦察探险队在北坳达到7000米（22970英尺）后，1922年的探险队将北岭路线推高至8320米（27300英尺），标志着人类首次攀登8000米（26247英尺）以上。1924年的探险带来了迄今为止珠穆朗玛峰上最大的谜团之一：乔治·马洛里和安德鲁·欧文于6月8日进行了最后一次登顶尝试，但从未返回，引发了关于他们是否是第一个登顶的争论。1953年，丹增·诺吉和埃德蒙·希拉里使用东南山脊路线首次登上珠穆朗玛峰。前一年，诺盖作为1952年瑞士探险队的一员，已经到达了8595米（28199英尺）。1960年5月25日，中国登山队王福州、贡布和曲银华首次从北岭登顶。 \
问：1922年他们爬了多高？根据文本，1922年的探险队到达了8，"""

# get input embeddings so that we can compute gradients w.r.t. input embeddings
input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(model.device)
input_embeds = model.get_input_embeddings()(input_ids)

# inference and get the maximum logit at the last position (we can also explain other tokens)
output_logits = model(inputs_embeds=input_embeds.requires_grad_(), use_cache=False).logits
max_logits, max_indices = torch.max(output_logits[0, -1, :], dim=-1)

# Backward pass (the relevance is initialized with the value of max_logits)
# This initiates the LRP computation through the network
max_logits.backward()

# obtain relevance by computing Input * Gradient
relevance = (input_embeds * input_embeds.grad).float().sum(-1).detach().cpu()[0] # cast to float32 before summation for higher precision

# normalize relevance between [-1, 1] for plotting
relevance = relevance / relevance.abs().max()

# remove special characters from token strings and plot the heatmap
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
tokens = clean_tokens(tokens)

# Create latex_files directory if it doesn't exist
os.makedirs('latex_files', exist_ok=True)

# Generate timestamp for filenames
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Generate LaTeX files without compiling to PDF
latex_heatmap(tokens, relevance, path=f'latex_files/{model_name.split("/")[-1]}_heatmap_{timestamp}.tex')

# Generate again without first token, because it receives large relevance values overshadowing the rest
latex_heatmap(tokens[1:], relevance[1:] / relevance[1:].max(), path=f'latex_files/qwen3_1.7B_heatmap_wo_first_{timestamp}.tex')