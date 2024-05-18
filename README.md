<h1 class="code-line" data-line-start=0 data-line-end=1><a id="MiniMedGPT_0"></a>MiniMedGPT: Efficient Large Vision-Language Model For Medical Visual Question Answering</h1>
<h2 class="code-line" data-line-start=1 data-line-end=2><a id="_VQA_System_for_Neuroradiology_Brain_Tumors_and_Lung_Cancer_Screening_Using_MRI_and_CT_Scans__1"></a><em>VQA System for Neuroradiology, Brain Tumors, and Lung Cancer Screening and More Using MRI and CT Scans.</em></h2>
<p class="has-line-data" data-line-start="2" data-line-end="3">MiniMedGPT is designed to assist radiologists by using AI to interpret MRI and CT scans, specifically for brain tumors and lung cancer. This AI-driven chatbot aims to provide real-time answers to clinical questions, improve diagnostic accuracy, and enhance the efficiency of medical imaging workflows. By integrating such technology, the goal is to reduce the heavy workload on radiologists, expedite the diagnostic process, and ultimately improve patient outcomes through earlier and more precise detection of critical health issues.</p>


<h2 class="code-line" data-line-start=5 data-line-end=6><a id="Installation_5"></a>Installation</h2>
<p class="has-line-data" data-line-start="6" data-line-end="8"><strong>1- Prepare the code and the environment</strong><br>
Git clone our repository, create a Python environment, and activate it via the following commands:</p>
<pre><code class="has-line-data" data-line-start="9" data-line-end="14" class="language-sh">git clone 
cd MiniMedGPT
conda env create -f environment.yml
conda activate MiniMedGPT
</code></pre>
<p class="has-line-data" data-line-start="14" data-line-end="21"><strong>2- Prepare the pretrained LLM weights</strong><br>
MiniMedGPT is a finetuned model of MiniGPT-v2, which is based on Llama2 Chat 7B. Download the LLM weights from the following Hugging Face space via cloning the repository using git-lfs. <a href="https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/tree/main">Download</a><br>
<strong>3- Prepare the pretrained model checkpoints</strong><br>
Download the pretrained model checkpoints.<br>
<a href="https://drive.google.com/file/d/1-uwRLa3xrD2h15UbdS8-gznx8UDH2zNY/view?usp=sharing">Download</a><br>
Set the path to the pretrained checkpoint in the evaluation config file <a href="https://github.com/SabbaghCodes/MiniMedGPT/blob/main/eval_configs/minigptv2_eval.yaml">eval_configs/minigptv2_eval.yaml</a> at Line 8.<br>
<a href="https://drive.google.com/file/d/11nAPjEok8eAGGEG1N2vXo3kBLCg0WgUk/view">Download</a> our pretrained model.</p>

<h2 class="code-line" data-line-start=22 data-line-end=23><a id="Launching_Demo_Locally_22"></a>Launching Demo Locally</h2>
<p class="has-line-data" data-line-start="23" data-line-end="24">Run:</p>
<pre><code class="has-line-data" data-line-start="25" data-line-end="27" class="language-sh">python demo_v2.py --cfg-path eval_configs/minigptv2_eval.yaml --gpu-id 0
</code></pre>
<p class="has-line-data" data-line-start="27" data-line-end="29">To save GPU memory, LLMs load as 8-bit by default, with a beam search width of 1. This configuration requires about 23G GPU memory for 13B LLM and 11.5G GPU memory for 7B LLM. For more powerful GPUs, you can run the model in 16-bit by setting <code>low_resource</code> to <code>False</code> in the relevant config file: <a href="https://github.com/SabbaghCodes/MiniMedGPT/blob/main/eval_configs/minigptv2_eval.yaml">minigptv2_eval.yaml</a></p>

<h2 class="code-line" data-line-start=30 data-line-end=31><a id="Training_30"></a>Training</h2>
<p class="has-line-data" data-line-start="31" data-line-end="33">You can download the data we used <a href="https://www.med-vqa.com/slake/">here</a>.<br>
In the train_configs/minigptv2_finetune.yaml, you need to set up the following paths:</p>
<p class="has-line-data" data-line-start="34" data-line-end="37">llama_model checkpoint path: ```/path/to/llama_checkpoint```<br>
ckpt: ```/path/to/pretrained_checkpoint```<br>
ckpt save path: ```/path/to/save_checkpoint```</p>
<p class="has-line-data" data-line-start="38" data-line-end="40">For ckpt, you may load from our pretrained model checkpoints:<br>
<a href="https://drive.google.com/file/d/1-uwRLa3xrD2h15UbdS8-gznx8UDH2zNY/view?usp=sharing">Download</a></p>
<pre><code class="has-line-data" data-line-start="41" data-line-end="43" class="language-sh">torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigptv2_finetune.yaml
</code></pre>

<h2 class="code-line" data-line-start=43 data-line-end=44><a id="Evaluation_43"></a>Evaluation</h2>
<p class="has-line-data" data-line-start="44" data-line-end="45"><strong>Evaluation dataset structure</strong></p>
<pre><code class="has-line-data" data-line-start="46" data-line-end="52" class="language-sh">├── Slake
│   └── imgs
│   ├── train.json
│   ├── validate.json
│   ├── test.json
</code></pre>

<h2 class="code-line" data-line-start=52 data-line-end=53><a id="Requirements_52"></a>Requirements</h2>
<ul>
<li class="has-line-data" data-line-start="53" data-line-end="54">python 3.8</li>
<li class="has-line-data" data-line-start="54" data-line-end="55">scikit-image (visit <a href="https://scikit-image.org/docs/stable/user_guide/install.html">this page</a> for installation)</li>
<li class="has-line-data" data-line-start="55" data-line-end="57">matplotlib (visit <a href="https://matplotlib.org/stable/users/installing/index.html">this page</a> for installation)</li>
</ul>

<h2 class="code-line" data-line-start=3 data-line-end=4><a id="Demo_3"></a>Examples</h2>
<p class="has-line-data" data-line-start="4" data-line-end="5">Check out the examples below:</p>
<img src="https://github.com/SabbaghCodes/MiniMedGPT/blob/main/examples/example1.jpg?raw=true" alt="Example 1">
<img src="https://github.com/SabbaghCodes/MiniMedGPT/blob/main/examples/example2.jpg?raw=true" alt="Example 2">
<img src="https://github.com/SabbaghCodes/MiniMedGPT/blob/main/examples/example3.jpg?raw=true" alt="Example 3">

<h2 class="code-line" data-line-start=57 data-line-end=58><a id="Acknowledgement_57"></a>Acknowledgement</h2>
<p class="has-line-data" data-line-start="58" data-line-end="59">Thanks to the creators of MiniGPT-v2, check out their website <a href="https://minigpt-4.github.io/">here</a>.</p>
