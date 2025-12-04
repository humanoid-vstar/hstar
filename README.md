# *Thinking in 360°*: Humanoid Visual Search in the Wild

<div class="byline" align="center">
    <div class="byline-container" align="center">
        <div class="authors" style="text-align: center; margin-bottom: 1rem;">
            <div style="margin-bottom: 0.5rem;">
                <a href="https://scholar.google.com/citations?user=GVI6jVsAAAAJ&hl=en" target="_blank" style="color: inherit; text-decoration: none !important;"><span class="author-name">Heyang Yu</span></a><sup>1*</sup>,
                <a href="" target="_blank" style="color: inherit; text-decoration: none !important;"><span class="author-name">Yinan Han</span></a><sup>3*</sup>,
                <a href="https://painkillerzzz.github.io/xiangyu_zhang.github.io/" target="_blank" style="color: inherit; text-decoration: none !important;"><span class="author-name">Xiangyu Zhang</span></a><sup>4</sup>,
                <a href="https://yyyybq.github.io/BaiqiaoYIN.github.io/" target="_blank" style="color: inherit; text-decoration: none !important;"><span class="author-name">Baiqiao Yin</span></a><sup>1</sup>,
                <a href="" target="_blank" style="color: inherit; text-decoration: none !important;"><span class="author-name">Bowen Chang</span></a><sup>1</sup>,
                <a href="https://han-xiangyu.github.io/" target="_blank" style="color: inherit; text-decoration: none !important;"><span class="author-name">Xiangyu Han</span></a><sup>1</sup>,
                <a href="https://gaaaavin.github.io/" target="_blank" style="color: inherit; text-decoration: none !important;"><span class="author-name">Xinhao Liu</span></a><sup>1</sup>,
                <a href="https://jingz6676.github.io/" target="_blank" style="color: inherit; text-decoration: none !important;"><span class="author-name">Jing Zhang</span></a><sup>1</sup>
            </div>
            <div>
                <a href="https://web.stanford.edu/~pavone/index.html" target="_blank" style="color: inherit; text-decoration: none !important;"><span class="author-name"><span class="author-name">Marco Pavone</span></a><sup>2,5</sup>,
                <a href="https://ai4ce.github.io/" target="_blank" style="color: inherit; text-decoration: none !important;"><span class="author-name">Chen Feng</span></a><sup>1&dagger;</sup>,
                <a href="https://www.sainingxie.com/" target="_blank" style="color: inherit; text-decoration: none !important;"><span class="author-name">Saining Xie</span></a><sup>1&dagger;</sup>,
                <a href="https://yimingli-page.github.io/" target="_blank" style="color: inherit; text-decoration: none !important;"><span class="author-name"><span class="author-name">Yiming Li</span></a><sup>1,2&dagger;</sup>
            </div>
        </div>
        <div style="text-align: center; font-size: 1.1rem; color: #666;"><span><sup>*</sup>Equal contribution</span> &nbsp; <span><sup>&dagger;</sup>Corresponding author</span></div>
        <div class="affiliations" style="text-align: center; font-size: 1.1rem; color: #666;">
            <span><sup>1</sup>New York University</span> &nbsp;
            <span><sup>2</sup>NVIDIA</span> &nbsp;
            <span><sup>3</sup>TU Darmstadt</span> &nbsp;
            <span><sup>4</sup>UC Berkeley</span> &nbsp;
            <span><sup>5</sup>Stanford University</span>
        </div>
    </div>
</div>

<a href="https://arxiv.org/abs/2511.20351" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2511.20351-b31b1b.svg?logo=arxiv" height="25" />
</a>
<a href="https://humanoid-vstar.github.io/" target="_blank">
    <img alt="Website" src="https://img.shields.io/badge/🌎_Website-humanoid--vstar.github.io-blue" height="25" />
</a>
<a href="https://huggingface.co/collections/humanoid-vstar/hvs-models" target="_blank">
    <img alt="HF Model: HVS-3B" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Model-HVS--3B-ffc107?color=ffc107&logoColor=white" height="25" />
</a>
<a href="https://huggingface.co/collections/humanoid-vstar/hvs-train-datasets" target="_blank">
    <img alt="HF Dataset: hvs-train-datasets" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Data-hvs--train--datasets-ffc107?color=ffc107&logoColor=white" height="25" />
</a>
<a href="https://huggingface.co/datasets/humanoid-vstar/hstar_bench" target="_blank">
    <img alt="HF Dataset: hstar_benchmark" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Benchmark-hstar__bench-ffc107?color=ffc107&logoColor=white" height="25" />
</a>

![teaser](assets/teaserv4.png)

## News
* [2025/11/26] Our [paper](https://arxiv.org/abs/2511.20351) is available on arXiv.
* [2025/11/26] We release our finetuend HVS-3B model on HuggingFace.
* [2025/11/26] We release our training datasets on HuggingFace.
* [2025/11/26] We release our benchmarking dataset on HuggingFace.

## Getting Started

### Installation

Set up the [VAGEN](https://github.com/mll-lab-nu/VAGEN) environment for training.

```bash
conda create -n vagen python=3.10
conda activate vagen
git clone --recursive https://github.com/humanoid-vstar/hstar.git
cd hstar
cd verl && pip install -e .
cd ..
bash scripts/install.sh
```
For benchmarking, we need a different envrionment for later transformers and vllm version.

```bash
conda create -n hstar python=3.10
conda activate hstar
cd vagen/inference && pip install -r requirements.txt # This env is build for CUDA 12 and torch 2.7.1
# You need to adjust the environment to adapt your machine.
cd ../..
cd verl && pip install -e . --no-deps
cd .. && pip install -e .
```
In addition, if you want to train the model from scratch, you need to install [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory.git) for SFT training.
## Training
### SFT
Use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory.git) and our SFT dataset [hos_sft and hps_sft](https://huggingface.co/collections/humanoid-vstar/hvs-train-datasets) to train or directly download our fine-tuned model [HVS-3B-sft-only](https://huggingface.co/humanoid-vstar/HVS-3B-sft-only).

### RL
* Download our RL dataset [hvs_rl](https://huggingface.co/datasets/humanoid-vstar/hvs_rl) (use mixed_rl.zip if you want to trained on the mixed dataset)
* Change your downloaded dataset path in the [training config](scripts/examples/masked_grpo/hstar/free_think/env_config.yaml).
  ```yaml
  env1:
    env_name: hstar  
    env_config:
        render_mode: vision
        prompt_format: free_think
        data_path: /path/to/your/dataset
        use_state_reward: false
        traj_success_reward: 0.5
        traj_fail_penalty: 0
        format_reward: 0.5
        resolution: 720

    train_size: 3200  
    test_size: 32
  ```
* Change your model path in [`scripts/examples/masked_grpo/hstar/free_think/run_tmux.sh`](scripts/examples/masked_grpo/hstar/free_think/run_tmux.sh) or [the tmux-free script](scripts/examples/masked_grpo/hstar/free_think/run.sh) and modify other hyperparameters.
  ```bash
  # ...
  actor_rollout_ref.model.path=/path/to/your/model \\
  # ...
  critic.model.path=/path/to/your/model \\
  # ...
  ```
* Then run the experiment by:
  ```bash
  # With tmux
  bash scripts/examples/masked_grpo/hstar/free_think/run_tmux.sh
  # Without tmux
  bash scripts/examples/masked_grpo/hstar/free_think/run.sh
  ```

## Benchmarking
* Download our [hstar_bench](https://huggingface.co/datasets/humanoid-vstar/hstar_bench) dataset.
* Change your downloaded dataset path (2 task splits) in the [`scripts/examples/masked_grpo/hstar/free_think/hos_val_config.yaml`](scripts/examples/masked_grpo/hstar/free_think/hos_val_config.yaml) and [`scripts/examples/masked_grpo/hstar/free_think/hps_val_config.yaml`](scripts/examples/masked_grpo/hstar/free_think/hps_val_config.yaml).
  ```yaml
  env1:
    env_name: hstar  
    env_config:
        render_mode: vision
        prompt_format: free_think
        use_state_reward: false
        data_path: /path/to/your/dataset/split
        resolution: 1080
  ```
* Create test dataset seeds.
  ```bash
  # Create one full dataset
  python vagen/env/create_dataset.py
    --yaml_path "scripts/examples/masked_grpo/hstar/free_think/hos_val_config.yaml" \
    --train_path "data/hos_bench/train.parquet" \
    --test_path "data/hos_bench/test.parquet"
  python vagen/env/create_dataset.py \
    --yaml_path "scripts/examples/masked_grpo/hstar/free_think/hps_val_config.yaml" \
    --train_path "data/hps_bench/train.parquet" \
    --test_path "data/hps_bench/test.parquet"
  # Or dataset clips for better efficiency
  python vagen/env/create_dataset_clip.py \
    --yaml_path "scripts/examples/masked_grpo/hstar/free_think/hos_val_config.yaml" \
    --train_path "data/hos_bench_clip/train.parquet" \
    --test_path "data/hos_bench_clip/test.parquet" \
    --num_clip 10 
  python vagen/env/create_dataset_clip.py \
    --yaml_path "scripts/examples/masked_grpo/hstar/free_think/hps_val_config.yaml" \
    --train_path "data/hps_bench_clip/train.parquet" \
    --test_path "data/hps_bench_clip/test.parquet" \
    --num_clip 10
  ```
* Modify inference settings in [`vagen/inference/inf_cfg.yaml`](vagen/inference/inf_cfg.yaml) and model settings in [`vagen/inference/model_cfg.yaml`](vagen/inference/model_cfg.yaml)
* Deploy your model using vllm OpenAI API Server on `localhost:8000`, see example [`vagen/inference/deploy.sh`](vagen/inference/deploy.sh)
* Run the experiment
  ```bash
  cd vagen/inference
  python -m vagen.server.server server.port=5000 & > ./inf_server.log &
  python run_inference.py \
	    --inference_config_path inf_cfg.yaml \
	    --model_config_path model_cfg.yaml \
		  --val_files_path /path/to/your/generated/seeds/path \
		  --wandb_path_name hstar_bench &
        [--output_dir /path/to/output/dir] # default ./temp_result
        [--save_all_results False] # save all the ouputs when set to True 
  ```
* View result
  ```bash
  python show_result.py [--result_dir /path/to/output/dir] # default ./temp_result
  ```

## References and Acknowledgement

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory.git): Easy and Efficient LLM Fine-Tuning

- [VAGEN](https://github.com/mll-lab-nu/VAGEN.git): VAGEN: Reinforcing World Model Reasoning for Multi-Turn VLM Agents

- [verl](https://github.com/volcengine/verl.git): Volcano Engine Reinforcement Learning for LLM

## Citation
```bibtex
@misc{yu2025thinking360deghumanoidvisual,
      title={Thinking in 360°: Humanoid Visual Search in the Wild}, 
      author={Heyang Yu and Yinan Han and Xiangyu Zhang and Baiqiao Yin and Bowen Chang and Xiangyu Han and Xinhao Liu and Jing Zhang and Marco Pavone and Chen Feng and Saining Xie and Yiming Li},
      year={2025},
      eprint={2511.20351},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.20351}, 
}
```
