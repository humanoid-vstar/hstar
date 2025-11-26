# *Thinking in 360°*: Humanoid Visual Search in the Wild
![teaser](assets/teaserv4.png)
[Paper](https://arxiv.org/abs/2511.20351) | [Page](https://humanoid-vstar.github.io/) | [Model](https://huggingface.co/collections/humanoid-vstar/hvs-models) | [Dataset](https://huggingface.co/collections/humanoid-vstar/hvs-train-datasets) | [Benchmark](https://huggingface.co/datasets/humanoid-vstar/hstar_bench)
## Getting Started

### Installation

Set up the [VAGEN](https://github.com/mll-lab-nu/VAGEN) environment for training.

```bash
conda create -n vagen python=3.10
conda activate vagen
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
* Change your downloaded dataset path (2 task splits) in the [`scripts/examples/masked_grpo/hstar/free_think/hos_val_config.yaml`](scripts/examples/masked_grpo/hstar/free_think/hos_val_config.yaml) and [`scripts/examples/masked_grpo/hstar/free_think/hos_val_config.yaml`](scripts/examples/masked_grpo/hstar/free_think/hos_val_config.yaml).
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
      title={Thinking in 360{\deg}: Humanoid Visual Search in the Wild}, 
      author={Heyang Yu and Yinan Han and Xiangyu Zhang and Baiqiao Yin and Bowen Chang and Xiangyu Han and Xinhao Liu and Jing Zhang and Marco Pavone and Chen Feng and Saining Xie and Yiming Li},
      year={2025},
      eprint={2511.20351},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.20351}, 
}
```