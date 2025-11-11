python -m vagen.server.server server.port=5000 & > ./inf_server.log &
python run_inference.py \
	    --inference_config_path inf_cfg.yaml \
	        --model_config_path model_cfg.yaml \
		    --val_files_path /path/to/your/generated/seeds/path \
		    --wandb_path_name hstar_bench &
