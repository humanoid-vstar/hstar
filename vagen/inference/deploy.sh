CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m vllm.entrypoints.openai.api_server \
	    --model "google/gemma-3-4b-it" \
		--max_num_batched_tokens 40960 \
	    --tensor-parallel-size 8 \
		--host 0.0.0.0 \
		--port 8000 \
		--trust_remote_code &
