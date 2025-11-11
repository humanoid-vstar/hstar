from typing import Dict, List, Tuple, Optional, Any
from vagen.env.base.base_service import BaseService
from vagen.env.hstar.env import HstarEnv
from vagen.env.hstar.env_config import HstarEnvConfig
from vagen.server.serial import serialize_observation
from vagen.env.hstar.service_config import HstarServiceConfig
from concurrent.futures import ThreadPoolExecutor, as_completed

class HstarService(BaseService):
    def __init__(self, config: HstarServiceConfig):
        self.max_workers = config.max_workers
        self.environments = {}
        self.env_configs = {}
        self.config = config
        print(f"[DEBUG] {self.config}")

    def create_environments_batch(self, ids2configs: Dict[str, Any]) -> None:
        def create_single_env(env_id, config):
            env_config_dict = config.get('env_config', {})
            env_config = HstarEnvConfig(**env_config_dict)
            env = HstarEnv(env_config)
            return env_id, (env, env_config)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(create_single_env, env_id, config): env_id for env_id, config in ids2configs.items()}
            for future in as_completed(futures):
                env_id, (env, env_config) = future.result()
                self.environments[env_id] = env
                self.env_configs[env_id] = env_config

    def reset_batch(self, ids2seeds: Dict[str, Any]) -> Dict[str, Tuple[Any]]:
        results = {}
        def reset_single_env(env_id, seed):
            env = self.environments[env_id]
            observation, info = env.reset(seed=seed)
            # Serialize observation for network transfer
            serialized_observation = serialize_observation(observation)
            return env_id, (serialized_observation, info)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(reset_single_env, env_id, seed): env_id for env_id, seed in ids2seeds.items()}
            for future in as_completed(futures):
                env_id, result = future.result()
                results[env_id] = result

        return results
    
    def step_batch(self, ids2actions: Dict[Any, Any]) -> Dict[Any, Tuple[Dict, float, bool, Dict]]:
        """Execute LLM actions across multiple environments"""
        results = {}
        def step_single_env(env_id, action):
            env = self.environments[env_id]
            try:
                observation, reward, done, info = env.step(action)
            except Exception as e:
                print(f"Error stepping environment {env_id}: {e}")
                try:
                    observation,info = env.reset()
                    reward =0
                    done = True
                except Exception as e:
                    print(f"Error resetting environment {env_id} after failure: {e}")
                    env.close()
                    config=self.env_configs[env_id]
                    env = HstarEnv(config)
                    self.environments[env_id] = env
                    observation, info = env.reset()
                    done = True
                    reward = 0
            # Serialize observation for network transfer
            serialized_observation = serialize_observation(observation)
            return env_id, (serialized_observation, reward, done, info)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(step_single_env, env_id, action): env_id for env_id, action in ids2actions.items()}
            for future in as_completed(futures):
                env_id, result = future.result()
                results[env_id] = result
        return results
    
    def compute_reward_batch(self, env_ids: List[str]) -> Dict[Any, float]:
        """Get final rewards for multiple environments"""
        results = {}
        def compute_single_reward(env_id):
            env = self.environments[env_id]
            return env_id, env.compute_reward()
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(compute_single_reward, env_id): env_id for env_id in env_ids}
            for future in as_completed(futures):
                env_id, reward = future.result()
                results[env_id] = reward
        return results
    def get_system_prompts_batch(self, env_ids: List[str]) -> Dict[Any, str]:
        """Get system prompts for multiple environments"""
        results = {}
        def get_single_system_prompt(env_id):
            env = self.environments[env_id]
            return env_id, env.system_prompt()
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(get_single_system_prompt, env_id): env_id for env_id in env_ids}
            for future in as_completed(futures):
                env_id, system_prompt = future.result()
                results[env_id] = system_prompt
        return results
    
    def close_batch(self, env_ids: Optional[List[str]] = None) -> None:
        """Clean up multiple environments"""
        if env_ids is None:
            env_ids = list(self.environments.keys())
        def close_single_env(env_id):
            env = self.environments.pop(env_id)
            env.close()
            return None
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(close_single_env, env_id) for env_id in env_ids]
            
            # Wait for all tasks to complete
            for future in as_completed(futures):
                error = future.result()
                if error:
                    print(f"Error closing environment: {error}")
        for env_id in env_ids:
            self.env_configs.pop(env_id, None)
            self.environments.pop(env_id, None)
            
            