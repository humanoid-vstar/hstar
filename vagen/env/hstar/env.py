from vagen.env.base.base_env import BaseEnv
from vagen.env.hstar.env_config import HstarEnvConfig
from PIL import Image
from vagen.env.utils.parse_utils import PARSE_FUNC_MAP
from vagen.env.utils.state_reward_text_utils import env_state_reward_wrapper
import cv2
import numpy as np
import json
import math
import random
from pathlib import Path
from vagen.env.hstar.prompt import format_prompt, system_prompt, action_template, init_observation_template
def normalize_angle(angle):
    return (angle + 360) % 360

def calulate_distance_yaw(self, a, b):
    a = normalize_angle(a)
    b = normalize_angle(b)
    dis = abs(a - b)
    if dis > 180:
        dis = 360 - dis
    return dis

def calulate_distance_pitch(self, a, b):
    return abs(a - b)
class HstarEnv(BaseEnv):
    ValidEvalSets = ["object_search","navigation","mixed"]
    
    ACTION_LOOKUP = {
        "rotate":1,
        "zoom":2,
        "submit":3,
    }
    def __init__(self, config:HstarEnvConfig):
        super().__init__()
        self.config = config
        self.resolution = (config.resolution//9*16, config.resolution)
        assert config.eval_set in self.ValidEvalSets
        self.data_path = config.data_path
        self.img_buf = {}
        self.dataset = self._load_dataset()
        
        self.number_of_episodes = len(self.dataset)
        self.fov = 100
        self.yaw = 0
        self.pitch = 0
        self.format_prompt_func = format_prompt[self.config.prompt_format]
        self.parse_func = PARSE_FUNC_MAP[self.config.prompt_format]
        self.episode_data = None
        self.episode_language_instruction = ""
        self.episode_log = []
        self.format_reward = self.config.format_reward
        self._current_episode_num = 0
        self._current_step = 0
        self.reward = 0
        self.total_reward = 0
        self.target_yaw = [0, 0]
        self.target_pitch = [0, 0]
        self.valid_action = []
        self.success = False
        self.level = 0
        self.is_nav = False
    def _load_dataset(self):
        dataset = []
        for id,dir in enumerate(sorted(Path(self.data_path).iterdir())):
            if not dir.is_dir():
                continue
            if not (dir/"annotation.json").exists():
                continue
            img_file = list(dir.glob("*.png"))+list(dir.glob("*.jpg"))
            if not img_file:
                continue
            img_file = img_file[0]
            self.img_buf[id] = img_file
            with open(dir/"annotation.json", "r") as f:
                annotation = json.load(f)

            for item in annotation:
                item["image_path_id"] = id
                if "initial yaw" not in item:
                    for start_yaw in range(0, 360, 90):
                        item_copy = item.copy()
                        item_copy["start_yaw"] = start_yaw
                        dataset.append(item_copy)
                else:
                    for start_yaw,level in zip(item["initial yaw"],item['level']):
                        item_copy = item.copy()
                        item_copy["start_yaw"] = start_yaw
                        item_copy["level"] = level
                        dataset.append(item_copy)
        random.seed(self.config.seed)
        random.shuffle(dataset)  # Shuffle the dataset
        return dataset
        
    def _e2p(self, img, fov, yaw, pitch, roll, w, h):
        fov,yaw,pitch,roll = map(math.radians,(fov,yaw,pitch,roll))
        f = w/(2*math.tan(fov/2))
        i,j = np.meshgrid(np.arange(w),np.arange(h))
        x=(i-w/2)/f; y=(j-h/2)/f; z=np.ones_like(x)
        norm=np.sqrt(x*x+y*y+1); x,y,z=x/norm,y/norm,z/norm
        Rx=np.array([[1,0,0],[0,math.cos(pitch),-math.sin(pitch)],[0,math.sin(pitch),math.cos(pitch)]])
        Ry=np.array([[math.cos(yaw),0,math.sin(yaw)],[0,1,0],[-math.sin(yaw),0,math.cos(yaw)]])
        Rz=np.array([[math.cos(roll),-math.sin(roll),0],[math.sin(roll),math.cos(roll),0],[0,0,1]])
        R = Rz @ Ry @ Rx
        xyz=np.stack((x,y,z),-1).reshape(-1,3).T
        x2,y2,z2 = R @ xyz
        lon,lat  = np.arctan2(x2,z2), np.arcsin(y2)
        H,W = img.shape[:2]
        mx=((lon/np.pi+1)/2*W).reshape(h,w).astype(np.float32)
        my=((lat/np.pi+0.5)*H).reshape(h,w).astype(np.float32)

        output = cv2.remap(img,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_WRAP)

        center_x, center_y = w // 2, h // 2
        length = 10  
        color = (0, 255, 0)  
        thickness = 1

       
        cv2.line(output, (center_x - length, center_y), (center_x + length, center_y), color, thickness)
      
        cv2.line(output, (center_x, center_y - length), (center_x, center_y + length), color, thickness)
        return output
    
    def _render(self, init_obs=True):
        img_placeholder = self.config.get("image_placeholder", "<image>")
        
        format_prompt_text = self.format_prompt_func(
            add_example=False
        )
        
        pano_img = cv2.imread(str(self.img_buf[self.episode_data["image_path_id"]]))
        
        view_img = self._e2p(
            pano_img, 
            self.fov, 
            normalize_angle(self.yaw), 
            self.pitch, 
            0, 
            self.resolution[0], 
            self.resolution[1]
        )
        multi_modal_data = {
            img_placeholder: [Image.fromarray(cv2.cvtColor(view_img,cv2.COLOR_BGR2RGB))]
        }
        if init_obs:
            obs_str = init_observation_template(
                observation=img_placeholder,
                instruction=self.episode_language_instruction
            )+ "\n" + format_prompt_text
        else:
            obs_str = action_template(
                observation=img_placeholder,
                instruction=self.episode_language_instruction,
                valid_action=self.valid_action,
                env_feedback=self.info["env_feedback"],
                done=self.info.get("done", False)
            ) + "\n" + format_prompt_text
        
        return {
            "obs_str": obs_str,
            "multi_modal_data": multi_modal_data
        }
                
        
    def reset(self, seed=None):
        idx = seed % self.number_of_episodes if seed is not None else 0
        data = self.dataset[idx]
        self.episode_data = data
        self.episode_language_instruction = data["task"]
        self.target_yaw = data['yaw']
        if "pitch" in data:
            self.target_pitch = data['pitch']
        else:
            self.target_pitch = [-90, 90]
            self.config.yaw_tolerance = 10
            self.is_nav = True
        if self.target_yaw[0] > self.target_yaw[1]:
            self.target_yaw[1] += 360  # Handle wrap-around case for yaw

        if (self.target_yaw[1]- self.target_yaw[0]) < 2*self.config.yaw_tolerance:
            yaw = (self.target_yaw[0] + self.target_yaw[1]) / 2
            self.target_yaw = [normalize_angle(yaw - self.config.yaw_tolerance), normalize_angle(yaw + self.config.yaw_tolerance)]
        else:
            self.target_yaw[0] = normalize_angle(self.target_yaw[0])
            self.target_yaw[1] = normalize_angle(self.target_yaw[1])

        if (self.target_pitch[1]- self.target_pitch[0]) < 2*self.config.pitch_tolerance:
            pitch = (self.target_pitch[0] + self.target_pitch[1]) / 2
            self.target_pitch = [max(-90, pitch - self.config.pitch_tolerance), min(90, pitch + self.config.pitch_tolerance)]
        self.episode_log = []
        self._current_episode_num = 0
        self._current_step = 0
        self.reward = 0
        self.format_reward = 0
        self.total_reward = 0
        self.valid_action = []
        self.success = False
        self.yaw = data["start_yaw"]
        self.level = data['level']
        self.pitch = 0
        return self._render(init_obs=True), {}
    
    
    @env_state_reward_wrapper
    def step(self, llm_raw_response: str):
        action_str = llm_raw_response
        rst = self.parse_func(
            response=action_str,
            special_token_list= self.config.get("special_token_list", None),
            max_actions= self.config.max_actions_per_step,
            action_sep="|"
        )
        
       
        metrics = {
            "turn_metrics":{
                "action_is_valid":len(rst["actions"])==1,
                "action_is_effective":False,
            },
            "traj_metrics":{
                "success":False,
                "level": self.level,
            }
        } 
        
        self.reward = 0
        self.valid_action = []
        done = False
        info = {}
        info.update(rst)
        self._current_step += 1
        if metrics["turn_metrics"]["action_is_valid"]: 
            action = rst["actions"][0] 
            try:
                if action.startswith("rotate"):
                    yaw_distance_before = calulate_distance_yaw(self, self.yaw, self.target_yaw[0]) + calulate_distance_yaw(self, self.yaw, self.target_yaw[1])
                    pitch_distance_before = calulate_distance_pitch(self, self.pitch, self.target_pitch[0]) + calulate_distance_pitch(self, self.pitch, self.target_pitch[1])
                    # if self._current_step >= self.config.step_tolerance:
                    #     self.reward += self.config.max_action_penalty
                    angles = action.replace("rotate", "").strip("()").split(",")
                    delta_yaw = int(angles[0].strip())
                    delta_pitch = int(angles[1].strip())
                    self.yaw = normalize_angle(self.yaw + delta_yaw)
                    self.pitch += delta_pitch
                    self.pitch = max(-90, min(90, self.pitch))  # Clamp pitch to [-90, 90]
                    self.valid_action.append(action)
                    yaw_distance_after = calulate_distance_yaw(self, self.yaw, self.target_yaw[0]) + calulate_distance_yaw(self, self.yaw, self.target_yaw[1])
                    pitch_distance_after = calulate_distance_pitch(self, self.pitch, self.target_pitch[0]) + calulate_distance_pitch(self, self.pitch, self.target_pitch[1])
                    effective_reward = (yaw_distance_before-yaw_distance_after)/180 + (pitch_distance_before-pitch_distance_after)/180
                    metrics["turn_metrics"]["action_is_effective"] = effective_reward > 0
                    if metrics["turn_metrics"]["action_is_effective"]:
                        self.reward += effective_reward * self.config.effective_reward_weight
                    else: 
                        self.reward += effective_reward * self.config.ineffective_penalty_weight
                        
                elif action.startswith("submit"):
                    
                    angles = action.replace("submit", "").strip("()").split(",")
                    yaw = normalize_angle(int(angles[0]))
                    pitch = int(angles[1])
                    
                    if self.target_yaw[0] > self.target_yaw[1]:
                        # Handle wrap-around case for yaw
                        if self.target_yaw[0] <= yaw <= 360 or 0 <= yaw <= self.target_yaw[1]:
                            self.success = True
                            self.reward += self.config.traj_success_reward
                            metrics["traj_metrics"]["success"] = True
                        else:
                            self.success = False
                            self.reward += self.config.traj_fail_penalty
                    elif self.target_yaw[0] <= yaw <= self.target_yaw[1] and self.target_pitch[0] <= pitch <= self.target_pitch[1]:
                        self.success = True
                        self.reward += self.config.traj_success_reward
                        
                        metrics["traj_metrics"]["success"] = True
                    else:
                        self.success = False
                        self.reward += self.config.traj_fail_penalty
                    done = True
                    self.reward += self.format_reward
                else:
                    metrics["turn_metrics"]["action_is_valid"] = False
            except Exception as e:
                metrics["turn_metrics"]["action_is_valid"] = False
        if metrics["turn_metrics"]["action_is_valid"] and rst.get("format_correct", True):
            info["is_format_rewarded"] = True  
        else:
            self.format_reward = 0
            info["is_format_rewarded"] = False           
        info["metrics"] = metrics
        info["instruction"] = self.episode_language_instruction
        info["env_step"] = self._current_step
        info['task_success'] = self.success
        info['done'] = done
        info['level'] = self.level
        info["env_feedback"] = f"Last action executed successfully, your current direction (yaw,pitch) is ({self.yaw},{self.pitch})" if metrics["turn_metrics"]["action_is_valid"] else "Last action is not executed successfully."
        self.info = info
        self.total_reward += self.reward
        
        return self._render(init_obs=False), self.reward, done, info
    def system_prompt(self) -> str:
        format_prompt_text = self.format_prompt_func(
            max_actions_per_step=self.config.max_actions_per_step,
            add_example=True
        )
        
        return system_prompt(format = self.config.prompt_format, is_nav = self.is_nav) + "\n" + format_prompt_text
    
    def close(self):
        # Do nothing
        pass
           
    
if __name__ == "__main__":
    import os
    config = HstarEnvConfig()
    env = HstarEnv(config=config)
    print(env.system_prompt())
    obs, info = env.reset(seed=0)
    print(obs["obs_str"])
    i = 0
    os.makedirs("./test_ev", exist_ok=True)
    img = obs["multi_modal_data"][config.image_placeholder][0]
    img.save(f"./test_ev/step_{i}.png")
    done = False
    while not done:
        action = input("Enter action: rotate(int,int), zoom(in|out), submit(int,int): ")
        action = f"<think>Let me think about the next action.</think>\n<answer>{action}</answer>"
        obs, reward, done, info = env.step(action)
        print(f"Reward: {reward}, Done: {done}")
        print(info)
        print(obs["obs_str"])
        img = obs["multi_modal_data"][config.image_placeholder][0]
        img.save(f"./test_ev/step_{i}.png")
        print(f"Success: {info['metrics']['traj_metrics']['success']}")
        i += 1
        
        if done:
            break
    print(f"Total Reward: {env.total_reward}")
    env.close()
        