# This FORMAT_CONFIGS is for the robot navigation task,
# structured like the FORMAT_CONFIGS in your first (FrozenLake) example.
FORMAT_CONFIGS = {
    "free_think": {
        "description": "You should first give your thought process, and then your answer.",
        "format": "<think>...</think><answer>...</answer>",
        "example": """<think>I need to find the coffee machine. I can see a table on on my left, a couch in front of me, and a door to the right. The coffee machine is likely on the table, which is to my left.</think><answer>rotate(-45,0)</answer>"""
    },
    "no_think": {
        "description": "You should provide only your answer.",
        "format": "<answer>...</answer>",
        "example": """<answer>rotate(30,5)</answer>"""
    },
    "grounding": {
        "description": "You should first give your thought process with your observation and reasoning, and finally your answer.\nThe observation should be described in detail about what you see in the environment.",
        "format": "<think><observation>...</observation><reasoning>...</reasoning></think><answer>...</answer>",
        "example": """<think><observation>I am in a living room. There is a couch to my left, a TV in front of me, and a doorway to the kitchen on my right. The target object, a vase, appears to be on a shelf near the kitchen doorway.</observation><reasoning>I need to move toward the kitchen doorway to reach the vase. So I'll turn right and head toward the vase.</reasoning></think><answer>rotate(60,0)</answer>"""
    }
}

def system_prompt(**kwargs):
    example = "" # Default empty example
    # Internally uses kwargs.get("format"), as in your original code
    selected_format = kwargs.get("format", "default")
    is_nav = kwargs.get("is_nav", False)
    if selected_format in ["free_think", "default"]:
        example="""Example:
Round 1:
image_1
<think>I need to find the coffee machine. I can see a table on on my left, a couch in front of me, and a door to the right. The coffee machine is likely on the table, which is to my left.</think><answer>rotate(-45,0)</answer>
Round 2:
Env_feedback: Last action is executed successfully, your current direction (yaw,pitch) is (315,0).
image_2
<think>From the secene, I see that by turning left 45 degrees, a kitchen table is in front of me. The coffee machine is on the left of the table and slightly lower than the camera center. I need to turn leftward and downward a little bit.</think>
<answer>rotate(-30,-5)</answer>
Round 3:
Env_feedback: Last action is executed successfully, your current direction (yaw,pitch) is (285,-5).
image_3
<think>The coffee machine is right now at the center of my camera, I think I can submit the task.</think>
<answer>submit(285,-5)</answer>
Round 4:
Env_feedback: Success"""
    elif selected_format == "grounding":
        example="""Example:
Round 1:
image_1
<think><observation>I can see a table on on my left, a couch in front of me, and a door to the right.</observation><reasoning>I need to find the coffee machine. The coffee machine is likely on the table, which is to my left.</reasoning></think><answer>rotate(-45,0)</answer>
Round 2:
Env_feedback: Last action is executed successfully, your current direction (yaw,pitch) is (315,0).
image_2
<think><observation>From the secene, I see that by turning left 45 degrees, a kitchen table is in front of me. The coffee machine is on the left of the table and slightly lower than the camera center.</observation><reasoning>In order to center the target object, I need to turn left and look down a little bit.</reasoning></think>
<answer>rotate(-30,-5)</answer>
Round 3:
Env_feedback: Last action is executed successfully, your current direction (yaw,pitch) is (285,-5).
image_3
<think><observation>The coffee machine is right now at the center of my camera.</observation><reasoning>I think I can submit the task.</reasoning></think>
<answer>submit(285,-5)</answer>
Round 4:
Env_feedback: Success"""
    elif selected_format == "no_think":
        example="""Example:
Round 1:
image_1
<answer>rotate(-45,0)</answer>
Round 2:
Env_feedback: Last action is executed successfully, your current direction (yaw,pitch) is (315,0).
image_2
<answer>rotate(-30,-5)</answer>
Round 3:
Env_feedback: Last action is executed successfully, your current direction (yaw,pitch) is (285,-5).
image_3
<answer>submit(285,-5)</answer>
Round 4:
Env_feedback: Success"""
    base_prompt_text = """You are a robot and perform object searching tasks according to instructions. Your goal is to rotate the camera to center the target object in the camera view, and then submit the task. The camera center is presented as a green cross in the picture.
Actions you can take: rotate(yaw:int,pitch:int), submit(yaw:int,pitch:int). 
rotate(yaw:int,pitch:int): rotate the camera in the yaw and pitch direction relative to the current direction. Yaw is the rotation angle in the x-y plane, pitch is the rotation angle in the y-z plane. Yaw angle > 0 means rotate to the right, yaw angle < 0 means rotate to the left. Pitch angle > 0 means look up, pitch angle < 0 means look down.
submit(yaw:int,pitch:int): submit the task with the current camera view with the target object at the center, yaw and pitch are the angles of the current camera view, which is reported by the environment.

You can only take one action at a time. The instruction will be provided with each observation. Look at the image carefully to complete the instruction.
""" if not is_nav else """You are a robot and perform navigation tasks according to instructions. Your goal is to turn your camera center to the target direction you need to move towards to reach the target location. The camera center is presented as a green cross in the picture. Don't move in the inavailable direction, such as obstacles or gaps.
Actions you can take: rotate(yaw:int,pitch:int), submit(yaw:int,pitch:int). 
rotate(yaw:int,pitch:int): rotate the camera in the yaw and pitch direction relative to the current direction. Yaw is the rotation angle in the x-y plane, pitch is the rotation angle in the y-z plane. Yaw angle > 0 means rotate to the right, yaw angle < 0 means rotate to the left. Pitch angle > 0 means look up, pitch angle < 0 means look down.
submit(yaw:int,pitch:int): submit the task with the current direction to move torwards, yaw and pitch are the angles of the current camera view, which is reported by the environment.
You can only take one action at a time. The instruction will be provided with each observation. Look at the image carefully to complete the instruction.
"""
    return base_prompt_text + '\n' + example

def init_observation_template(**kwargs):
    observation = kwargs.get("observation", "No observation provided.")
    instruction = kwargs.get("instruction", "No instruction provided.")
    return f"""[initial observation]:
{observation}
Human Instruction: {instruction}
Decide your next action."""
def action_template(**kwargs):
    observation = kwargs.get("observation", "No observation provided.")
    instruction = kwargs.get("instruction", "No instruction provided.")
    valid_action = kwargs.get("valid_action", "No valid action provided.")
    env_feedback = kwargs.get("env_feedback", "No environment feedback provided.")
    done = kwargs.get("done", False)
    if done == True:
        return f"""After your answer, the extracted valid action is {valid_action}.
The environment feedback is: {env_feedback}. Task is done."""
    return f"""After your answer, the extracted valid action is {valid_action}.
The environment feedback is: {env_feedback}
done: {done}
After that, the observation is:
{observation}
Human Instruction: {instruction}
Decide your next action."""


# format_prompt_generator function, similar to your first (FrozenLake) example
def format_prompt_generator(format_type):
    """
    Generates a prompt function for the specified format type.
    This returned function creates the per-turn instruction for the LLM.
    """
    def prompt_function(**kwargs):
        """
        Generate a prompt for the specified format for the robot navigation task.
        
        Args:
            max_actions_per_step (int): Max actions. Defaults to 1 (common for robot).
            add_example (bool): Whether to add an example. Defaults to True.
            
        Returns:
            str: The formatted prompt.
        """
        # Defaults suitable for the robot navigation task
        max_actions_per_step = kwargs.get("max_actions_per_step", 1)
        add_example = kwargs.get("add_example", True) # Default to True as per robot examples
        
        if format_type not in FORMAT_CONFIGS:
            raise ValueError(f"Unknown format_type: {format_type}")
        config = FORMAT_CONFIGS[format_type]
        
        base_prompt = f"""You can take {max_actions_per_step} action(s) at a time.
{config["description"]}"""
        
        if "additional_info" in config: # In case it's added to FORMAT_CONFIGS later
            base_prompt += f"\n{config['additional_info']}"
        
        base_prompt += f"""
Your response should be in the format of:
{config["format"]}"""
        
        if add_example:
            # The 'e.g.' is already part of the example string in this FORMAT_CONFIGS
            example_text = config["example"]
            return base_prompt + '\n' + f"e.g. {example_text}"
        
        return base_prompt
    
    return prompt_function

# format_prompt dictionary, as in your first (FrozenLake) example
format_prompt = {
    ft: format_prompt_generator(ft) 
    for ft in FORMAT_CONFIGS  # Iterate directly over keys in FORMAT_CONFIGS
}


if __name__ == "__main__":
    # Example usage
    max_actions_per_step = 1
    
    for key, func in format_prompt.items():
        print(f"{key} format prompt:")
        print(system_prompt(format=key))
        print(func(max_actions_per_step=max_actions_per_step))
        print("\n" + "="*50 + "\n")