# coding: utf-8
import os
import json
import cv2
import numpy as np
import imageio
import json
import torch
import cv2
import time
import argparse

from uninavid.mm_utils import get_model_name_from_path
from uninavid.model.builder import load_pretrained_model
from uninavid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from uninavid.conversation import conv_templates, SeparatorStyle
from uninavid.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria




seed = 30
torch.manual_seed(seed)
np.random.seed(seed)






class UniNaVid_Agent():
    def __init__(self, model_path):
        
        print("Initialize UniNaVid")
        
        self.conv_mode = "vicuna_v1"

        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, None, get_model_name_from_path(model_path))

        assert self.image_processor is not None

        print("Initialization Complete")
        
        self.promt_template = "Imagine you are a robot programmed for navigation tasks. You have been given a video of historical observations and an image of the current observation <image>. Your assigned task is: '{}'. Analyze this series of images to determine your next four actions. The predicted action should be one of the following: forward, left, right, or stop."
        self.rgb_list = []
        self.count_id = 0
        self.reset()

    def process_images(self, rgb_list):

        
        batch_image = np.asarray(rgb_list)
        self.model.get_model().new_frames = len(rgb_list)
        video = self.image_processor.preprocess(batch_image, return_tensors='pt')['pixel_values'].half().cuda()

        return [video]


    def predict_inference(self, prompt):
        
        question = prompt.replace(DEFAULT_IMAGE_TOKEN, '').replace('\n', '')
        qs = prompt

        VIDEO_START_SPECIAL_TOKEN = "<video_special>"
        VIDEO_END_SPECIAL_TOKEN = "</video_special>"
        IMAGE_START_TOKEN = "<image_special>"
        IMAGE_END_TOKEN = "</image_special>"
        NAVIGATION_SPECIAL_TOKEN = "[Navigation]"
        IAMGE_SEPARATOR = "<image_sep>"
        image_start_special_token = self.tokenizer(IMAGE_START_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        image_end_special_token = self.tokenizer(IMAGE_END_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        video_start_special_token = self.tokenizer(VIDEO_START_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        video_end_special_token = self.tokenizer(VIDEO_END_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        navigation_special_token = self.tokenizer(NAVIGATION_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        image_seperator = self.tokenizer(IAMGE_SEPARATOR, return_tensors="pt").input_ids[0][1:].cuda()

        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs.replace('<image>', '')
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs.replace('<image>', '')

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        token_prompt = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda()
        indices_to_replace = torch.where(token_prompt == -200)[0]
        new_list = []
        while indices_to_replace.numel() > 0:
            idx = indices_to_replace[0]
            new_list.append(token_prompt[:idx])
            new_list.append(video_start_special_token)
            new_list.append(image_seperator)
            new_list.append(token_prompt[idx:idx + 1])
            new_list.append(video_end_special_token)
            new_list.append(image_start_special_token)
            new_list.append(image_end_special_token)
            new_list.append(navigation_special_token)
            token_prompt = token_prompt[idx + 1:]
            indices_to_replace = torch.where(token_prompt == -200)[0]
        if token_prompt.numel() > 0:
            new_list.append(token_prompt)
        input_ids = torch.cat(new_list, dim=0).unsqueeze(0)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        imgs = self.process_images(self.rgb_list)
        self.rgb_list = []

        cur_prompt = question
        with torch.inference_mode():
            self.model.update_prompt([[cur_prompt]])
            output_ids = self.model.generate(
                input_ids,
                images=imgs,
                do_sample=True,
                temperature=0.5,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        return outputs




    def reset(self, task_type='vln'):

        self.transformation_list = []
        self.rgb_list = []
        self.last_action = None
        self.count_id += 1
        self.count_stop = 0
        self.pending_action_list = []
        self.task_type = task_type

        self.first_forward = False
        self.executed_steps = 0
        self.model.config.run_type = "eval"
        self.model.get_model().initialize_online_inference_nav_feat_cache()
        self.model.get_model().new_frames = 0


    def act(self, data):
    
        rgb = data["observations"]
        self.rgb_list.append(rgb)


        navigation_qs = self.promt_template.format(data["instruction"])
        
        navigation = self.predict_inference(navigation_qs)
                
        action_list = navigation.split(" ")

        traj = [[0.0, 0.0, 0.0]]
        for action in action_list: 
            if action == "stop":
                waypoint = [x + y for x, y in zip(traj[-1], [0.0, 0.0, 0.0])]
                traj = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
                break
            elif action == "forward":
                waypoint = [x + y for x, y in zip(traj[-1], [0.5, 0.0, 0.0])]
                traj.append(waypoint)
            elif action == "left":
                waypoint = [x + y for x, y in zip(traj[-1], [0.0, 0.0, -np.deg2rad(30)])]
                traj.append(waypoint)
            elif action == "right":
                waypoint = [x + y for x, y in zip(traj[-1], [0.0, 0.0, np.deg2rad(30)])]
                traj.append(waypoint)

                                    
        if len(action_list)==0:
            raise ValueError("No action found in the output")
            
        self.executed_steps += 1
            
        self.latest_action = {"step": self.executed_steps, "path":[traj], "actions":action_list}
            
        return self.latest_action.copy()

def get_sorted_images(recording_dir):
    image_dir = os.path.join(recording_dir, 'images')
    
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    
    image_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    
    images = []
    for step, image_file in enumerate(image_files):
        image_path = os.path.join(image_dir, image_file)
        np_image = cv2.imread(image_path)
        images.append(np_image)
    
    return images

def get_traj_data(recording_dir):
    json_path = os.path.join(recording_dir, "instruction.json")

    with open(json_path, 'r', encoding='utf-8') as f:
        instruction = json.load(f)["instruction"]

    return instruction

def draw_traj_arrows_fpv(
    img,
    actions,
    arrow_len=10,                
    arrow_gap=2,                 
    arrow_color=(0, 255, 0),    
    arrow_thickness=2,
    tipLength=0.35,
    stop_color=(0, 0, 255),      
    stop_radius=5
):
 
    out = img.copy()
    h, w = out.shape[:2]

    base_x, base_y = w // 2, int(h * 0.95)

    for i, action in enumerate(actions):
        if action == "stop":
            waypoint = [0.0, 0.0, 0.0]
        elif action == "forward":
            waypoint = [0.5, 0.0, 0.0]
        elif action == "left":
            waypoint = [0.0, 0.0, -np.deg2rad(30)]
        elif action == "right":
            waypoint = [0.0, 0.0, np.deg2rad(30)]
        else:
            continue  

        x, y, yaw = waypoint

        start = (
            int(base_x),
            int(base_y - i * (arrow_len + arrow_gap))
        )

        if action == "stop":
            cv2.circle(out, start, stop_radius, stop_color, 2)
        else:
            end = (
                int(start[0] + arrow_len * np.sin(yaw)),
                int(start[1] - arrow_len * np.cos(yaw))
            )
            cv2.arrowedLine(out, start, end, arrow_color, arrow_thickness, tipLength=tipLength)
    
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return out





if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('test_case', help='test case path (images dir)')
    parser.add_argument('output_dir', help='output dir to save results')
    

    
    args = parser.parse_args()
    
    
    
    
    agent = UniNaVid_Agent("model_zoo/uninavid-7b-full-224-video-fps-1-grid-2")
    agent.reset()
     
    images = get_sorted_images(args.test_case)
    instruction = get_traj_data(args.test_case)
    print(f"Total {len(images)} images")
    h,w,n = images[0].shape
        
    result_vis_list = []
    step_count = 0
    for i, img in enumerate(images):
        image=img

        import time
        t_s = time.time()
        result = agent.act({'instruction': instruction, 'observations': image})
        step_count += 1
        
        print("step", step_count, "inference time", time.time()-t_s)
        
        traj = result['path'][0]
        actions = result['actions']

        vis = draw_traj_arrows_fpv(img, actions, arrow_len=20)
        result_vis_list.append(vis)

    
    imageio.mimsave(os.path.join(args.output_dir,"result.gif"), result_vis_list)