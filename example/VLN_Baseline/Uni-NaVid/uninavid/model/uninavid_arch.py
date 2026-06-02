#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# ------------------------------------------------------------------------
# Modified from LLaVA (https://github.com/haotian-liu/LLaVA)
# Copyright 2023 Yanwei Li
# ------------------------------------------------------------------------

from abc import ABC, abstractmethod
import os
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector


from uninavid.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, VIDEO_START_SPECIAL_TOKEN, VIDEO_END_SPECIAL_TOKEN, IMAGE_START_TOKEN, IMAGE_END_TOKEN, NAVIGATION_SPECIAL_TOKEN, NAVIGATION_IDENTIFIER, IAMGE_SEPARATOR




class UniNaVIDMetaModel:

    def __init__(self, config):
        super(UniNaVIDMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower
    
    
    def initialize_online_inference_nav_feat_cache(self):        
        self.feat_cache = None
        self.long_feat_cache = None
        self.weight = 1
        self.new_frames = 0

    

    def initialize_vision_modules(self, model_args, fsdp=None, max_token=2048):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower
        self.config.image_processor = getattr(model_args, 'image_processor', None)
        self.config.compress_type = getattr(model_args, "compress_type", None)
        self.config.run_type = model_args.run_type
        
        vision_tower = build_vision_tower(model_args)

        if fsdp is not None and len(fsdp) > 0:
            self.vision_tower = [vision_tower]
        else:
            self.vision_tower = vision_tower

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.max_token = max_token
        
        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))




class UniNaVIDMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()


    def online_process_tensor(self, nav_size, length_threshold=64, similarity_threshold=0.985):
        k, m, c = self.get_model().feat_cache.shape
                
        assert m % nav_size == 0, f"m ({m}) must be divisible by nav_size ({nav_size})"
        result_list = []
        
        
        if k <= length_threshold:
            result_list = [nav_size] * k
            return self.get_model().feat_cache.reshape(-1, c), result_list

        
        cos = torch.nn.CosineSimilarity(dim=0)

        for frame_index in range(self.get_model().new_frames-1, -1, -1):
            
            if  k - length_threshold - frame_index - 1 < 0:
                continue 
            
            
            oldest_short_mem_token = self.get_model().feat_cache[k-length_threshold-frame_index-1,:,:].mean(dim=0)
            long_term_cache = self.get_model().long_feat_cache
        
            if long_term_cache is not None:
                assert long_term_cache[-1, :].shape == oldest_short_mem_token.shape, \
                f"Shape mismatch: long_term_cache[-1, :] {long_term_cache[-1, :].shape} vs oldest_short_mem_token {oldest_short_mem_token.shape}"
                
                similarity = cos(long_term_cache[-1,:], oldest_short_mem_token)
                
                if similarity > similarity_threshold:                    
                    new_mean = (long_term_cache[-1,:] * self.get_model().weight + oldest_short_mem_token) / (self.get_model().weight + 1)
                    self.get_model().weight = self.get_model().weight + 1
                    long_term_cache[-1] = new_mean
                    self.get_model().long_feat_cache = long_term_cache
                else:
                    self.get_model().long_feat_cache = torch.cat([long_term_cache, oldest_short_mem_token[None,:]], dim=0)
                    self.get_model().weight = 1  
            else:
                self.get_model().long_feat_cache = oldest_short_mem_token[None, :]
            
        
        
        result_list = [1] * self.get_model().long_feat_cache.shape[0] + [nav_size] * length_threshold
        
        result_tensor = torch.cat([self.get_model().long_feat_cache, self.get_model().feat_cache[k - length_threshold:].reshape(-1,c)], dim=0)
        
        assert result_tensor.shape[0] == sum(result_list), f"The sum of the list does not match the tensor dimension {result_tensor.shape[0]}, {sum(result_list)}"



        return result_tensor, result_list




    def process_tensor(self, tensor, nav_size, length_threshold=64, similarity_threshold=0.985):
        n, m, t = tensor.shape


        if m % nav_size != 0:
            raise ValueError("m must be divisible by nav_size")

        k = m // nav_size # number of frames 

        if k <= length_threshold:
            result_list = [nav_size] * k
            return tensor, result_list

        elif k == length_threshold + 1:
            split_tensors = tensor.view(n, k, nav_size, t)
            means = split_tensors[:, :k - length_threshold, :, :].mean(dim=2)
            remaining_tensors = split_tensors[:, k - length_threshold:, :, :].reshape(n, -1, t)
            result_tensor = torch.cat([means, remaining_tensors], dim=1)
            result_list = [1] + [nav_size] * length_threshold
            return result_tensor, result_list

        split_tensors = tensor.view(n, k, nav_size, t)

        means_tensor = split_tensors[:, :k - length_threshold, :, :].mean(dim=2)  
        cos = torch.nn.CosineSimilarity(dim=2)

        means = [means_tensor[:, 0:1, :]]
        weights = [1]

        for i in range(1, k - length_threshold):
            last_mean = means[-1]
            current_tensor = means_tensor[:, i:i+1, :]

            similarity = cos(last_mean, current_tensor).mean(dim=0)  

            if similarity > similarity_threshold:
                new_weight = weights[-1] + 1
                new_mean = (last_mean * weights[-1] + current_tensor) / new_weight
                means[-1] = new_mean  
                weights[-1] = new_weight 
            else:
                means.append(current_tensor)
                weights.append(1)  

        means = torch.cat(means, dim=1)




        remaining_tensors = split_tensors[:, k - length_threshold:, :, :].reshape(n, -1, t)

        result_tensor = torch.cat([means, remaining_tensors], dim=1)
        result_list = [1] * means.shape[1] + [nav_size] * length_threshold

        assert result_tensor.shape[1] == sum(result_list), "The sum of the list does not match the tensor dimension"

        return result_tensor, result_list





    def encode_images(self, images, prompts=None, image_counts=None, long_video=False):
        if long_video:
            # use pre-computed features
            image_features = images
        else:
            # (n, 3, 224, 224)
            image_features = self.get_model().get_vision_tower()(images)
            # (n, 257, 1408)

        image_features, video_or_not, nav_or_not, final_token_length_lst = self.vlm_attention(image_features,
                                                                                              prompts=prompts,
                                                                                              image_counts=image_counts,
                                                                                              long_video=long_video)
        return image_features, video_or_not, nav_or_not, final_token_length_lst

    

    def vlm_attention(self, image_features, prompts=None, image_counts=None, long_video=False):
        compress_type = self.config.compress_type
        compress_grid_sizes = {"grid:2": 4, "grid:4": 16, "mean": 1}

        nav_size = compress_grid_sizes.get(compress_type)
        if nav_size is None:
            raise ValueError(f"Unsupported compress type: {compress_type}")

        if image_counts is None:
            assert len(image_features) == len(prompts), f"Size mismatch! image_features: {len(image_features)}, prompts: {len(prompts)}"
        else:
            assert len(prompts) == len(image_counts), f"Size mismatch! prompts: {len(prompts)}, image_counts: {len(image_counts)}"

        img_feat_lst = []
        video_or_not = []
        nav_or_not = []
        final_token_length_lst = []
        total_count = 0

        for _idx, prompt in enumerate(prompts):
            assert isinstance(prompt, list), f"Prompt should be a list, but got {type(prompt)}"

            if image_counts is None:
                img_feat_prompt = image_features[_idx, None]
            else:
                img_feat_prompt = image_features[total_count:total_count + image_counts[_idx]]
                total_count += image_counts[_idx]

            is_navigation = NAVIGATION_IDENTIFIER in prompt[0]
            if is_navigation:
                if image_counts is None or image_counts[_idx] < 1 or len(prompt) != 1: 
                    raise ValueError('[Navigation] wrong')

            if self.config.mm_vision_select_feature == 'patch' and img_feat_prompt.shape[1] % 2 == 1: 
                img_feat_prompt = img_feat_prompt[:, 1:]

            final_token, final_token_nav = self.token_generation(
                img_feat_prompt,
                image_counts=None if image_counts is None else image_counts[_idx],
                navigation=is_navigation
            )

            if is_navigation and final_token_nav is None:
                raise ValueError('[Navigation] wrong')

            final_token = final_token[None].expand(len(prompt), -1, -1, -1).flatten(1, 2) 
            if image_counts is not None:
                if is_navigation: # Navigation
                    final_token_nav = final_token_nav[None].expand(len(prompt), -1, -1, -1).flatten(1, 2)
                    assert final_token_nav.shape[0] == 1 and final_token_nav.shape[1] == 64 and final_token.shape[0] == 1
                    
                    if self.config.run_type == "eval":
                        final_token, lengths_list = self.online_process_tensor(nav_size)
                        final_token = final_token.unsqueeze(0)
                    else:
                        final_token, lengths_list = self.process_tensor(final_token, nav_size)
                    
                    video_or_not.append(True) 
                    final_token_length_lst.append(lengths_list)
                    nav_or_not.append(final_token_nav)

                elif not is_navigation and image_counts[_idx] > 1: # video
                    final_token, lengths_list = self.process_tensor(final_token, nav_size)
                    video_or_not.append(True)
                    final_token_length_lst.append(lengths_list)
                    nav_or_not.append(None)
                    
                elif not is_navigation and image_counts[_idx] == 1: # image
                    video_or_not.append(False)
                    final_token_length_lst.append(None)
                    nav_or_not.append(None)
                    
                else:
                    raise ValueError("unexpected case")
            

            else:
                assert final_token.shape[1] == 64 
                video_or_not.append(False)
                nav_or_not.append(None)
                final_token_length_lst.append(None)

            img_feat_lst.append(final_token)

        return img_feat_lst, video_or_not, nav_or_not, final_token_length_lst



    def token_generation(self, vis_embed, image_counts=None, navigation=False):
        
                                
        def process_grid(vis_embed, grid_size):
            cur_shape = int(vis_embed.shape[1] ** 0.5)
            assert grid_size > 1, f'Grid size should be larger than 1, but got {grid_size}'
            vis_embed = vis_embed.reshape(vis_embed.shape[0], cur_shape, cur_shape, -1)
            
            
            grid_stride = cur_shape // grid_size
            vis_embed = F.avg_pool2d(vis_embed.permute(0, 3, 1, 2),
                                     padding=0,
                                     kernel_size=grid_stride,
                                     stride=grid_stride)
            return vis_embed.permute(0, 2, 3, 1).flatten(1, 2)

    
        grid_size = int(self.config.compress_type.split('grid:')[-1])
        
        if image_counts is None or (image_counts == 1 and not navigation):
            vis_embed = process_grid(vis_embed, 8)
        elif navigation:
               
            vis_embed_nav = vis_embed[-1:]
            vis_embed_nav = process_grid(vis_embed_nav, 8)
            vis_embed = process_grid(vis_embed, grid_size)
        
        else:
            vis_embed = process_grid(vis_embed, grid_size)


        vis_embed = self.get_model().mm_projector(vis_embed)

        if self.config.run_type == "eval":
            temp_embed = self.get_model().feat_cache
            vis_embed = torch.cat([temp_embed, vis_embed], dim=0) if temp_embed is not None else vis_embed
            self.get_model().feat_cache = vis_embed
              
        vis_embed_nav = self.get_model().mm_projector(vis_embed_nav) if navigation else None

        return vis_embed, vis_embed_nav



    def update_prompt(self, prompts=None):
        self.prompts = prompts


    def prepare_inputs_labels_for_multimodal(self, input_ids, attention_mask, past_key_values, labels, images,
                                             prompts=None):
        if 'grid' in self.config.compress_type:
            grid_size = int(self.config.compress_type.split('grid:')[-1])
            if grid_size == 2:
                nav_size = 4
            elif grid_size == 4:
                nav_size = 16
            else:
                raise ValueError
        elif 'mean' in self.config.compress_type:
            nav_size = 1
        else:
            raise ValueError

        if prompts is None and hasattr(self, 'prompts'):
            prompts = self.prompts

        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[
                1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1),
                                            dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels

        long_video = False

        if type(images) is list or images.ndim == 5:
            # not reseshape for long video
            if not long_video:
                images = [image if len(image.shape) == 4 else image.unsqueeze(0) for image in images]
            image_counts = [image.shape[0] for image in images]
            concat_images = torch.cat(images, dim=0)
            image_features, video_or_not, nav_or_not, final_token_length_lst = self.encode_images(concat_images,
                                                                                                  prompts, image_counts,
                                                                                                  long_video=long_video)
        else:
            image_features, video_or_not, nav_or_not, final_token_length_lst = self.encode_images(images, prompts,
                                                                                                  long_video=long_video)

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                # FIXME: this is a hacky fix, for deepspeed zero3 to work
                half_len = cur_input_ids.shape[0] // 2
                if isinstance(image_features, list):
                    cur_image_features = image_features[cur_image_idx][0]
                else:
                    cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0], cur_input_embeds_2], dim=0)
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape

            if not long_video:
                token_idx = 0  
                while image_token_indices.numel() > 0:
                    if isinstance(image_features, list):
                        cur_image_features = image_features[cur_image_idx][token_idx]
                    else:
                        cur_image_features = image_features[cur_image_idx]
                    image_token_start = image_token_indices[0]

                    if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config,
                                                                                      'mm_use_im_start_end', False):
                        raise ValueError('wrong')
                        cur_new_input_embeds.append(
                            self.get_model().embed_tokens(cur_input_ids[:image_token_start - 1]).detach())
                        cur_new_input_embeds.append(
                            self.get_model().embed_tokens(cur_input_ids[image_token_start - 1:image_token_start]))
                        cur_new_input_embeds.append(cur_image_features)
                        cur_new_input_embeds.append(
                            self.get_model().embed_tokens(cur_input_ids[image_token_start + 1:image_token_start + 2]))
                        if labels is not None:
                            cur_new_labels.append(cur_labels[:image_token_start])
                            cur_new_labels.append(
                                torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device,
                                           dtype=labels.dtype))
                            cur_new_labels.append(cur_labels[image_token_start:image_token_start + 1])
                            cur_labels = cur_labels[image_token_start + 2:]
                    else:
                        if nav_or_not[cur_image_idx] is None and video_or_not[cur_image_idx] is False:
                            
                            cur_new_input_embeds.append(
                                self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                            cur_new_input_embeds.append(cur_image_features)
                            assert cur_image_features.shape[0] == 64
                            
                        elif nav_or_not[cur_image_idx] is None and video_or_not[cur_image_idx] is True:
                            
                            cur_new_input_embeds.append(
                                self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                            seperator_token = self.get_model().embed_tokens(cur_input_ids[image_token_start - 1, None])
                            assert not final_token_length_lst[cur_image_idx] is None  
                            video_index = 0
                            for ii_index, ii in enumerate(final_token_length_lst[cur_image_idx]):
                                cur_new_input_embeds.append(cur_image_features[video_index:video_index + ii])
                                if ii_index == len(final_token_length_lst[cur_image_idx]) - 1:
                                    break
                                cur_new_input_embeds.append(seperator_token)
                                video_index += ii
                                
                        else:
                            
                            assert video_or_not[cur_image_idx] is True  
                            assert token_idx == 0  
                            assert nav_or_not[cur_image_idx][token_idx].shape[0] == 64  
                            cur_new_input_embeds.append(
                                self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                            seperator_token = self.get_model().embed_tokens(cur_input_ids[image_token_start - 1, None])
                            assert not final_token_length_lst[cur_image_idx] is None  
                            video_index = 0
                            for ii_index, ii in enumerate(final_token_length_lst[cur_image_idx]):
                                cur_new_input_embeds.append(cur_image_features[video_index:video_index + ii])
                                if ii_index == len(final_token_length_lst[cur_image_idx]) - 1:
                                    break
                                cur_new_input_embeds.append(seperator_token)
                                video_index += ii
                            cur_new_input_embeds.append(self.get_model().embed_tokens(
                                cur_input_ids[image_token_start + 1:image_token_start + 3]))
                            cur_new_input_embeds.append(nav_or_not[cur_image_idx][token_idx])
                            
                            
                            
                        if labels is not None:
                            if nav_or_not[cur_image_idx] is None and video_or_not[cur_image_idx] is False:
                                cur_new_labels.append(cur_labels[:image_token_start])
                                cur_new_labels.append(
                                    torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device,
                                               dtype=labels.dtype))
                                cur_labels = cur_labels[image_token_start + 1:]
                            elif nav_or_not[cur_image_idx] is None and video_or_not[cur_image_idx] is True:
                                cur_new_labels.append(cur_labels[:image_token_start])
                                cur_new_labels.append(
                                    torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device,
                                               dtype=labels.dtype))
                                cur_new_labels.append(
                                    torch.full((int(len(final_token_length_lst[cur_image_idx]) - 1),), IGNORE_INDEX,
                                               device=labels.device, dtype=labels.dtype))
                                cur_labels = cur_labels[image_token_start + 1:]
                            else:
                                cur_new_labels.append(cur_labels[:image_token_start])
                                cur_new_labels.append(
                                    torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device,
                                               dtype=labels.dtype))
                                cur_new_labels.append(
                                    torch.full((int(len(final_token_length_lst[cur_image_idx]) - 1),), IGNORE_INDEX,
                                               device=labels.device, dtype=labels.dtype))
                                cur_new_labels.append(
                                    torch.full((nav_or_not[cur_image_idx][token_idx].shape[0] + 2,), IGNORE_INDEX,
                                               device=labels.device, dtype=labels.dtype))
                                cur_labels = cur_labels[image_token_start + 3:]

                    if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config,
                                                                                      'mm_use_im_start_end', False):
                        raise ValueError('wrong')
                    else:
                        if nav_or_not[cur_image_idx] is not None:
                            cur_input_ids = cur_input_ids[image_token_start + 3:]
                        else:
                            cur_input_ids = cur_input_ids[image_token_start + 1:]
                    image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
                    token_idx += 1

                # changle image idx after processing one sample
                cur_image_idx += 1
                if cur_input_ids.numel() > 0:
                    if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config,
                                                                                      'mm_use_im_start_end', False):
                        cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
                    else:
                        cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                    if labels is not None:
                        cur_new_labels.append(cur_labels)
                cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
                cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
                new_input_embeds.append(cur_new_input_embeds)
                if labels is not None:
                    cur_new_labels = torch.cat(cur_new_labels, dim=0)
                    assert cur_new_input_embeds.shape[0] == cur_new_labels.shape[0]
                    new_labels.append(cur_new_labels)
            else:
                cur_new_input_embeds = torch.Tensor(len(cur_input_ids), self.config.hidden_size).to(dtype=self.dtype,
                                                                                                    device=self.device)
                text_token_indices = torch.where(cur_input_ids != IMAGE_TOKEN_INDEX)[0]
                if not self.training and self.get_model().embed_tokens.weight.device != cur_input_ids.device:
                    model_device = self.get_model().embed_tokens.weight.device
                    data_device = cur_input_ids.device
                    cur_input_ids_text = cur_input_ids[text_token_indices].to(device=model_device)
                    cur_new_input_embeds[text_token_indices] = self.get_model().embed_tokens(cur_input_ids_text).to(
                        device=data_device)
                else:
                    cur_new_input_embeds[text_token_indices] = self.get_model().embed_tokens(
                        cur_input_ids[text_token_indices])
                cur_image_features = image_features[cur_image_idx]
                cur_new_input_embeds[image_token_indices] = cur_image_features
                new_input_embeds.append(cur_new_input_embeds)
                if labels is not None:
                    new_labels.append(cur_labels)
                cur_image_idx += 1

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed,
                                           torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]),
                                                       dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label,
                                               torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX,
                                                          dtype=cur_new_label.dtype, device=cur_new_label.device)),
                                              dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            # only used for right padding in tokenlizer
            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels,
                                                                                    new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True,
                                                        dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],),
                                                         False, dtype=attention_mask.dtype,
                                                         device=attention_mask.device)
                    cur_new_attention_mask = torch.cat(
                        (new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)

            # only used for right padding in tokenlizer
            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full(
                    (attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True,
                    dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels
    def initialize_vision_tokenizer(self, model_args, tokenizer):
        tokenizer.add_tokens([VIDEO_START_SPECIAL_TOKEN, VIDEO_END_SPECIAL_TOKEN, IMAGE_START_TOKEN, IMAGE_END_TOKEN, NAVIGATION_SPECIAL_TOKEN, IAMGE_SEPARATOR], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

   


