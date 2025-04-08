from typing import Dict

import numpy as np
import torch
import torch.nn.functional as nnf
from torch import nn, Tensor
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    InstructBlipQFormerModel,
    InstructBlipQFormerConfig
)

from models.common.position_embedding import PositionEmbeddingCoordsSine
from models.perla.generation_utils import generation
from models.perla.merger import LearnCut
from libs.box_util import box3d_iou_batch_tensor


def save_point_cloud_to_ply(points, colors=None, filename='fps'):
    """
    Save a point cloud to a PLY file.

    Args:
        filename (str): The file path to save the PLY file.
        points (numpy.ndarray): A (N, 3) array of XYZ coordinates.
        colors (numpy.ndarray, optional): A (N, 3) array of RGB values.
    """
    num_points = points.shape[0]

    # Define the PLY header
    header = [
        'ply',
        'format ascii 1.0',
        f'element vertex {num_points}',
        'property float x',
        'property float y',
        'property float z'
    ]

    # Add color properties if colors are provided
    if colors is not None:
        header += [
            'property uchar red',
            'property uchar green',
            'property uchar blue'
        ]

    header.append('end_header')

    # Combine points and colors (if provided)
    if colors is not None:
        data = np.hstack([points, colors])
    else:
        data = points

    # Save to PLY file
    with open(filename, 'w') as ply_file:
        # Write header
        ply_file.write('\n'.join(header) + '\n')

        # Write point data
        for point in data:
            ply_file.write(' '.join(map(str, point)) + '\n')


def proposal_dimension_select(features: Tensor, indices: Tensor) -> Tensor:
    '''
    
    Parameters
    ----------
    features : Tensor, with size [batch x nsrc x ...]
        Data bank, from which to gather information.
    indices : Tensor, with size [batch x ntgt]
        Indices for gathering information from data bank.

    Returns
    -------
    Tensor, with size [batch x ntgt x ...]
        Gathers features in proposal dimension.
    
    '''
    return torch.gather(
        features, 1,
        indices.reshape(
            *(indices.shape + tuple(1 for _ in features.shape[2:]))
        ).repeat(
            *((1, 1) + features.shape[2:])
        )
    )


def select_proposal_feature(
        prop_features: Tensor, prop_box_corners: Tensor, prop_sem_mask: Tensor,  box_query: Tensor
) -> Tensor:
    '''
    
    Parameters
    ----------
    prop_features : Tensor, with size [batch x nproposal x n_embd]
    prop_box_corners : Tensor, with size [batch x nproposal x 8 x 3]
    prop_sem_mask : Tensor, with size [batch x nproposal], 0 for background
    box_query : Tensor, with size [batch x nquery x 8 x 3]
    center_xyz : Tensor, with size [batch x nquery x 3]

    Returns
    -------
    Tensor, with size [batch x nquery x n_embd]
        Gathers features in proposal dimension.
    
    '''
    # prop_features
    batch_size, nproposal, _, _ = prop_box_corners.shape
    nquery = box_query.shape[1]

    matched_box_iou = box3d_iou_batch_tensor(
        prop_box_corners.unsqueeze(1).repeat(1, nquery, 1, 1, 1).reshape(-1, 8, 3),
        box_query.unsqueeze(2).repeat(1, 1, nproposal, 1, 1).reshape(-1, 8, 3)
    )
    matched_box_iou = matched_box_iou.reshape(batch_size, nquery, nproposal)
    matched_box_iou = matched_box_iou * prop_sem_mask.unsqueeze(1)

    matched_indices = matched_box_iou.argmax(-1)  # batch x nquery

    return proposal_dimension_select(prop_features, matched_indices)


class PromptEncoder(nn.Module):

    def __init__(self, encoder_hidden_size, visual_nquery, qformer_hidden_size, n_embd):
        super(PromptEncoder, self).__init__()
        self.n_embd = n_embd
        self.visual_nquery = visual_nquery
        self.qformer_hidden_size = qformer_hidden_size
        self.encoder_hidden_size = encoder_hidden_size

        self.box_prompt_projector = nn.Sequential(
            nn.Linear(encoder_hidden_size, qformer_hidden_size),
            nn.ReLU(),
            nn.Linear(qformer_hidden_size, visual_nquery * qformer_hidden_size),
        )
        self.click_prompt_projector = nn.Sequential(
            nn.Linear(2 * encoder_hidden_size, qformer_hidden_size),
            nn.ReLU(),
            nn.Linear(qformer_hidden_size, visual_nquery * qformer_hidden_size),
        )
        self.pos_emb3d = PositionEmbeddingCoordsSine(
            d_pos=encoder_hidden_size,
            pos_type='fourier',
            normalize=True
        )

    def enhance(self, prompt_xyz, xyz, prompt_fea, fea):
        ids = torch.cdist(prompt_xyz, xyz).min(dim=-1)[1]  # BxQ
        k_fea = torch.gather(fea, 1, ids.unsqueeze(-1).expand(-1, -1, self.encoder_hidden_size))
        return torch.cat([k_fea.permute(0, 2, 1), prompt_fea], dim=1)

    def expand_prompt_representation(self, prompt_feature: Tensor, prompt_mask: Tensor = None):
        # input:
        #   prompt_feature: batch x nprompt x (ntkn x channel)
        #   prompt_mask: batch x nprompt
        # output:
        #   prompt_feature: batch x (nprompt x ntkn) x channel
        #   prompt_mask: batch x (nprompt x ntkn)
        batch_size, nprompt = prompt_feature.shape[:2]
        if prompt_mask is None:
            prompt_mask = torch.ones_like(prompt_feature[..., 0])
        prompt_mask = prompt_mask.unsqueeze(-1).repeat(1, 1, self.visual_nquery)
        prompt_mask = prompt_mask.reshape(batch_size, nprompt * self.visual_nquery)
        prompt_feature = prompt_feature.reshape(batch_size, nprompt, self.visual_nquery, self.qformer_hidden_size)
        prompt_feature = prompt_feature.reshape(batch_size, nprompt * self.visual_nquery, self.qformer_hidden_size)
        return prompt_feature, prompt_mask

    def forward(self,
                detector_output,
                point_cloud_dims,
                box_query=None,
                box_qmask=None,
                click_query=None,
                click_qmask=None,
                ):
        sem_cls_logits = detector_output['sem_cls_logits']
        prop_sem_mask = (sem_cls_logits.argmax(-1) != (sem_cls_logits.shape[-1] - 1)).float()

        net_device = sem_cls_logits.device
        batch_size = sem_cls_logits.shape[0]

        ### prompt encoding
        # box prompt encoding
        visual_prompt = [torch.zeros(batch_size, 0, self.qformer_hidden_size).to(net_device)]
        visual_mask = [torch.zeros(batch_size, 0).to(net_device)]
        if box_query is not None:
            box_prompt = select_proposal_feature(
                detector_output['prop_features'][-1],
                detector_output['box_corners'],
                prop_sem_mask,
                box_query
            )
            box_prompt = self.box_prompt_projector(box_prompt)
            box_prompt, box_qmask = self.expand_prompt_representation(box_prompt, box_qmask)
            visual_prompt.append(box_prompt)
            visual_mask.append(box_qmask)

        # click prompt encoding: batch x nquery x nproposal
        enc_xyz = detector_output['enc_xyz']
        if click_query is not None:
            click_xyz = click_query  # batch x nquery x 3
            click_prompt = self.pos_emb3d(click_xyz, input_range=point_cloud_dims)
            encoder_hidden_states = detector_output['enc_features']
            click_prompt = self.enhance(click_xyz, enc_xyz, click_prompt, encoder_hidden_states)
            click_prompt = self.click_prompt_projector(click_prompt.permute(0, 2, 1))
            click_prompt, click_qmask = self.expand_prompt_representation(click_prompt, click_qmask)
            visual_prompt.append(click_prompt)
            visual_mask.append(click_qmask)

        ## concat box and click prompts as well as prompt masks
        prompt_feature = torch.cat(visual_prompt, dim=1)  # batch x (2 x ntoken) x channel
        prompt_mask = torch.cat(visual_mask, dim=1)  # batch x (2 x ntoken)

        return prompt_feature, prompt_mask


class Captioner(nn.Module):

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_llm is True:
            self.transformer.eval()
            for param in self.transformer.parameters():
                param.requires_grad = False
        return self

    def __init__(self, args, nlatent_query=32):
        super(Captioner, self).__init__()

        self.encoder_hidden_size = 256
        self.dtype = torch.float16
        self.visual_nquery = 8
        self.nlatent_query = nlatent_query
        self.freeze_llm = args.freeze_llm

        ## initialize tokenizer for batch decoding
        self.tokenizer = AutoTokenizer.from_pretrained(args.vocab)
        self.nvocabs = len(self.tokenizer)

        ## caption generation cores
        self.transformer = AutoModelForCausalLM.from_pretrained(
            args.vocab,
            torch_dtype=self.dtype
        )
        self.n_embd = self.transformer.config.hidden_size

        ## Multi-modality Transformer
        qformer_config = InstructBlipQFormerConfig(
            num_hidden_layers=6,
            encoder_hidden_size=self.encoder_hidden_size
        )
        self.qformer = InstructBlipQFormerModel.from_pretrained(
            args.qformer_vocab,
            config=qformer_config
        )
        self.qformer_hidden_size = qformer_config.hidden_size

        ## for prompt feature projection
        self.encoder_to_qformer_projection = nn.Sequential(
            nn.Linear(self.encoder_hidden_size, qformer_config.encoder_hidden_size),
            nn.ReLU(),
            nn.Linear(qformer_config.encoder_hidden_size, qformer_config.encoder_hidden_size),
            nn.ReLU(),
        )
        self.prompt_encoder = PromptEncoder(
            self.encoder_hidden_size,
            self.visual_nquery,
            self.qformer_hidden_size,
            self.n_embd
        )
        self.latent_query = nn.Embedding(self.nlatent_query, self.qformer_hidden_size)
        self.qformer_to_language_projection = nn.Linear(self.qformer_hidden_size, self.n_embd)

        self.en_slic = LearnCut(self.encoder_hidden_size, self.encoder_hidden_size,
                                args.n_neighs, n_clus=args.n_clus, n_split=args.n_splits, radius=1.5, tau=1e-1)

        self.max_gen_per_iter = 8
        # ---- super parameters for evaluation
        self.caption_config = {
            'early_stopping': True,
            'eos_token_id': self.tokenizer.eos_token_id,
            'num_beams': 4 if args.use_beam_search is True else None,
        }
        self.train()

        # Initialize the reference point with the current model parameters
        self.reference_point = {name: param.clone().detach() for name, param in self.named_parameters() if
                                param.requires_grad}

    def loss_caption(self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, reg=-1) -> torch.Tensor:
        """
        Compute the mirror descent regularized loss for captioning.

        Args:
            logits (Tensor): The predicted logits of shape (batch_size, seq_len, vocab_size).
            target (Tensor): The ground truth target indices of shape (batch_size, seq_len).
            mask (Tensor): The mask tensor of shape (batch_size, seq_len) to ignore padding tokens.

        Returns:
            Tensor: The computed loss.
        """
        # Compute the per-word cross-entropy loss
        loss_per_word = nnf.cross_entropy(
            logits.permute(0, 2, 1).contiguous(),  # Change shape to (batch_size, vocab_size, seq_len)
            target,
            reduction='none',
        )

        # Apply the mask and normalize the loss
        final_loss = torch.sum(loss_per_word * mask) / torch.sum(mask + 1e-6)
        
        # Ensure the cross-entropy loss is finite
        if not torch.isfinite(final_loss):
            raise ValueError("Cross-Entropy loss is not finite.")

        # Mirror descent regularization
        if reg > 0:
            for name, param in self.named_parameters():
                if param.requires_grad:
                    # Calculate the distance from the reference point
                    distance = torch.norm(param - self.reference_point[name])
                    # Add the distance to the loss (adjust the weight as needed)
                    final_loss += reg * distance  # Example weight

        return final_loss

    def update_reference_point(self):
        """Update the reference point to the current model parameters."""
        self.reference_point = {name: param.clone().detach() for name, param in self.named_parameters() if
                                param.requires_grad}

    def _get_instruction_response(self,
                                  detector_output: dict,
                                  inputs: dict,
                                  box_query: Tensor = None,
                                  box_qmask: Tensor = None,
                                  click_query: Tensor = None,
                                  click_qmask: Tensor = None
                                  ):

        point_cloud_dims = [
            inputs["point_cloud_dims_min"],
            inputs["point_cloud_dims_max"],
        ]
        net_device = inputs["point_clouds"].device
        batch_size = inputs["point_clouds"].shape[0]
        encoder_hidden_states = detector_output['enc_features']
        l_enc_xyz = detector_output['l_enc_xyz']
        l_enc_feats = detector_output['l_enc_feats']
        enc_xyz = detector_output['enc_xyz']
        sp_spts = detector_output['enc_spts']
        l_spt_list = detector_output['l_enc_spts']
        merge_features = encoder_hidden_states
        merge_features, l_loss, g_loss = self.en_slic(merge_features, enc_xyz, l_enc_feats, l_enc_xyz, sp_spts, l_spt_list)

        # prompt encoding
        prompt_feature, prompt_mask = self.prompt_encoder(
            detector_output,
            point_cloud_dims,
            box_query=box_query,
            box_qmask=box_qmask,
            click_query=click_query,
            click_qmask=click_qmask
        )
        # gather query feature for qformer: batch x (n_query + n_tokens) x n_embd
        query_tokens = self.latent_query.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        query_tokens = torch.cat((query_tokens, prompt_feature), dim=1)
        query_attention_mask = torch.cat((torch.ones(batch_size, self.nlatent_query).to(net_device),
                                          prompt_mask), dim=1)

        # prepare qformer inputs: batch x ntoken x n_embd
        query_attention_mask = torch.cat((query_attention_mask, inputs['qformer_attention_mask']), dim=1)

        query_outputs = self.qformer(
            input_ids=inputs['qformer_input_ids'],
            attention_mask=query_attention_mask,
            query_embeds=query_tokens,
            encoder_hidden_states=self.encoder_to_qformer_projection(merge_features),
        )
        
        query_outputs = query_outputs[0][:, : self.nlatent_query, :]
        prefix_feature = self.qformer_to_language_projection(query_outputs)

        return prefix_feature, l_loss, g_loss
    
    def forward(self, detector_output: dict, inputs: dict, is_eval: bool = False, task_name: str = 'qa') -> dict:

        if is_eval is False:
            return self.forward_training(detector_output, inputs)

        response_config = {
            'ov-det': 64,
            'dense-cap': 48,
            'qa': 16,
            'chat': 512,
        }
        max_gen_length = response_config[task_name]

        if task_name in {'ov-det', 'dense-cap'}:
            return self.predict_densecap(detector_output, inputs, task_name, max_gen_length=max_gen_length)
        elif task_name == 'qa':
            return self.predict_answer(detector_output, inputs, max_gen_length=max_gen_length)
        else:
            return self.predict_chat(detector_output, inputs, max_gen_length=max_gen_length)

    def forward_training(self, detector_output: Dict, inputs: Dict) -> Dict:
        # get word embeddings, NOTE: captioner does not predict <bos> token
        input_ids = inputs['input_ids']  # batch x ntokens
        input_mask = inputs['attention_mask']  # batch x ntokens
        gradient_mask = inputs['gradient_mask']  # batch x ntokens

        box_query = inputs.get('box_query', None)  # batch x nquery x 8 x 3
        box_qmask = inputs.get('box_mask', None)  # batch x nquery
        click_query = inputs.get('click_query', None)  # batch x nquery x 3
        click_qmask = inputs.get('click_mask', None)  # batch x nquery

        embedding_layer = self.transformer.get_input_embeddings()

        # ---- batch x ntoken x n_embd
        prefix_tokens, l_loss, g_loss = self._get_instruction_response(
            detector_output=detector_output,
            inputs=inputs,
            box_query=box_query,
            box_qmask=box_qmask,
            click_query=click_query,
            click_qmask=click_qmask
        )
        prefix_mask = torch.ones_like(prefix_tokens[..., 0])
        # ---- batch x (ntoken + nword) x n_embd
        inputs_embeds = torch.cat((prefix_tokens, embedding_layer(input_ids)), dim=1)
        attention_mask = torch.cat((prefix_mask, input_mask), dim=1)

        # ---- calculate transformer loss
        outputs = self.transformer(
            inputs_embeds=inputs_embeds.to(self.dtype),
            attention_mask=attention_mask.to(self.dtype),
        )

        detector_output['loss'] += self.loss_caption(
            logits=outputs.logits[:, prefix_tokens.shape[1] - 1: -1],
            target=input_ids,
            mask=gradient_mask.to(self.dtype),
        )
        detector_output['l_loss'] = l_loss
        detector_output['g_loss'] = g_loss
        return detector_output

    # def loss_caption(self, logits: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    #     loss_per_word = nnf.cross_entropy(
    #         logits.permute(0, 2, 1).contiguous(),
    #         target,
    #         reduction='none',
    #     )
    #     final_loss = torch.sum(loss_per_word * mask) / torch.sum(mask + 1e-6)
    #     # parameter activation for multi-gpu training
    #     for param in self.parameters():
    #         if param.requires_grad:
    #             final_loss += 0 * torch.sum(param.to(final_loss.dtype) ** 2)
    #     return final_loss

    def predict_densecap(self, detector_output: Dict, inputs: Dict, task_name: str, max_gen_length: int = 64) -> Dict:
        # ---- necessary elements
        embedding_layer = self.transformer.get_input_embeddings()
        net_device = next(self.parameters()).device
        batch_size, nproposals, _, _ = detector_output['box_corners'].shape
        # ---- to store llm outputs
        output_ids = torch.ones(batch_size, nproposals, max_gen_length).long().to(net_device)
        output_ids = output_ids * self.tokenizer.eos_token_id
        # ---- llm input preparation
        instruction = inputs['instruction'][0]  # ntoken
        instruction_mask = inputs['instruction_mask'][0]  # ntoken
        instruction_id = instruction[instruction_mask == 1]  # ntoken
        instruction_id = instruction_id[None, :].repeat(batch_size, 1)
        instruction_embedding = embedding_layer(instruction_id)  # batch x ntoken x n_embd

        prefix_tokens = []
        l_loss = 0.0
        g_loss = 0.0
        for proposal_id in range(nproposals):
            box_query = detector_output['box_corners'][:, [proposal_id]]  # batch x 1 x 8 x 3

            click_query = None
            if task_name == 'ov-det':
                click_query = detector_output['query_xyz'][:, [proposal_id]]  # batch x 1 x 3

            instruct_prefix_feature, l_loss_i, g_loss_i = self._get_instruction_response(  # batch x ntoken x n_embd
                detector_output=detector_output,
                inputs=inputs,
                box_query=box_query,  # batch x 1 x 8 x 3
                click_query=click_query,
            )
            l_loss += l_loss_i
            g_loss += g_loss_i
            instruct_prefix_feature = torch.cat((instruct_prefix_feature, instruction_embedding), dim=1)
            prefix_tokens.append(instruct_prefix_feature.unsqueeze(1))
        # batch x nproposal x 1 x n_embd
        prefix_tokens = torch.cat(prefix_tokens, dim=1).to(self.dtype)

        ## filter and rank the queries
        sem_cls_logits = detector_output["sem_cls_logits"]
        objectness_mask = sem_cls_logits.argmax(-1) != (sem_cls_logits.shape[-1] - 1)

        ## limit the proposals for generating captions
        candidate_prefix = prefix_tokens[objectness_mask].to(self.dtype)

        gather_output_ids = []
        for start_idx in range(0, candidate_prefix.shape[0], self.max_gen_per_iter):
            prefix = candidate_prefix[start_idx: start_idx + self.max_gen_per_iter]
            scene_cap_output = generation(
                self.transformer,
                inputs_embeds=prefix,
                max_length=max_gen_length,
                **self.caption_config
            )
            gather_output_ids.append(scene_cap_output['output_ids'])
        gather_output_ids = torch.cat(gather_output_ids, dim=0)

        output_ids[objectness_mask] = gather_output_ids
        detector_output['output_ids'] = output_ids
        detector_output['l_loss'] = l_loss
        detector_output['g_loss'] = g_loss

        return detector_output

    def predict_answer(self, detector_output: Dict, inputs: Dict, max_gen_length: int = 8) -> Dict:

        # ---- necessary elements
        embedding_layer = self.transformer.get_input_embeddings()
        # net_device = next(self.parameters()).device
        # ---- to store llm outputs
        output_ids = []

        # ---- llm input preparation
        instruction = inputs['instruction']  # ntoken
        instruction_mask = inputs['instruction_mask']  # ntoken

        prefix_tokens, l_loss, g_loss = self._get_instruction_response(
            detector_output=detector_output,
            inputs=inputs,
        )
        prefix_tokens = prefix_tokens.to(self.dtype)

        for batch_id in range(prefix_tokens.shape[0]):
            sample_instruction = instruction[batch_id]
            sample_mask = instruction_mask[batch_id]  # ntoken

            output = generation(
                self.transformer,
                inputs_embeds=torch.cat(
                    [
                        prefix_tokens[batch_id].unsqueeze(0),  # 1 x nprefix x n_embd
                        embedding_layer(sample_instruction[sample_mask == 1]).unsqueeze(0)
                    ],
                    dim=1
                ),
                max_length=max_gen_length,
                **self.caption_config
            )
            output_ids.append(output['output_ids'])

        output_ids = torch.cat(output_ids, dim=0)
        detector_output['output_ids'] = output_ids
        detector_output['l_loss'] = l_loss
        detector_output['g_loss'] = g_loss

        return detector_output

    def predict_chat(self, detector_output: Dict, inputs: Dict, max_gen_length: int = 512) -> Dict:

        # ---- necessary elements
        embedding_layer = self.transformer.get_input_embeddings()
        net_device = next(self.parameters()).device
        # ---- to store llm outputs
        output_ids = []

        # ---- llm input preparation
        instruction = inputs['instruction']  # ntoken
        instruction_mask = inputs['instruction_mask']  # ntoken

        prefix_tokens, l_loss, g_loss = self._get_instruction_response(
            detector_output=detector_output,
            inputs=inputs,
        )
        prefix_tokens = prefix_tokens.to(self.dtype)

        for batch_id in range(prefix_tokens.shape[0]):
            sample_instruction = instruction[batch_id]
            sample_mask = instruction_mask[batch_id]  # ntoken

            output = self.transformer.generate(
                inputs_embeds=torch.cat(
                    [
                        prefix_tokens[batch_id].unsqueeze(0),  # 1 x nprefix x n_embd
                        embedding_layer(sample_instruction[sample_mask == 1]).unsqueeze(0)
                    ],
                    dim=1
                ),
                max_new_tokens=max_gen_length,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                no_repeat_ngram_size=4,
                num_return_sequences=1,
            )  # 1 x max_gen_length
            output = output.squeeze(0)
            placeholder = torch.ones(max_gen_length).to(net_device) * self.tokenizer.eos_token_id
            output = output[:min(max_gen_length, output.shape[0])]
            placeholder[:output.shape[0]] = output

            output_ids.append(placeholder.unsqueeze(0).long())

        output_ids = torch.cat(output_ids, dim=0)
        detector_output['output_ids'] = output_ids
        detector_output['l_loss'] = l_loss
        detector_output['g_loss'] = g_loss

        return detector_output
