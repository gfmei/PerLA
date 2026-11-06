import contextlib
from typing import Dict

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
from models.perla.rotary import Rotary3D, add_cross_attention_rotary, cross_attention_angles
from libs.box_util import box3d_iou_batch_tensor


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


def match_proposals(prop_box_corners: Tensor, prop_sem_mask: Tensor, box_query: Tensor):
    '''
    The foreground proposal with the highest IoU for each query box, and that IoU.

    prop_box_corners : batch x nproposal x 8 x 3
    prop_sem_mask : batch x nproposal, 0 for background
    box_query : batch x nquery x 8 x 3
    Returns indices and IoU, each batch x nquery.
    '''
    batch_size, nproposal, _, _ = prop_box_corners.shape
    nquery = box_query.shape[1]

    matched_box_iou = box3d_iou_batch_tensor(
        prop_box_corners.unsqueeze(1).repeat(1, nquery, 1, 1, 1).reshape(-1, 8, 3),
        box_query.unsqueeze(2).repeat(1, 1, nproposal, 1, 1).reshape(-1, 8, 3)
    )
    matched_box_iou = matched_box_iou.reshape(batch_size, nquery, nproposal)
    matched_box_iou = matched_box_iou * prop_sem_mask.unsqueeze(1)

    matched_indices = matched_box_iou.argmax(-1)  # batch x nquery
    return matched_indices, matched_box_iou.gather(-1, matched_indices.unsqueeze(-1)).squeeze(-1)


def select_proposal_feature(
        prop_features: Tensor, prop_box_corners: Tensor, prop_sem_mask: Tensor,  box_query: Tensor
) -> Tensor:
    '''Features (batch x nquery x n_embd) of the proposals matched to the query boxes.'''
    matched_indices, _ = match_proposals(prop_box_corners, prop_sem_mask, box_query)
    return proposal_dimension_select(prop_features, matched_indices)


def aabb_iou(corners_a: Tensor, corners_b: Tensor) -> Tensor:
    """IoU of the axis-aligned extents of two box sets: batch x A x 8 x 3, batch x B x 8 x 3 -> batch x A x B."""
    low_a, high_a = corners_a.min(-2).values, corners_a.max(-2).values
    low_b, high_b = corners_b.min(-2).values, corners_b.max(-2).values
    low = torch.maximum(low_a[:, :, None], low_b[:, None])
    high = torch.minimum(high_a[:, :, None], high_b[:, None])
    intersection = (high - low).clamp_min(0).prod(-1)
    volume_a = (high_a - low_a).clamp_min(0).prod(-1)
    volume_b = (high_b - low_b).clamp_min(0).prod(-1)
    union = volume_a[:, :, None] + volume_b[:, None] - intersection
    return intersection / union.clamp_min(1e-9)


OCTANT_SIGNS = torch.tensor([[1. if octant >> (2 - axis) & 1 else -1. for axis in range(3)]
                             for octant in range(8)])


def pool_box_parts(features: Tensor, xyz: Tensor, box_query: Tensor, margin: float = 0.1):
    """Average the encoder tokens in each octant of each (slightly enlarged) box.

    features: batch x ntoken x channel, xyz: batch x ntoken x 3, box_query: batch x nbox x 8 x 3.
    Octant o holds points above the box center along x, y, z where bits 2, 1, 0 of o are set.
    Returns pooled features batch x nbox x 8 x channel, positions batch x nbox x 8 x 3 (the
    members' mean, or the octant's center when empty) and member counts batch x nbox x 8.
    """
    low, high = box_query.min(2).values, box_query.max(2).values
    extent, center = high - low, (low + high) / 2
    points = xyz[:, None]
    inside = ((points >= (low - margin * extent)[:, :, None])
              & (points <= (high + margin * extent)[:, :, None])).all(-1)
    above = (points > center[:, :, None]).long()
    octant = above[..., 0] * 4 + above[..., 1] * 2 + above[..., 2]
    member = inside[:, :, None] & (octant[:, :, None] == torch.arange(8, device=xyz.device)[:, None])
    weight = member.to(features.dtype)
    count = weight.sum(-1)
    scale = count.clamp_min(1)[..., None]
    pooled = torch.einsum('bpon,bnc->bpoc', weight, features) / scale
    position = torch.einsum('bpon,bnd->bpod', weight, xyz.to(features.dtype)) / scale
    octant_center = center[:, :, None] + OCTANT_SIGNS.to(xyz) * extent[:, :, None] / 4
    position = torch.where(count[..., None] > 0, position, octant_center.to(position.dtype))
    return pooled, position, count


def context_neighbors(box_corners: Tensor, sem_cls_logits: Tensor, count: int, overlap: float = 0.25):
    """The `count` nearest other objects for every proposal: indices and validity, batch x nproposal x count.

    The detector emits several near-duplicate proposals per object, so neighbors come from
    one representative per object (fast NMS over foreground proposals by foreground
    probability) and must not overlap the target itself.
    """
    probability = sem_cls_logits.softmax(-1)
    foreground = sem_cls_logits.argmax(-1) != sem_cls_logits.shape[-1] - 1
    score = torch.where(foreground, 1 - probability[..., -1], torch.full_like(probability[..., -1], -1.0))
    nproposal = score.shape[1]
    iou = aabb_iou(box_corners, box_corners)
    order = torch.arange(nproposal, device=score.device)
    # i outranks j by score, ties broken by index.
    outranks = (score[:, :, None] > score[:, None]) | (
        (score[:, :, None] == score[:, None]) & (order[:, None] < order[None]))
    suppressed = (outranks & foreground[:, :, None] & (iou > overlap)).any(1)
    representative = foreground & ~suppressed
    candidate = representative[:, None] & (iou <= overlap)  # excludes each target itself
    centers = box_corners.mean(2)
    distance = torch.cdist(centers, centers).masked_fill(~candidate, float('inf'))
    nearest = distance.topk(min(count, nproposal), dim=-1, largest=False)
    indices, valid = nearest.indices, torch.isfinite(nearest.values)
    if count > nproposal:
        pad = count - nproposal
        indices = nnf.pad(indices, (0, pad))
        valid = nnf.pad(valid, (0, pad), value=False)
    return indices, valid


class PromptEncoder(nn.Module):

    def __init__(self, encoder_hidden_size, visual_nquery, qformer_hidden_size, n_embd,
                 context_objects=0, part_tokens=False, part_colors=False):
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
        if context_objects < 0:
            raise ValueError("The number of context objects cannot be negative")
        self.context_objects = context_objects
        self.part_tokens = part_tokens
        if context_objects or part_tokens:
            # Offsets in meters from the prompted box center.
            self.offset_code = PositionEmbeddingCoordsSine(
                d_pos=encoder_hidden_size, pos_type='fourier', normalize=False)
        self.part_colors = part_tokens and part_colors
        if part_tokens:
            # The proposal feature is a single vector. Part tokens add what each region of the
            # object looks like (from the denser local encoder tokens) and where it is. The
            # detector was trained for boxes and may keep little color, so each part also gets
            # the mean and spread of its raw point colors.
            color_stats = 6 if self.part_colors else 0
            self.part_projector = nn.Sequential(
                nn.Linear(2 * encoder_hidden_size + color_stats, qformer_hidden_size),
                nn.ReLU(),
                nn.Linear(qformer_hidden_size, qformer_hidden_size),
            )
        if context_objects:
            # Relational descriptions name anchor objects ("the chair next to the desk"):
            # one token per nearby object, its detector feature plus its offset in meters.
            self.context_projector = nn.Sequential(
                nn.Linear(2 * encoder_hidden_size, qformer_hidden_size),
                nn.ReLU(),
                nn.Linear(qformer_hidden_size, qformer_hidden_size),
            )

    def part_prompt(self, detector_output, box_query: Tensor, prompt_mask: Tensor, point_clouds=None):
        """Eight part tokens per box, their mask and positions: batch x (nbox x 8)."""
        features, xyz = detector_output.get('l_enc_feats'), detector_output.get('l_enc_xyz')
        if features is None:  # no local split: use the scene tokens
            features, xyz = detector_output['enc_features'], detector_output['enc_xyz']
        pooled, position, count = pool_box_parts(features, xyz, box_query)
        offset = (position - box_query.mean(2)[:, :, None]).flatten(1, 2)
        parts = [pooled.flatten(1, 2), self.offset_code(offset).permute(0, 2, 1)]
        present = count.gt(0)
        if self.part_colors:
            if point_clouds is None or point_clouds.shape[-1] < 6:
                raise ValueError("Part colors need point clouds with RGB in channels 3-5")
            # Colors are (rgb - mean) / 256; scale them to roughly unit spread.
            color = point_clouds[..., 3:6] * 4
            moments, _, points = pool_box_parts(torch.cat((color, color.square()), dim=-1),
                                                point_clouds[..., :3], box_query)
            mean, square = moments.split(3, dim=-1)
            spread = (square - mean.square()).clamp_min(0).sqrt()
            parts.append(torch.cat((mean, spread), dim=-1).flatten(1, 2).to(pooled.dtype))
            present = present | points.gt(0)
        tokens = self.part_projector(torch.cat(parts, dim=-1))
        valid = present & prompt_mask.bool()[..., None]
        return tokens, valid.flatten(1).to(tokens.dtype), position.flatten(1, 2)

    def context_prompt(self, detector_output, target: Tensor, prompt_mask: Tensor):
        """Tokens, mask and centers for the objects nearest each prompted proposal.

        target and prompt_mask: batch x nprompt. Returns batch x (nprompt x count) tokens.
        """
        # Computed once per scene; dense captioning prompts every proposal of a scene in turn.
        table = detector_output.get('_context_neighbors')
        if table is None:
            table = context_neighbors(detector_output['box_corners'], detector_output['sem_cls_logits'],
                                      self.context_objects)
            detector_output['_context_neighbors'] = table
        neighbor_index, neighbor_valid = table
        batch = torch.arange(target.shape[0], device=target.device)[:, None]
        index = neighbor_index[batch, target].flatten(1)  # batch x (nprompt x count)
        valid = (neighbor_valid[batch, target] & prompt_mask.bool()[..., None]).flatten(1)
        centers = detector_output['box_corners'].mean(2)
        neighbor_center = centers[batch, index]
        offset = neighbor_center - centers[batch, target].repeat_interleave(self.context_objects, dim=1)
        feature = proposal_dimension_select(detector_output['prop_features'][-1], index)
        code = self.offset_code(offset).permute(0, 2, 1)
        tokens = self.context_projector(torch.cat((feature, code), dim=-1))
        return tokens, valid.to(tokens.dtype), neighbor_center

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
                point_clouds=None,
                ):
        sem_cls_logits = detector_output['sem_cls_logits']
        prop_sem_mask = (sem_cls_logits.argmax(-1) != (sem_cls_logits.shape[-1] - 1)).float()

        net_device = sem_cls_logits.device
        batch_size = sem_cls_logits.shape[0]

        ### prompt encoding
        # box prompt encoding; positions follow the token order for rotary attention
        visual_prompt = [torch.zeros(batch_size, 0, self.qformer_hidden_size).to(net_device)]
        visual_mask = [torch.zeros(batch_size, 0).to(net_device)]
        visual_xyz = [torch.zeros(batch_size, 0, 3, device=net_device)]
        if box_query is not None:
            matched, _ = match_proposals(detector_output['box_corners'], prop_sem_mask, box_query)
            box_prompt = proposal_dimension_select(detector_output['prop_features'][-1], matched)
            # The mean of the eight corners is the box center.
            box_center = box_query.mean(2)
            box_prompt = self.box_prompt_projector(box_prompt)
            prompt_used = box_qmask if box_qmask is not None else torch.ones_like(matched, dtype=torch.float)
            box_prompt, box_qmask = self.expand_prompt_representation(box_prompt, box_qmask)
            visual_prompt.append(box_prompt)
            visual_mask.append(box_qmask)
            visual_xyz.append(box_center.repeat_interleave(self.visual_nquery, dim=1))
            if self.part_tokens:
                parts, part_mask, part_xyz = self.part_prompt(detector_output, box_query, prompt_used,
                                                              point_clouds)
                visual_prompt.append(parts)
                visual_mask.append(part_mask)
                visual_xyz.append(part_xyz)
            if self.context_objects:
                context, context_mask, context_xyz = self.context_prompt(detector_output, matched, prompt_used)
                visual_prompt.append(context)
                visual_mask.append(context_mask)
                visual_xyz.append(context_xyz)

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
            visual_xyz.append(click_xyz.repeat_interleave(self.visual_nquery, dim=1))

        ## concat box and click prompts as well as prompt masks
        prompt_feature = torch.cat(visual_prompt, dim=1)  # batch x (2 x ntoken) x channel
        prompt_mask = torch.cat(visual_mask, dim=1)  # batch x (2 x ntoken)
        prompt_xyz = torch.cat(visual_xyz, dim=1)  # batch x (2 x ntoken) x 3

        return prompt_feature, prompt_mask, prompt_xyz


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
            torch_dtype=self.dtype)
        self.n_embd = self.transformer.config.hidden_size

        ## Multi-modality Transformer
        qformer_config = InstructBlipQFormerConfig(
            num_hidden_layers=6,
            encoder_hidden_size=self.encoder_hidden_size
        )
        self.qformer = InstructBlipQFormerModel.from_pretrained(
            args.qformer_vocab,
            config=qformer_config,
        )
        self.qformer_hidden_size = qformer_config.hidden_size
        if getattr(args, 'qformer_rope', False):
            self.rotary = Rotary3D(qformer_config.hidden_size // qformer_config.num_attention_heads)
            add_cross_attention_rotary(self.qformer)

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
            self.n_embd,
            context_objects=getattr(args, 'context_objects', 0),
            part_tokens=getattr(args, 'box_part_tokens', False),
            part_colors=getattr(args, 'use_color', False),
        )
        self.min_box_prompt_iou = getattr(args, 'min_box_prompt_iou', 0.0)
        self.latent_query = nn.Embedding(self.nlatent_query, self.qformer_hidden_size)
        self.qformer_to_language_projection = nn.Linear(self.qformer_hidden_size, self.n_embd)

        self.en_slic = LearnCut(self.encoder_hidden_size, self.encoder_hidden_size,
                                args.n_neighs, n_clus=args.n_clus, n_split=args.n_splits, radius=1.5, tau=1e-1,
                                fusion_method=getattr(args, 'fusion_method', 'baseline'))

        self.max_gen_per_iter = 8
        # ---- super parameters for evaluation
        self.caption_config = {
            'early_stopping': True,
            'eos_token_id': self.tokenizer.eos_token_id,
            'num_beams': 4 if args.use_beam_search is True else None,
        }
        self.train()

    def loss_caption(self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Masked token-level cross-entropy for captioning.

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
        return final_loss

    def _prepare_scene_features(self, detector_output):
        """Merge local and global features once per scene; reused across object proposals."""
        merge_features, l_loss, g_loss = self.en_slic(
            detector_output['enc_features'], detector_output['enc_xyz'],
            detector_output['l_enc_feats'], detector_output['l_enc_xyz'],
            detector_output['enc_spts'], detector_output['l_enc_spts'],
            compute_auxiliary_losses=self.training,
        )
        encoder_hidden_states = self.encoder_to_qformer_projection(merge_features)
        return encoder_hidden_states, l_loss, g_loss

    def _get_instruction_response(self,
                                  detector_output: dict,
                                  inputs: dict,
                                  box_query: Tensor = None,
                                  box_qmask: Tensor = None,
                                  click_query: Tensor = None,
                                  click_qmask: Tensor = None,
                                  scene_features=None,
                                  ):

        point_cloud_dims = [
            inputs["point_cloud_dims_min"],
            inputs["point_cloud_dims_max"],
        ]
        net_device = inputs["point_clouds"].device
        batch_size = inputs["point_clouds"].shape[0]
        if scene_features is None:
            scene_features = self._prepare_scene_features(detector_output)
        encoder_hidden_states, l_loss, g_loss = scene_features

        # prompt encoding
        prompt_feature, prompt_mask, prompt_xyz = self.prompt_encoder(
            detector_output,
            point_cloud_dims,
            box_query=box_query,
            box_qmask=box_qmask,
            click_query=click_query,
            click_qmask=click_qmask,
            point_clouds=inputs['point_clouds'],
        )
        # gather query feature for qformer: batch x (n_query + n_tokens) x n_embd
        query_tokens = self.latent_query.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        query_tokens = torch.cat((query_tokens, prompt_feature), dim=1)
        query_attention_mask = torch.cat((torch.ones(batch_size, self.nlatent_query).to(net_device),
                                          prompt_mask), dim=1)

        # prepare qformer inputs: batch x ntoken x n_embd
        query_attention_mask = torch.cat((query_attention_mask, inputs['qformer_attention_mask']), dim=1)

        rotary = contextlib.nullcontext()
        if hasattr(self, 'rotary'):
            # Meters from the center of the scene bounds. Point-cloud tokens carry
            # positions: prompt tokens at their box center, part centroid, context
            # object or click, and scene keys at their encoder points. Latent queries
            # have no position and attend to the keys by content only.
            center = (point_cloud_dims[0] + point_cloud_dims[1])[:, None] / 2
            latent_angles = prompt_xyz.new_zeros(batch_size, self.nlatent_query, self.rotary.frequency.shape[0])
            query_angles = torch.cat((latent_angles, self.rotary(prompt_xyz - center)), dim=1)
            key_angles = self.rotary(detector_output['enc_xyz'] - center)
            point_cloud_rows = torch.cat((
                torch.zeros(batch_size, self.nlatent_query, dtype=torch.bool, device=net_device),
                torch.ones(prompt_xyz.shape[:2], dtype=torch.bool, device=net_device)), dim=1)
            rotary = cross_attention_angles(self.qformer, query_angles, key_angles, point_cloud_rows)
        with rotary:
            query_outputs = self.qformer(
                input_ids=inputs['qformer_input_ids'],
                attention_mask=query_attention_mask,
                query_embeds=query_tokens,
                encoder_hidden_states=encoder_hidden_states,
                use_cache=False,
                return_dict=True,
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

        min_iou = getattr(self, 'min_box_prompt_iou', 0.0)
        if min_iou > 0:
            # A box prompt whose best foreground proposal misses the object passes a feature
            # of a fragment or another object, while the target is this object's caption.
            # Evaluation only scores matched proposals, so such samples are label noise.
            missed = torch.zeros_like(gradient_mask[:, 0], dtype=torch.bool)
            if box_query is not None:
                sem_cls_logits = detector_output['sem_cls_logits']
                foreground = (sem_cls_logits.argmax(-1) != sem_cls_logits.shape[-1] - 1).float()
                _, matched_iou = match_proposals(detector_output['box_corners'], foreground, box_query)
                used = box_qmask.bool() if box_qmask is not None else torch.ones_like(matched_iou, dtype=torch.bool)
                missed = (used & matched_iou.lt(min_iou)).any(-1)
            gradient_mask = gradient_mask * (~missed).to(gradient_mask.dtype)[:, None]
            detector_output['missed_box_prompts'] = missed.float().mean()

        # ---- batch x ntoken x n_embd
        prefix_tokens, l_loss, g_loss = self._get_instruction_response(
            detector_output=detector_output,
            inputs=inputs,
            box_query=box_query,
            box_qmask=box_qmask,
            click_query=click_query,
            click_qmask=click_qmask,
        )
        prefix_mask = torch.ones_like(prefix_tokens[..., 0])
        # ---- batch x (ntoken + nword) x n_embd
        inputs_embeds = torch.cat((prefix_tokens, embedding_layer(input_ids)), dim=1)
        attention_mask = torch.cat((prefix_mask, input_mask), dim=1)

        # ---- calculate transformer loss
        outputs = self.transformer(
            inputs_embeds=inputs_embeds.to(self.dtype),
            attention_mask=attention_mask.to(self.dtype),
            use_cache=False,
        )

        detector_output['loss'] = detector_output['loss'] + self.loss_caption(
            logits=outputs.logits[:, prefix_tokens.shape[1] - 1: -1],
            target=input_ids,
            mask=gradient_mask.to(self.dtype),
        )
        detector_output['l_loss'] = l_loss
        detector_output['g_loss'] = g_loss
        return detector_output

    def predict_densecap(self, detector_output: Dict, inputs: Dict, task_name: str, max_gen_length: int = 64) -> Dict:
        # ---- necessary elements
        embedding_layer = self.transformer.get_input_embeddings()
        net_device = next(self.parameters()).device
        batch_size, nproposals, _, _ = detector_output['box_corners'].shape
        # ---- to store llm outputs
        output_ids = torch.full(
            (batch_size, nproposals, max_gen_length), self.tokenizer.eos_token_id,
            dtype=torch.long, device=net_device,
        )
        sem_cls_logits = detector_output["sem_cls_logits"]
        objectness_mask = sem_cls_logits.argmax(-1) != (sem_cls_logits.shape[-1] - 1)
        if not objectness_mask.any():
            detector_output['output_ids'] = output_ids
            detector_output['l_loss'] = sem_cls_logits.new_zeros(())
            detector_output['g_loss'] = sem_cls_logits.new_zeros(())
            return detector_output
        # ---- llm input preparation
        instruction = inputs['instruction'][0]  # ntoken
        instruction_mask = inputs['instruction_mask'][0]  # ntoken
        instruction_id = instruction[instruction_mask == 1]  # ntoken
        instruction_id = instruction_id[None, :].repeat(batch_size, 1)
        instruction_embedding = embedding_layer(instruction_id)  # batch x ntoken x n_embd

        prefix_tokens = []
        l_loss = 0.0
        g_loss = 0.0
        scene_features = self._prepare_scene_features(detector_output)
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
                scene_features=scene_features,
            )
            l_loss += l_loss_i
            g_loss += g_loss_i
            instruct_prefix_feature = torch.cat((instruct_prefix_feature, instruction_embedding), dim=1)
            prefix_tokens.append(instruct_prefix_feature.unsqueeze(1))
        # batch x nproposal x 1 x n_embd
        prefix_tokens = torch.cat(prefix_tokens, dim=1).to(self.dtype)

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
