import json
import os
from copy import deepcopy

import numpy as np
from transformers import AutoTokenizer

from datasets.scannet_base_dataset import BASE, ScanNetBaseDataset
from datasets.task_prompts import TASK_PROPMT, BOX_FORMAT
from eval_utils.evaluate_qa import evaluate


class Dataset(ScanNetBaseDataset):

    def __init__(
            self,
            args,
            dataset_config,
            split_set="train",
            num_points=40000,
            use_color=False,
            use_normal=False,
            use_multiview=False,
            use_height=False,
            augment=False,
    ):
        super().__init__(
            args,
            dataset_config,
            split_set=split_set,
            num_points=num_points,
            use_color=use_color,
            use_normal=use_normal,
            use_multiview=use_multiview,
            use_height=use_height,
            augment=augment,
            use_random_cuboid=False,
            random_cuboid_min_points=None,
        )

        self.task_name = 'scanqa'
        self.grid_size_3d = args.grid_size_3d
        self.max_prompts = args.max_prompts
        self.split = split_set
        self.dataset_config = dataset_config
        self.max_des_len = args.max_des_len
        self.eval_func = evaluate

        ## initialize tokenizer and set tokenizer's `padding token` to `eos token`
        self.tokenizer = AutoTokenizer.from_pretrained(args.vocab, add_bos_token=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'right'
        self.qtokenizer = AutoTokenizer.from_pretrained(args.qformer_vocab)
        self.qtokenizer.pad_token = self.tokenizer.eos_token
        self.qtokenizer.padding_side = 'right'

        ## load annotations
        annotation_file = os.path.join(BASE, 'data', '3D_qa_data', 'ScanQA', f'ScanQA_v1.0_{split_set}.json')
        self.annotations = json.load(open(annotation_file, 'r'))
        self._tag_dataset(self.annotations, 'qa')

        ## super configuration
        self.tokenizer_config = dict(
            max_length=self.max_des_len,
            padding='max_length',
            truncation='longest_first',
            return_tensors='np'
        )
        print(f"kept {len(self.annotations)} annotations in {len(self.scan_names)} scans...")

    def _tag_dataset(self, corpus, task_name):
        for anno in corpus:
            anno['task_name'] = task_name
        return

    def _encode_box_coords(self, annotation_mask, ret_dict):
        center_normalized = ret_dict['gt_box_centers_normalized']
        size_normalized = ret_dict['gt_box_sizes_normalized']
        box_normalized = np.hstack((center_normalized, size_normalized))  # (-1, 6)
        # <cx, cy, cz, w, h, l>
        box_normalized = box_normalized[annotation_mask == 1]
        box_normalized = (box_normalized * self.grid_size_3d).astype(np.int64)
        return ' '.join(BOX_FORMAT.format(*box) for box in box_normalized)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        scan_name = self.annotations[idx]['scene_id']
        task_name = self.annotations[idx]['task_name']

        # load question and answer
        question = self.annotations[idx]['question'].lower()

        ret_dict = self._get_scan_data(scan_name)

        prompt = deepcopy(TASK_PROPMT[task_name][0])
        boxes = ''
        prompt['instruction'] = prompt['instruction'].format(locations=boxes, question=question)

        prompt_inputs = self.tokenizer.batch_encode_plus([prompt['instruction']], **self.tokenizer_config)
        qformer_inputs = self.qtokenizer.batch_encode_plus([prompt['instruction']], **self.tokenizer_config)

        box_query = np.zeros((self.max_prompts, 8, 3))
        box_mask = np.zeros((self.max_prompts,))
        click_query = np.zeros((self.max_prompts, 3))
        click_mask = np.zeros((self.max_prompts,))

        ret_dict['box_query'] = box_query.astype(np.float32)
        ret_dict['box_mask'] = box_mask.astype(np.float32)
        ret_dict['click_query'] = click_query.astype(np.float32)
        ret_dict['click_mask'] = click_mask.astype(np.float32)

        ## the below are for QFormer and LLaMA evaluation
        ret_dict['scan_idx'] = np.array(idx).astype(np.int64)
        ret_dict['instruction'] = prompt_inputs['input_ids'][0].astype(np.int64)
        ret_dict['instruction_mask'] = prompt_inputs['attention_mask'][0].astype(np.float32)
        ret_dict['qformer_input_ids'] = qformer_inputs['input_ids'][0].astype(np.int64)
        ret_dict['qformer_attention_mask'] = qformer_inputs['attention_mask'][0].astype(np.float32)

        return ret_dict
