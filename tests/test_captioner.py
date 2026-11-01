import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None
if HAS_TRANSFORMERS:
    from transformers import InstructBlipQFormerConfig, InstructBlipQFormerModel, OPTConfig, OPTForCausalLM
    from models.perla import captioner as captioner_module
    from models.perla.captioner import Captioner, PromptEncoder
    from models.perla.generation_utils import generation
    from models.perla.merger import LearnCut


def make_captioner_fixture(nproposals=4):
    """A small real model with random weights; no downloads or CUDA required."""
    torch.manual_seed(7)
    model = Captioner.__new__(Captioner)
    torch.nn.Module.__init__(model)
    model.freeze_llm = True
    model.nlatent_query = 2
    model.dtype = torch.float32
    model.tokenizer = SimpleNamespace(eos_token_id=2)
    model.max_gen_per_iter = 3
    model.caption_config = dict(eos_token_id=2, num_beams=None)
    model.reference_point = {}
    model.transformer = OPTForCausalLM(OPTConfig(
        vocab_size=24, hidden_size=16, word_embed_proj_dim=16, ffn_dim=32,
        num_hidden_layers=1, num_attention_heads=2, dropout=0.0,
    ))
    model.qformer = InstructBlipQFormerModel(InstructBlipQFormerConfig(
        vocab_size=24, hidden_size=16, encoder_hidden_size=16,
        intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
        hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
    ))
    model.en_slic = LearnCut(16, 16, n_neighs=4, n_clus=4, n_split=2)
    model.encoder_to_qformer_projection = torch.nn.Linear(16, 16)
    model.prompt_encoder = PromptEncoder(16, 2, 16, 16)
    model.latent_query = torch.nn.Embedding(2, 16)
    model.qformer_to_language_projection = torch.nn.Linear(16, 16)
    model.eval()

    centers = torch.rand(2, nproposals, 3)
    corners = torch.tensor([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    detector = dict(
        enc_features=torch.rand(2, 16, 16), enc_xyz=torch.rand(2, 16, 3),
        l_enc_feats=torch.rand(2, 32, 16), l_enc_xyz=torch.rand(2, 32, 3),
        enc_spts=torch.zeros(2, 16, dtype=torch.long),
        l_enc_spts=torch.zeros(2, 32, dtype=torch.long),
        box_corners=centers.unsqueeze(2) + corners * 0.05,
        prop_features=torch.rand(1, 2, nproposals, 16), query_xyz=centers,
        sem_cls_logits=torch.rand(2, nproposals, 3), loss=torch.tensor(0.0),
    )
    detector["sem_cls_logits"][..., 0] += 2
    if nproposals > 1:
        detector["sem_cls_logits"][0, 1, -1] += 4
    inputs = dict(
        point_clouds=torch.rand(2, 32, 3), point_cloud_dims_min=torch.zeros(2, 3),
        point_cloud_dims_max=torch.ones(2, 3),
        qformer_input_ids=torch.tensor([[1, 3], [1, 4]]),
        qformer_attention_mask=torch.ones(2, 2),
        instruction=torch.tensor([[1, 4, 5, 0], [1, 4, 5, 0]]),
        instruction_mask=torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]]),
        input_ids=torch.tensor([[1, 4, 5], [1, 3, 6]]),
        attention_mask=torch.ones(2, 3), gradient_mask=torch.ones(2, 3),
    )
    return model, detector, inputs


def uncached_densecap(model, detector, inputs, task_name, max_length=4):
    """Reference: independently encode the scene for each proposal."""
    instruction = inputs["instruction"][0][inputs["instruction_mask"][0] == 1]
    instruction = model.transformer.get_input_embeddings()(instruction).unsqueeze(0).expand(2, -1, -1)
    prefixes, local_losses, global_losses = [], [], []
    for proposal in range(detector["box_corners"].shape[1]):
        prefix, local, global_ = model._get_instruction_response(
            detector, inputs, box_query=detector["box_corners"][:, [proposal]],
            click_query=detector["query_xyz"][:, [proposal]] if task_name == "ov-det" else None,
        )
        prefixes.append(torch.cat((prefix, instruction), dim=1))
        local_losses.append(local)
        global_losses.append(global_)
    mask = detector["sem_cls_logits"].argmax(-1) != 2
    selected = torch.stack(prefixes, dim=1)[mask]
    tokens = torch.cat([
        generation(model.transformer, inputs_embeds=chunk, max_length=max_length,
                   **model.caption_config)["output_ids"]
        for chunk in selected.split(model.max_gen_per_iter)
    ])
    outputs = torch.full((*mask.shape, max_length), model.tokenizer.eos_token_id, dtype=torch.long)
    outputs[mask] = tokens
    return outputs, sum(local_losses), sum(global_losses)


@unittest.skipUnless(HAS_TRANSFORMERS, "Transformers is required for captioner tests")
class CaptionerRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_reusing_scene_features_preserves_captions_and_losses(self):
        model, detector, inputs = make_captioner_fixture()
        for task in ("dense-cap", "ov-det"):
            with self.subTest(task=task), torch.no_grad():
                expected_ids, expected_local, expected_global = uncached_densecap(model, detector, inputs, task)
                with patch.object(model.en_slic, "forward", wraps=model.en_slic.forward) as merge, \
                        patch.object(model.encoder_to_qformer_projection, "forward",
                                     wraps=model.encoder_to_qformer_projection.forward) as project:
                    output = model.predict_densecap(dict(detector), inputs, task, max_gen_length=4)
                merge.assert_called_once()
                project.assert_called_once()
                torch.testing.assert_close(output["output_ids"], expected_ids)
                torch.testing.assert_close(output["l_loss"], expected_local)
                torch.testing.assert_close(output["g_loss"], expected_global)

    def test_background_only_batch_skips_caption_computation(self):
        model, detector, inputs = make_captioner_fixture()
        detector["sem_cls_logits"][..., -1] = 10
        with patch.object(model.en_slic, "forward") as merge, \
                patch.object(captioner_module, "generation") as generate:
            output = model.predict_densecap(detector, inputs, "dense-cap", max_gen_length=4)
        merge.assert_not_called()
        generate.assert_not_called()
        torch.testing.assert_close(output["output_ids"], torch.full((2, 4, 4), 2, dtype=torch.long))

    def test_training_backward_without_unused_language_cache(self):
        model, detector, inputs = make_captioner_fixture()
        model.train()
        with patch.object(model.transformer, "forward", wraps=model.transformer.forward) as forward:
            output = model.forward_training(detector, inputs)
        self.assertFalse(forward.call_args.kwargs["use_cache"])
        (output["loss"] + output["l_loss"] + output["g_loss"]).backward()
        self.assertTrue(torch.isfinite(model.latent_query.weight.grad).all())
        self.assertEqual(model.reference_point, {})

    def test_regularizer_allocates_snapshots_only_when_enabled(self):
        model, _, _ = make_captioner_fixture()
        logits = torch.randn(2, 3, 24, requires_grad=True)
        target = torch.ones(2, 3, dtype=torch.long)
        mask = torch.ones(2, 3)
        plain = model.loss_caption(logits, target, mask)
        self.assertEqual(model.reference_point, {})
        regularized = model.loss_caption(logits, target, mask, reg=0.5)
        self.assertTrue(model.reference_point)
        torch.testing.assert_close(plain, regularized)
        with torch.no_grad():
            model.latent_query.weight.add_(1)
        self.assertGreater(model.loss_caption(logits, target, mask, reg=0.5).item(), plain.item())


if __name__ == "__main__":
    unittest.main()
