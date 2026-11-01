import importlib.util
import datetime
import os
import tempfile
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


def make_captioner_fixture(nproposals=4, fusion_method="baseline", neighbor_selection="hilbert"):
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
    model.en_slic = LearnCut(16, 16, n_neighs=4, n_clus=4, n_split=2,
                            fusion_method=fusion_method, neighbor_selection=neighbor_selection)
    if fusion_method == "instruction":
        model.instruction_to_scene = torch.nn.Sequential(torch.nn.LayerNorm(16), torch.nn.Linear(16, 16))
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


def run_distributed_fusion_steps(rank, rendezvous):
    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2,
        timeout=datetime.timedelta(seconds=30),
    )
    try:
        model, detector, inputs = make_captioner_fixture(fusion_method="instruction", neighbor_selection="geometry")
        model.train()
        distributed_model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        for step in range(2):
            optimizer.zero_grad()
            task_inputs = dict(inputs)
            if rank == 0 and step == 0:
                task_inputs["box_query"] = detector["box_corners"][:, [0]]
            elif rank == 1 and step == 1:
                task_inputs["click_query"] = detector["query_xyz"][:, [0]]
            output = distributed_model(dict(detector), task_inputs, is_eval=False)
            (output["loss"] + 0.1 * output["l_loss"] + 0.01 * output["g_loss"]).backward()
            optimizer.step()
        if not torch.isfinite(model.en_slic.update_gate.weight).all():
            raise AssertionError("Non-finite distributed gate update")
    finally:
        torch.distributed.destroy_process_group()


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

    def test_instruction_geometry_fusion_trains_and_preserves_cached_predictions(self):
        model, detector, inputs = make_captioner_fixture(fusion_method="instruction", neighbor_selection="geometry")
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        # The gate starts with zero weights; context projection receives its
        # first gradient after the gate learns to use context on step one.
        for _ in range(2):
            optimizer.zero_grad()
            output = model.forward_training(dict(detector), inputs)
            (output["loss"] + 0.1 * output["l_loss"] + 0.01 * output["g_loss"]).backward()
            optimizer.step()
        for parameter in (model.en_slic.update_gate.weight, model.en_slic.distance_strength,
                          model.instruction_to_scene[-1].weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        model.eval()
        with torch.no_grad():
            expected = uncached_densecap(model, detector, inputs, "dense-cap")
            with patch.object(model.en_slic, "forward", wraps=model.en_slic.forward) as merge:
                actual = model.predict_densecap(dict(detector), inputs, "dense-cap", max_gen_length=4)
            merge.assert_called_once()
        torch.testing.assert_close(actual["output_ids"], expected[0])
        torch.testing.assert_close(actual["l_loss"], expected[1])
        torch.testing.assert_close(actual["g_loss"], expected[2])

    def test_instruction_fusion_ignores_padding_and_response_tokens(self):
        model, detector, inputs = make_captioner_fixture(fusion_method="instruction")
        with torch.no_grad():
            # Activate context weights to exercise masking beyond initialization.
            model.en_slic.update_gate.weight.normal_(std=0.1)
            expected = model._prepare_scene_features(detector, inputs)[0]
            padded = dict(inputs)
            padded["qformer_input_ids"] = torch.cat([inputs["qformer_input_ids"], torch.full((2, 1), 19)], dim=1)
            padded["qformer_attention_mask"] = torch.cat([inputs["qformer_attention_mask"], torch.zeros(2, 1)], dim=1)
            padded["input_ids"] = torch.zeros_like(inputs["input_ids"])
            actual = model._prepare_scene_features(detector, padded)[0]
            changed = dict(inputs, qformer_input_ids=torch.full_like(inputs["qformer_input_ids"], 20))
            different_instruction = model._prepare_scene_features(detector, changed)[0]
        torch.testing.assert_close(actual, expected)
        self.assertGreater((different_instruction - expected).abs().max().item(), 1e-6)

    def test_inference_skips_graph_losses_without_changing_captions(self):
        for fusion, neighbors in (("baseline", "hilbert"), ("instruction", "geometry")):
            with self.subTest(fusion=fusion), torch.no_grad():
                model, detector, inputs = make_captioner_fixture(fusion_method=fusion, neighbor_selection=neighbors)
                original_forward = model.en_slic.forward

                def with_training_losses(*args, **kwargs):
                    kwargs['compute_auxiliary_losses'] = True
                    return original_forward(*args, **kwargs)

                with patch.object(model.en_slic, 'forward', side_effect=with_training_losses):
                    expected = model.predict_densecap(dict(detector), inputs, 'dense-cap', max_gen_length=4)
                with patch.object(model.en_slic, 'min_cut', side_effect=AssertionError('graph loss ran in inference')):
                    actual = model.predict_densecap(dict(detector), inputs, 'dense-cap', max_gen_length=4)
                torch.testing.assert_close(actual['output_ids'], expected['output_ids'])
                self.assertEqual(actual['l_loss'].item(), 0)
                self.assertEqual(actual['g_loss'].item(), 0)

    @unittest.skipUnless(torch.distributed.is_available() and torch.distributed.is_gloo_available(),
                         "Gloo is required for the distributed fusion test")
    def test_two_process_instruction_fusion_backward(self):
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = "file://" + os.path.join(directory, "rendezvous")
            torch.multiprocessing.spawn(run_distributed_fusion_steps, args=(rendezvous,), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
