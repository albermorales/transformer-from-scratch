"""Small offline tests: no dataset, tokenizer download or training required."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from src.models.Base import Transformer


@pytest.mark.parametrize("stopping", ["mixed", "eos", "limit"])
def test_onnx_generation_matches_pytorch(tmp_path, stopping):
    tokenizer = SimpleNamespace(
        vocab_size=12, bos_token_id=1, eos_token_id=2, pad_token_id=0,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4)
        model = Transformer(
            device="cpu", tokenizer=tokenizer, model_dim=8, query_dim=4,
            num_heads=2, num_blocks=1, dtype=torch.float32,
            max_length=9, dropout=0.0,
        )
        mixed_source = torch.randint(3, 12, (4, 5))
    with torch.no_grad():
        if stopping == "mixed":
            model.final_poyection.bias[2] += 0.5
        else:
            model.final_poyection.weight.zero_()
            model.final_poyection.bias.fill_(-20)
            model.final_poyection.bias[2 if stopping == "eos" else 4] = 20

    model.train()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    output = tmp_path / "generation.onnx"
    model.export_onnx(str(output))

    assert model.training
    assert model.device == "cpu"
    assert next(model.parameters()).dtype == torch.float32
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name]), name

    graph = onnx.load(str(output))
    onnx.checker.check_model(graph, full_check=True)
    assert any(node.op_type == "Loop" for node in graph.graph.node)
    session = ort.InferenceSession(
        str(output), providers=["CPUExecutionProvider"],
    )
    assert [item.name for item in session.get_inputs()] == [
        "enc_input", "enc_padding_mask",
    ]
    assert [item.name for item in session.get_outputs()] == [
        "token_ids", "padding_mask",
    ]

    model.eval()
    sources = [
        mixed_source,
        torch.tensor([[3]]),
        torch.tensor([[3, 4, 5, 0], [6, 7, 8, 9]]),
        torch.full((3, 9), 5, dtype=torch.long),
    ]
    for index, source in enumerate(sources):
        mask = source != tokenizer.pad_token_id
        with torch.no_grad():
            # The ONNX graph exports greedy decoding, equivalent to beam=1.
            expected = model.generate(source, mask, beam=1)
        token_ids, padding_mask = session.run(None, {
            "enc_input": source.numpy(),
            "enc_padding_mask": mask.numpy(),
        })
        np.testing.assert_array_equal(token_ids, expected["token_ids"].numpy())
        np.testing.assert_array_equal(padding_mask, expected["padding_mask"].numpy())

        if stopping == "eos":
            assert token_ids.shape[1] == 1
            assert np.all(token_ids == tokenizer.eos_token_id)
        elif stopping == "limit":
            assert token_ids.shape[1] == model.max_length
            assert np.all(token_ids == 4)
        elif index == 0:
            # Some rows stop immediately, others run to the length limit.
            has_eos = np.any(token_ids == tokenizer.eos_token_id, axis=1)
            assert has_eos.any() and not has_eos.all()
            for row in token_ids[has_eos]:
                eos_index = np.flatnonzero(row == tokenizer.eos_token_id)[0]
                assert np.all(row[eos_index + 1:] == tokenizer.pad_token_id)
