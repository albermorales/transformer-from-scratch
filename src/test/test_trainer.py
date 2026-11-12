import torch
from src.test.utils import get_example_batch

def test_trainer_compute_batch(dtype: torch.dtype=torch.float32):

    from src.trainer.TransformerTrainer import Trainer

    trainer = Trainer('config/base.yaml')

    batch = get_example_batch(trainer.model, trainer.device, dtype)

    trainer.train()
    trainer.current_train_step = 0

    loss, record = trainer.compute_batch_inference(batch, current_step=0)

    assert torch.isfinite(loss).all(), "Error: infinite loss."

    loss.backward()

    for name, param in trainer.model.named_parameters():
        assert param.grad is not None, f"Error: Gradiente nulo en {name}"
        assert torch.isfinite(param.grad).all(), f"Error: Gradiente infinito en {name}"
        assert (param.grad!=0.).any(), f"Error: Gradiente nulo en {name}"

    assert loss.ndim == 0



def test_trainer_overfitting(num_iterations: int=100,
                            dtype: torch.dtype=torch.float32):


    from src.trainer.TransformerTrainer import Trainer
    from itertools import repeat

    trainer = Trainer('config/base.yaml')

    batch = get_example_batch(trainer.model, trainer.device, dtype)

    trainer.train_dl = repeat(batch, num_iterations)
    trainer.epochs = 1
    trainer.current_epoch = 0
    trainer.current_train_step = 0


    trainer.train()

    trainer.compute_trainig_loop()

    trainer.eval()
    
    model_output = trainer.model(batch['en']['token_ids'], 
                                batch['de']['token_ids'], 
                                batch['en']['padding_mask'], 
                                batch['de']['padding_mask'])
    
    predictions = model_output.softmax(axis=-1)
    predictions = torch.argmax(predictions, axis=-1)
    
    assert (predictions == batch['target']['token_ids'])[batch['target']['padding_mask']].float().mean() > 0.99, "ERROR: Trainer does not overfits model, at least, with the given hyperparameter selection."

    generation = trainer.model.generate(batch['en']['token_ids'][0].unsqueeze(dim=0), batch['en']['padding_mask'][0].unsqueeze(dim=0))
    sentence = generation['token_ids'][0][generation['padding_mask'][0]]
    reference = batch['target']['token_ids'][0][batch['target']['padding_mask'][0]]
    assert torch.equal(sentence, reference), "ERROR: Trainer does not overfits model, at least, with the given hyperparameter selection."



