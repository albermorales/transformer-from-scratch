import torch 
from src.utils.util import get_available_device
from src.test.utils import get_example_batch

def test_VectorizedMultiHeadAttention_causalMask(model_dim: int = 512, 
                                               query_dim: int=128, 
                                               num_heads: int=8, 
                                               num_tokens: int = 100, 
                                               batch: int = 1, dtype: 
                                               torch.dtype=torch.float32, 
                                               device: torch.device = torch.device('cpu')):


    from src.models.Base import VectorizedMultiHeadAttention

    attentio_mechanism = VectorizedMultiHeadAttention(model_dim=model_dim, 
                                                      query_dim=query_dim,
                                                      num_heads=num_heads, 
                                                      dtype=dtype, 
                                                      device= device, 
                                                      causal_mask=True)


    x = torch.rand([batch,num_tokens, model_dim], dtype=dtype, device=device)

    attention = attentio_mechanism(x)

    x[0,-1] = torch.rand_like(x[0,-1])

    attention_hat = attentio_mechanism(x)

    assert torch.allclose(attention[0,:-1],attention_hat[0, :-1]), "Error en el comportamiento de la mascara causal"
    assert not torch.allclose(attention[0,-1],attention_hat[0, -1]), "Error en el comportamiento de la mascara causal"


def test_VectorizedMultiHeadAttention_paddingMask(model_dim: int = 512, 
                                                  query_dim: int=128, 
                                                  num_heads: int=8, 
                                                  num_tokens: int = 100, 
                                                  batch: int = 1, dtype: 
                                                  torch.dtype=torch.float32, 
                                                  device: torch.device = torch.device('cpu')):

    from src.models.Base import VectorizedMultiHeadAttention

    
    attentio_mechanism = VectorizedMultiHeadAttention(model_dim=model_dim, 
                                                      query_dim=query_dim,
                                                      num_heads=num_heads, 
                                                      dtype=dtype, 
                                                      device= device, 
                                                      causal_mask=False)


    x = torch.rand([batch,num_tokens, model_dim], dtype=dtype, device=device)

    # random padding mask
    padding_mask = torch.rand([batch,num_tokens], dtype=dtype, device=device)
    padding_mask = padding_mask>torch.quantile(padding_mask, q=0.2)

    x_hat = x.clone()
    x_hat[~padding_mask] = torch.rand_like(x)[~padding_mask]
    
    # Result expected to be the same as same padding mask is provided
    attention     = attentio_mechanism(x,     affinity_mask=padding_mask)
    attention_hat = attentio_mechanism(x_hat, affinity_mask=padding_mask)
    assert torch.allclose(attention[padding_mask],attention_hat[padding_mask]), "Error en el comportamiento de la mascara de padding"

    # Result expected to change since padding mask is not provided
    attention     = attentio_mechanism(x)
    attention_hat = attentio_mechanism(x_hat)
    assert not torch.allclose(attention[padding_mask],attention_hat[padding_mask]), "Error en el comportamiento de la mascara de padding"


def test_Transformer_output_shape(model_dim: int = 512, 
                                  query_dim: int=128, 
                                  num_heads: int=8, 
                                  num_tokens: int = 100, 
                                  num_blocks: int= 4,
                                  batch: int = 1, 
                                  dtype: torch.dtype=torch.float32, 
                                  device: torch.device = torch.device('cpu')):

    from src.models.Base import Transformer

    model = Transformer(model_dim=model_dim, 
                        num_heads=num_heads,
                        num_blocks=num_blocks,
                        query_dim=query_dim, 
                        dtype=dtype, 
                        device=device)

    enc_input = torch.randint(size=[batch, num_tokens-1], low=0, high=model.vocab_size)
    dec_input = torch.randint(size=[batch, num_tokens+1], low=0, high=model.vocab_size)

    output_shape = model(enc_input, dec_input).shape
    assert output_shape == torch.Size([batch, num_tokens+1, model.vocab_size])

    enc_output = model.encoder(enc_input)
    assert enc_output.shape == torch.Size([batch, num_tokens-1, model_dim])

    output_shape = model.decoder(enc_output, dec_input).shape
    assert output_shape == torch.Size([batch, num_tokens+1, model_dim])


def test_Transformer_loss_and_gradients(model_dim: int = 512, 
                                        query_dim: int=128, 
                                        num_heads: int=8, 
                                        num_tokens: int = 100, 
                                        num_blocks: int= 4,
                                        batch: int = 1, 
                                        dtype: torch.dtype=torch.float32, 
                                        device: torch.device = torch.device('cpu')):

    from src.models.Base import Transformer

    model = Transformer(model_dim=model_dim, 
                        num_heads=num_heads,
                        num_blocks=num_blocks,
                        query_dim=query_dim, 
                        dtype=dtype, 
                        device=device)

    enc_input = torch.randint(size=[batch, num_tokens-1], low=0, high=model.vocab_size)
    dec_input = torch.randint(size=[batch, num_tokens+1], low=0, high=model.vocab_size)
    target    = torch.randint(size=[batch, num_tokens+1], low=0, high=model.vocab_size)


    # artificially include 0's
    target = target.masked_fill(target<target.to(torch.float32).quantile(0.1), 0)
    
    model_output = model(enc_input, dec_input)

    loss_function = torch.nn.CrossEntropyLoss(label_smoothing= 0.1, ignore_index=0)
    
    loss = loss_function(model_output.reshape(-1, model.vocab_size), target.reshape(-1))
   
    assert torch.isfinite(loss).all(), "Error: infinite loss."

    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Error: Gradiente nulo en {name}"
        assert torch.isfinite(param.grad).all(), f"Error: Gradiente infinito en {name}"
        assert (param.grad!=0.).any(), f"Error: Gradiente nulo en {name}"

    assert (target==0).any()
    assert (target!=0).any()
    assert loss.ndim == 0



def test_Transformer_overfitting(model_dim: int = 512, 
                                 query_dim: int=128, 
                                 num_heads: int=8, 
                                 num_tokens: int = 100, 
                                 num_blocks: int= 4,
                                 batch_size: int = 5, 
                                 dtype: torch.dtype=torch.float32):

    from src.models.Base import Transformer
    from tqdm import tqdm 

    device = get_available_device()

    model = Transformer(model_dim=model_dim, 
                        num_heads=num_heads,
                        num_blocks=num_blocks,
                        query_dim=query_dim,
                        max_length=500, 
                        dtype=dtype, 
                        device=device, 
                        dropout=0.0)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas= [0.9, 0.98], eps= 1.0e-9, weight_decay= 0.0)
    loss_function = torch.nn.CrossEntropyLoss(ignore_index=0)

    batch = get_example_batch(model, device, dtype)

    # loss_record = []

    for _ in tqdm(range(100)):

        optimizer.zero_grad()

        model_output = model(batch['en']['token_ids'], 
                             batch['de']['token_ids'], 
                             batch['en']['padding_mask'], 
                             batch['de']['padding_mask'])

        loss = loss_function(model_output.reshape(-1, model.vocab_size), batch['target']['token_ids'].reshape(-1))

        # loss_record.append(loss.item())
        
        loss.backward()
        optimizer.step()

    model.eval()
    
    model_output = model(batch['en']['token_ids'], 
                            batch['de']['token_ids'], 
                            batch['en']['padding_mask'], 
                            batch['de']['padding_mask'])
    
    predictions = model_output.softmax(axis=-1)
    predictions = torch.argmax(predictions, axis=-1)
    
    assert (predictions == batch['target']['token_ids'])[batch['target']['padding_mask']].float().mean() > 0.99, "ERROR: Model does not overfit, at least, with the given hyperparameter selection."


    with torch.no_grad():
        generation = model.generate(batch['en']['token_ids'], batch['en']['padding_mask'])

    sentences = generation['token_ids']
    padding_mask = generation['padding_mask']
    assert sentences.ndim == 2
    assert sentences.shape == padding_mask.shape
    assert sentences.shape[0] == batch['target']['token_ids'].shape[0]
    for index, sentence in enumerate(sentences):
        generated = sentence[padding_mask[index]]
        reference = batch['target']['token_ids'][index][batch['target']['padding_mask'][index]].to(torch.long)
        assert torch.equal(generated, reference), f"ERROR: Generated sequence {index} does not match its training reference."





