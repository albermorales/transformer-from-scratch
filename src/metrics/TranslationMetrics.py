import torch 
from src.metrics.MetricEvaluator import Evaluator
import sacrebleu


class TranslationEvaluator(Evaluator):
    def __init__(self, metrics):

        self.all_metrics = {
            'token_accuracy' : TokenAccuracy,
            'bleu' : Bleu
        }

        super().__init__(metrics)

   
def TokenAccuracy(trainer):
    """Compute positional token accuracy, including length mismatches."""

    correct_tokens = 0
    evaluated_tokens = 0

    for batch in trainer.valid_dl:

        prediction = trainer.model.generate(enc_input = batch['en']['token_ids'], 
                                            enc_padding_mask = batch['en']['padding_mask'], 
                                            return_logits=False)

        prediction_ids = prediction['token_ids']
        prediction_mask = prediction['padding_mask']
        target_ids = batch['target']["token_ids"].to(trainer.device)   
        target_mask = batch['target']["padding_mask"].to(trainer.device)


        max_length = max(prediction_ids.shape[1], target_ids.shape[1])
        prediction_ids = torch.nn.functional.pad(
            prediction_ids,
            (0, max_length - prediction_ids.shape[1]),
            value=-1,
        )
        target_ids = torch.nn.functional.pad(
            target_ids,
            (0, max_length - target_ids.shape[1]),
            value=-1,
        )
        prediction_mask = torch.nn.functional.pad(
            prediction_mask,
            (0, max_length - prediction_mask.shape[1]),
            value=False,
        )
        target_mask = torch.nn.functional.pad(
            target_mask,
            (0, max_length - target_mask.shape[1]),
            value=False,
        )

        evaluation_mask = prediction_mask | target_mask
        matches = prediction_ids == target_ids
        correct_tokens += (matches & evaluation_mask).sum().item()
        evaluated_tokens += evaluation_mask.sum().item()

    token_accuracy = correct_tokens / evaluated_tokens if evaluated_tokens else 0.0
    trainer.writer.add_scalar('eval/TokenAccuracy', token_accuracy, trainer.current_epoch)



def Bleu(trainer):

    counts, totals, sys_len, ref_len = [], [], [], []

    for batch in trainer.valid_dl:

        prediction = trainer.model.generate(enc_input = batch['en']['token_ids'], 
                                            enc_padding_mask = batch['en']['padding_mask'], 
                                            return_logits=False)

        target = trainer.model.tokenizer.batch_decode(batch['target']['token_ids'])
        prediction = trainer.model.tokenizer.batch_decode(prediction['token_ids'])

        batch_bleu = sacrebleu.corpus_bleu(prediction, [target])

        counts.append(batch_bleu.counts)
        totals.append(batch_bleu.totals)
        sys_len.append(batch_bleu.sys_len)
        ref_len.append(batch_bleu.ref_len)

    counts_total = [sum(column) for column in zip(*counts)]
    totals_total = [sum(column) for column in zip(*totals)]

    bleu_global = sacrebleu.metrics.BLEU.compute_bleu(
        correct=counts_total,
        total=totals_total,
        sys_len=sum(sys_len),
        ref_len=sum(ref_len),
        smooth_method="exp",
    )

    trainer.writer.add_scalar(f'{'eval'}/{'Bleu'}', bleu_global.score, trainer.current_epoch)





