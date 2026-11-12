import os, yaml, argparse, re, logging, torch
from src.utils.util import resolve_callables_in_mapping, get_available_device
from datetime import datetime
from contextvars import ContextVar
from tqdm import tqdm
import onnx
from functools import partial
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    _self = ContextVar("current_trainer")

    @staticmethod
    def current():
        "Returns running trainer in the current context"
        return Trainer._self.get()

    def __init__(self, config_path: str="config.yaml"):

        # ------------------ Resolve configutarions y attributes ------------------
        self.logger = logging.getLogger()   

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        config = resolve_callables_in_mapping(config)

        training_attr = config.pop("training")

        for key, value in training_attr.items():
            setattr(self, key, value)

        self.cfg = config        

        Trainer._self.set(self)

        
        # ------------------ Logging y Salidas ------------------

        assert os.path.isdir(self.output_path), f"Output Path {self.output_path} doesn't exist"

        self.output_path = os.path.join(self.output_path, self.name)
    
        if os.path.isdir(self.output_path):

            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

            log_mesage = f"El directorio de salida {self.output_path} ya existe, usaremos {self.output_path}_{timestamp} en su lugar. Continuamos sobre-escribiendo"
            self.output_path += f'_{timestamp}'

            log_level = logging.WARNING 
            
        else:
            os.mkdir(self.output_path)
            log_mesage = f"El directorio de salida creado correctamente:\n\t{self.output_path}"
            log_level = logging.INFO

        os.makedirs(os.path.join(self.output_path, "checkpoints"), exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.output_path)

        logfile = os.path.join(self.output_path, "training.log")

        console_handler = logging.StreamHandler()
        console_handler.addFilter(
            lambda record: getattr(record, "console", False)
        )

        logging.basicConfig(level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.FileHandler(logfile, encoding="utf-8"),
                console_handler,
            ])

        self.logger.log(log_level, log_mesage)

        # ------------------ Device Selection ------------------
        
        if hasattr(self, 'hardware_acceleration') and self.hardware_acceleration:
            self.device = get_available_device()

            if self.device == torch.device('cpu'):
                self.logger.warning(f"No es posible usar aceleracion por hardware. Usando el device disponible: {self.device}")

            else: 
                self.logger.info(f"Usando el backend: {self.device}")

        else:
            self.device = torch.device('cpu')
            self.logger.info(f"Usando el device disponible: {self.device}")


        # ------------------ Dataset & Dataloader ------------------

        DatasetClass = self.cfg['dataset'].get('class', torch.utils.data.Dataset)
        self.logger.info(f"Clase usada para Dataset: {DatasetClass}")

        self.train_ds = DatasetClass(**self.cfg['dataset']['train'])
        self.valid_ds = DatasetClass(**self.cfg['dataset']['valid'])

        BatchSamplerClass = self.cfg['batch_sampler']['class']
        self.logger.info(f"Clase usada para BatchSampler: {BatchSamplerClass}")

        self.train_batch_sampler = BatchSamplerClass(self.train_ds, **self.cfg['batch_sampler']['train'])
        self.valid_batch_sampler = BatchSamplerClass(self.valid_ds,**self.cfg['batch_sampler']['valid'])

        CollateFunction = self.cfg['collate_fn']['function']
        self.logger.info(f"Función usada para collate: {CollateFunction}")
        train_collate_fn = partial(CollateFunction, pad_token_id=self.train_ds.tokenizer.pad_token_id)
        valid_collate_fn = partial(CollateFunction, pad_token_id=self.valid_ds.tokenizer.pad_token_id)

        DataloaderClass = self.cfg['dataloader'].get('class', torch.utils.data.DataLoader)
        self.logger.info(f"Clase usada para Dataloader: {DataloaderClass}")

        self.train_dl = DataloaderClass(
            self.train_ds,
            batch_sampler=self.train_batch_sampler,
            collate_fn=train_collate_fn,
            **self.cfg['dataloader']['train'],
        )
        self.valid_dl = DataloaderClass(
            self.valid_ds,
            batch_sampler=self.valid_batch_sampler,
            collate_fn=valid_collate_fn,
            **self.cfg['dataloader']['valid'],
        )

        if hasattr(self, 'total_training_steps'):
            if hasattr(self, 'epochs'):
                self.logger.warning("Conflicto de atributos: total_training_steps y epoch son incompatibles." \
                "Se usará total_training_steps. Epochs será ignorado.")

            self.epochs = self.total_training_steps // len(self.train_dl)

        else: 
            if hasattr(self, 'epochs'):
                self.total_training_steps = len(self.train_dl) * self.epochs
            else: 
                self.logger.error('Either total_training_steps or epochs attributes must be provided to configure a training')
                raise ValueError("Not total_training_steps neither epochs were be provided.")

        # ------------------ Modelo y LossFunction ------------------

        Model = self.cfg['model']['class']
        model_args = dict(self.cfg['model']['args'])
        model_args['tokenizer'] = self.train_ds.tokenizer
        model_args['device'] = self.device
        self.model = Model(**model_args).to(self.device)
        self.logger.info(f"Clase usada para Model: {Model}")

        Loss = self.cfg['loss']['class']
        self.loss_function = Loss(**self.cfg['loss']['args'])
        self.logger.info(f"Clase usada para Loss: {Loss}")


        # ------------------ Optimizador y Scheduler ------------------

        OptimizerClass = self.cfg['optimizer'].get('class', torch.optim.Adam)
        self.optimizer = OptimizerClass(self.model.parameters(), **self.cfg['optimizer']['args'])
        self.logger.info(f"Clase usada para Optimizer: {OptimizerClass}")

        SchedulerClass = self.cfg['scheduler'].get('class', torch.optim.lr_scheduler.ReduceLROnPlateau)
        self.scheduler = SchedulerClass(self.optimizer, **self.cfg['scheduler']['args'])
        self.logger.info(f"Clase usada para Scheduler: {SchedulerClass}")    

        # ------------------ Evaluation Metrics ------------------

        EvaluatorClass = self.cfg['evaluator'].get('class')
        self.epoch_evaluation = EvaluatorClass(**self.cfg['evaluator']['args'])
        self.logger.info(f"Clase usada para Evaluator: {EvaluatorClass}")

        # ------------------ Save for Reproducibility ------------------

        self._dump(os.path.join(self.output_path, 'resolved_config.yaml'))



    def _dump(self, config_output_path):
       config = self.cfg.copy()
       
       config['training'] = {}
       pattern = re.compile(r"^__.+__$") 

       for attr, value in self.__dict__.items():
           if not pattern.match(attr) and (isinstance(value, str) or isinstance(value, float) or isinstance(value, int)) and attr!='cfg':
               config['training'][attr] = value

       with open(config_output_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)


    def _save_checkpoint(self, checkpoint_path):
        torch.save({
            "epoch": self.current_epoch,
            "train_step": self.current_train_step,
            "valid_step": self.current_valid_step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
        }, checkpoint_path)
        self.logger.info(f"Checkpoint guardado -> {checkpoint_path}")


    def train_model(self):
        self.current_train_step = 0
        self.current_valid_step = 0


        for self.current_epoch in range(self.epochs):

            epoch_train_loss = self.compute_trainig_loop()
            epoch_valid_loss = self.compute_validation_loop()
            self.epoch_evaluation(self)

            # Normalizar segun tamaño del dataset
            epoch_train_loss /= len(self.train_dl.dataset)
            epoch_valid_loss /= len(self.valid_dl.dataset)

            self.logger.info(
                f"[{self.current_epoch:03d}/{self.epochs}] "
                f"train loss={epoch_train_loss:.6f} | valid loss={epoch_valid_loss:.6f}",
                extra={"console": True},
            )

            # Save checkpoints
            if (self.current_epoch + 1) % self.checkpoint_freq == 0:
                checkpoint_path = os.path.join(
                    self.output_path,
                    'checkpoints',
                    f'epoch_{self.current_epoch:03d}.pt',
                )
                self._save_checkpoint(checkpoint_path)

        self.save_and_close_training()

        return



    def compute_trainig_loop(self):
        self.train()
        epoch_loss = 0.0

        for batch in tqdm(self.train_dl, desc=f"Epoch {self.current_epoch} - Train"):
            
            self.optimizer.zero_grad()

            loss, record = self.compute_batch_inference(batch, self.current_train_step)

            for key in record:
                self.writer.add_scalar(f'train/{key}', record[key], self.current_train_step)
            self.writer.add_scalar(f'train/Learning Rate', self.scheduler.get_last_lr()[0], self.current_train_step)

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.current_train_step +=1 
            
            epoch_loss += loss.item() * len(batch['target']['token_ids'])

        return epoch_loss


    def compute_validation_loop(self):
        self.eval()
        valid_loss = 0.0

        with torch.no_grad():

            for batch in tqdm(self.valid_dl, desc=f"Epoch {self.current_epoch} - Valid"):
                
                loss, record = self.compute_batch_inference(batch, self.current_valid_step)

                for key in record:
                    self.writer.add_scalar(f'eval/{key}', record[key], self.current_valid_step)

                valid_loss += loss.item() * len(batch['target']['token_ids'])
                self.current_valid_step += 1

            return valid_loss


    def compute_batch_inference(self, batch, current_step):

        enc_input = batch['en']['token_ids'].to(self.device)
        dec_input = batch['de']['token_ids'].to(self.device)
        target    = batch['target']['token_ids'].to(self.device)

        model_output = self.model(enc_input, dec_input, 
                                  batch['en']['padding_mask'], 
                                  batch['de']['padding_mask'])

        loss = self.loss_function(model_output.reshape(-1, self.model.vocab_size), target.reshape(-1))

        # prediction = {
        #     "logits": model_output,
        #     "padding_mask": batch["target"]["padding_mask"],
        # }
        # evaluation_target = {
        #     "token_ids": target,
        #     "padding_mask": batch["target"]["padding_mask"],
        # }
        # self.evaluator(prediction, evaluation_target, self.mode, current_step, batch)

        return loss, {'epoch_loss': loss.item()}


    def save_and_close_training(self):

        final_checkpoint_path = os.path.join(
            self.output_path,
            'checkpoints',
            'final.pt',
        )
        self._save_checkpoint(final_checkpoint_path)
        self.writer.flush()

        try:
            if not hasattr(self, 'onnx_path'):
                self.onnx_path = os.path.join(self.output_path, f'{self.name}.onnx')
                self.model.export_onnx(self.onnx_path)

            model_onnx = onnx.load(self.onnx_path)
            onnx.checker.check_model(model_onnx)
            self.logger.info(f"Modelo ONNX exportado y verificado ✅ -> {self.onnx_path}")
            
        except Exception as e:
            self.logger.warning(f"Fallo en la exportacion o verificacion ONNX: {e}")
        finally:
            self.writer.close()

        self.logger.info('Mischief Managed.')

    
    def eval(self):
        self.model.eval()
        self.mode = 'eval'

    def train(self):
        self.model.train()
        self.mode = 'train'



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/base.yaml")
    args = parser.parse_args()

    trainer = Trainer(args.config)

    trainer.train_model()
