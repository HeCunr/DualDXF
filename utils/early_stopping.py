# utils/early_stopping.py
import os
import torch

class EarlyStopping:
    def __init__(self, patience=20, delta=0, checkpoint_path='checkpoints/Dual_best.pth'):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.phase_best_loss = {'phase1': None, 'transition': None, 'phase2': None}
        self.early_stop = False
        self.checkpoint_path = checkpoint_path
        self.phase_checkpoint_path = {
            'phase1': checkpoint_path.replace('.pth', '_phase1.pth'),
            'transition': checkpoint_path.replace('.pth', '_transition.pth'),
            'phase2': checkpoint_path.replace('.pth', '_phase2.pth')
        }
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    def __call__(
            self, val_loss, current_phase,
            geom_model, geom_align,
            seq_model, seq_align, seq_pool,
            Geom_Ref, Seq_Ref,
            optimizer, epoch
    ):
        if self.phase_best_loss[current_phase] is None:
            self.phase_best_loss[current_phase] = val_loss
            self.save_checkpoint(
                current_phase,
                geom_model, geom_align,
                seq_model, seq_align, seq_pool,
                Geom_Ref, Seq_Ref,
                optimizer, val_loss, epoch
            )
            print(f'[EarlyStopping] First checkpoint for {current_phase}, val_loss={val_loss:.4f}')
        elif val_loss < self.phase_best_loss[current_phase] - self.delta:
            self.phase_best_loss[current_phase] = val_loss
            self.save_checkpoint(
                current_phase,
                geom_model, geom_align,
                seq_model, seq_align, seq_pool,
                Geom_Ref, Seq_Ref,
                optimizer, val_loss, epoch
            )
            self.counter = 0
            print(f'[EarlyStopping] {current_phase} val_loss improved to {val_loss:.4f}')
        else:
            if current_phase == 'phase2':
                self.counter += 1
                print(f'[EarlyStopping] {current_phase} counter: {self.counter} / {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
                    print(f'[EarlyStopping] Early stopping triggered in {current_phase}')
            else:
                print(f'[EarlyStopping] {current_phase} val_loss did not improve, but early stopping disabled in this phase')

    def save_checkpoint(
            self, phase,
            geom_model, geom_align,
            seq_model, seq_align, seq_pool,
            Geom_Ref, Seq_Ref,
            optimizer, val_loss, epoch
    ):
        checkpoint_dict = {
            'epoch': epoch,
            'val_loss': val_loss,
            'geom_model_state_dict': geom_model.state_dict(),
            'geom_align_state_dict': geom_align.state_dict(),
            'seq_model_state_dict': seq_model.state_dict(),
            'seq_align_state_dict': seq_align.state_dict(),
            'seq_pool_state_dict': seq_pool.state_dict(),
            'Geom_Ref_state_dict': Geom_Ref.state_dict(),
            'Seq_Ref_state_dict': Seq_Ref.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }

        phase_path = self.phase_checkpoint_path[phase]
        torch.save(checkpoint_dict, phase_path)
        print(f'[EarlyStopping] {phase} val_loss={val_loss:.4f}, saved to {phase_path}')

        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            torch.save(checkpoint_dict, self.checkpoint_path)
            print(f'[EarlyStopping] Global best model updated, val_loss={val_loss:.4f}, saved to {self.checkpoint_path}')

    def load_checkpoint(
            self,
            geom_model, geom_align,
            seq_model, seq_align, seq_pool,
            Geom_Ref, Seq_Ref,
            optimizer=None,
            phase=None
    ):

        path_to_load = self.phase_checkpoint_path.get(phase, self.checkpoint_path) if phase else self.checkpoint_path

        if not os.path.exists(path_to_load):
            print(f"[EarlyStopping] No checkpoint found at {path_to_load}, skip loading.")
            return

        ckpt = torch.load(path_to_load, map_location='cpu')
        geom_model.load_state_dict(ckpt['geom_model_state_dict'])
        geom_align.load_state_dict(ckpt['geom_align_state_dict'])
        seq_model.load_state_dict(ckpt['seq_model_state_dict'])
        seq_align.load_state_dict(ckpt['seq_align_state_dict'])
        seq_pool.load_state_dict(ckpt['seq_pool_state_dict'])
        Geom_Ref.load_state_dict(ckpt['Geom_Ref_state_dict'])
        Seq_Ref.load_state_dict(ckpt['Seq_Ref_state_dict'])

        if optimizer is not None and 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        source = f"{phase} phase" if phase else "global best"
        print(f'[EarlyStopping] Loaded {source} checkpoint from {path_to_load}. '
              f'epoch={ckpt["epoch"]}, val_loss={ckpt["val_loss"]:.4f}')