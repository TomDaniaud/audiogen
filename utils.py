import os, re
import torch




def save_best_checkpoint(model, optimizer, loss, epoch, dir='checkpoints'):
    if not os.path.exists(dir):
        os.makedirs(dir)
    # Cherche tous les checkpoints vae_epoch*_loss*.pth
    best_loss = float('inf')
    pattern = re.compile(r'vae_epoch(\d+)_loss([\d\.]+)\.pth')
    for fname in os.listdir(dir):
        match = pattern.match(fname)
        if match:
            ckpt_loss = float(match.group(2))
            if ckpt_loss < best_loss:
                best_loss = ckpt_loss
    # Compare la loss courante au meilleur checkpoint
    if best_loss == float('inf') or loss < best_loss:
        epoch_name = f"vae_epoch{epoch}_loss{loss:.4f}.pth"
        path = os.path.join(dir, epoch_name)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, path)
        # print(f"Checkpoint sauvegardé ({epoch_name}, loss={loss:.4f})")
    else:
        pass
        # print(f"Checkpoint NON sauvegardé (loss={loss:.4f} >= best_loss={best_loss:.4f})")
    

def load_best_checkpoint(model, optimizer, checkpoints_dir='checkpoints'):
    """
    Charge le checkpoint avec la plus petite loss (si les fichiers sont nommés avec la loss, ex: vae_checkpoint_epochX_lossY.pth)
    et return l'epoch actuel associé à la loss
    """
    best_loss = float('inf')
    best_path = None
    pattern = re.compile(r'\w+_epoch(\d+)_loss([\d\.]+)\.pth')
    best_epoch = 0
    for fname in os.listdir(checkpoints_dir):
        match = pattern.match(fname)
        if match:
            loss = float(match.group(2))
            if loss < best_loss:
                best_loss = loss
                best_path = os.path.join(checkpoints_dir, fname)
                best_epoch = int(match.group(1))
    if best_path:
        checkpoint = torch.load(best_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f'Best checkpoint chargé: {best_path} (loss={best_loss})')
        return best_epoch
    else:
        print('Aucun checkpoint avec loss trouvé.')
        return 0


def save_checkpoint(model, optimizer, path='checkpoints/vae.pth'):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)


def load_checkpoint(model, optimizer, path):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('Checkpoint chargé.')
    else:
        print('Aucun checkpoint trouvé, entraînement à partir de zéro.')
