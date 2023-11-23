import torch
from time import time
import torch.distributed as dist
from collections import OrderedDict


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)



def train_one_epoch(vae, diffusion, model, ema, data_loader,
                    device, opt, logger, train_steps, log_steps,
                    running_loss, start_time, rank, checkpoint_dir, args):

    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            # Map input images to latent space + normalize latents:
            x = vae.encode(x).latent_dist.sample().mul_(0.18215)

        t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
        model_kwargs = dict(y=y)
        loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
        loss = loss_dict["loss"].mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        update_ema(ema, model.module)

        # Log loss values:
        running_loss += loss.item()
        log_steps += 1
        train_steps += 1

        if train_steps % args.log_every == 0:
            # Measure training speed:
            torch.cuda.synchronize()
            end_time = time()
            steps_per_sec = log_steps / (end_time - start_time)
            # Reduce loss history over all processes:
            avg_loss = torch.tensor(running_loss / log_steps, device=device)
            dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
            avg_loss = avg_loss.item() / dist.get_world_size()
            logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
            # Reset monitoring variables:
            running_loss = 0
            log_steps = 0
            start_time = time()

        # Save DiT checkpoint:
        if train_steps % args.ckpt_every == 0 and train_steps > 0:
            if rank == 0:
                checkpoint = {
                    "model": model.module.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "args": args
                }
                checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
            dist.barrier()